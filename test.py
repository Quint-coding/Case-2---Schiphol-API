import streamlit as st
import pandas as pd
import pydeck as pdk
import time
import threading
import requests
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title='Schiphol API',  layout='wide', page_icon=':plane:')

# Schiphol API details
url_base = "https://api.schiphol.nl/public-flights/flights"
headers = {
    'accept': 'application/json',
    'resourceversion': 'v4',
    'app_id': 'b1ff0af7',
    'app_key': '43567fff2ced7e77e947c1b71ebc5f38'
}
# --- Data Fetching and Caching ---
@st.cache_data(ttl=60)  # Cache for 1 minute for real-time updates
def get_raw_realtime_flight_data():
    """Fetches raw real-time flight data from the API."""
    url = f"{url_base}?includedelays=false&sort=%2BscheduleTime"
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raise an exception for HTTP errors
    data = response.json()
    return pd.DataFrame(data["flights"])

@st.cache_data(ttl=3600)  # Cache function for 1 hour for the main dataset
def get_processed_flight_data():
    all_flights = []
    max_pages = 5  # Reduced for faster initial load

    for page in range(max_pages):
        url = f"{url_base}?includedelays=false&page={page}&sort=%2BscheduleTime"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        all_flights.extend(data["flights"])
        time.sleep(0.5)  # Reduced sleep

    df = pd.DataFrame(all_flights)
    return _process_flight_data(df)

def _process_flight_data(df):
    """Processes the raw flight data."""
    df['destination'] = df['route'].apply(lambda x: x.get('destinations', [None])[0] if isinstance(x, dict) else None)
    df['eu'] = df['route'].apply(lambda x: x.get('eu') if isinstance(x, dict) else None)
    df['visa'] = df['route'].apply(lambda x: x.get('visa') if isinstance(x, dict) else None)
    df['vlucht_status'] = df['publicFlightState'].apply(lambda x: ','.join(x['flightStates']) if isinstance(x, dict) and 'flightStates' in x and isinstance(x['flightStates'], list) else None)
    df["baggage_belt"] = df["baggageClaim"].apply(
        lambda x: x["belts"][0] if isinstance(x, dict) and "belts" in x and isinstance(x["belts"], list) and x["belts"] else None
    )
    df['iataMain'] = df['aircraftType'].apply(lambda x: x.get('iataMain') if isinstance(x, dict) else None)
    df['iataSub'] = df['aircraftType'].apply(lambda x: x.get('iataSub') if isinstance(x, dict) else None)
    df['flightDirection'] = df['flightName'].str[2] if 'flightName' in df.columns else None
    df['scheduleTime'] = df['scheduleDateTime'].str.split('T').str[1].str[:5] if 'scheduleDateTime' in df.columns else None
    return df

# --- Load Data ---
df_processed = get_processed_flight_data()


# --- Sidebar Navigation ---
st.sidebar.title("📍 Navigatie")
options = st.sidebar.radio("Ga naar", ['Statistiek',
                                       'Geografische map',
                                       'Aanpassingen'])

# --- Helper Functions for Statistiek Tab ---
def vlucht1(dataframe):
    st.header('Aantal vluchten')
    st.plotly_chart(px.histogram(dataframe, x="isOperationalFlight", width=400), use_container_width=True)

def vlucht2(dataframe):
    st.header('Vluchten per tijdstip')
    st.plotly_chart(px.histogram(dataframe, x="scheduleTime", width=600), use_container_width=True)

def vlucht3(dataframe):
    x_axis_value = st.selectbox('Selecteer de X-as', options=dataframe.columns)
    y_axis_value = st.selectbox('Selecteer de Y-as', options=dataframe.columns)
    show_trendline = st.checkbox("Toon trendlijn")

    if show_trendline:
        try:
            plot = px.scatter(dataframe, x=x_axis_value, y=y_axis_value, trendline="ols")
            col = st.color_picker("Kies een kleur")
            plot.update_traces(marker=dict(color=col))
            st.plotly_chart(plot, use_container_width=True)
        except Exception as e:
            st.warning("Trendlijn niet mogelijk voor deze data.")
            st.write(f"Oorzaak: {e}")
            plot = px.scatter(dataframe, x=x_axis_value, y=y_axis_value)
            col = st.color_picker("Kies een kleur")
            plot.update_traces(marker=dict(color=col))
            st.plotly_chart(plot, use_container_width=True)
    else:
        plot = px.scatter(dataframe, x=x_axis_value, y=y_axis_value)
        col = st.color_picker("Kies een kleur")
        plot.update_traces(marker=dict(color=col))
        st.plotly_chart(plot, use_container_width=True)

def vlucht4(dataframe):
    st.header("Geplande vs. Werkelijke landingstijden per vluchtmaatschappij")
    luchtvaartmaatschappijen = dataframe['prefixICAO'].unique()
    opties = st.selectbox("Selecteer een luchtvaartmaatschappij:", luchtvaartmaatschappijen)
    df_filtered = dataframe[dataframe['prefixICAO'] == opties]
    st.plotly_chart(px.scatter(
        df_filtered,
        x='scheduleDateTime',
        y='actualLandingTime',
        title=f'Geplande vs. Werkelijke landingstijden ({opties})',
        labels={'scheduleDateTime': 'Geplande tijd', 'actualLandingTime': 'Werkelijke tijd'},
        hover_name='flightNumber',
        hover_data=['scheduleDateTime', 'actualLandingTime']), use_container_width=True)

# --- Helper Functions for Geografische Map Tab ---
# Schiphol Airport Coordinates
SCHIPHOL_LON = 4.763889
SCHIPHOL_LAT = 52.308611

def visualize_flights(df_flights, airports_df, is_realtime=False):
    """Visualizes flight paths with ArcLayers."""
    if df_flights.empty:
        st.warning("No flight data to display.")
        return

    departing = df_flights[df_flights['flightDirection'] == 'D'].copy()
    arriving = df_flights[df_flights['flightDirection'] == 'A'].copy()

    departing['from'] = [[SCHIPHOL_LON, SCHIPHOL_LAT]] * len(departing)
    departing = departing.merge(airports_df[['iata_code', 'longitude_deg', 'latitude_deg']],
                                left_on='destination', right_on='iata_code', how='left')
    departing.rename(columns={'longitude_deg': 'lon_dest', 'latitude_deg': 'lat_dest'}, inplace=True)
    departing['to_coords'] = departing.apply(
        lambda row: [row['lon_dest'], row['lat_dest']] if pd.notna(row['lon_dest']) and pd.notna(row['lat_dest']) else None, axis=1
    )
    departing.dropna(subset=['to_coords'], inplace=True)

    arriving = arriving.merge(airports_df[['iata_code', 'longitude_deg', 'latitude_deg']],
                              left_on='destination', right_on='iata_code', how='left')
    arriving.rename(columns={'longitude_deg': 'lon_origin', 'latitude_deg': 'lat_origin'}, inplace=True)
    arriving['from_coords'] = arriving.apply(
        lambda row: [row['lon_origin'], row['lat_origin']] if pd.notna(row['lon_origin']) and pd.notna(row['lat_origin']) else None, axis=1
    )
    arriving.dropna(subset=['from_coords'], inplace=True)
    arriving['to'] = [[SCHIPHOL_LON, SCHIPHOL_LAT]] * len(arriving)

    departing_arc_layer = pdk.Layer(
        "ArcLayer",
        data=departing,
        get_source_position="from",
        get_target_position="to_coords",
        get_source_color=[0, 0, 255, 200],
        get_target_color=[0, 255, 0, 200],
        auto_highlight=True,
        width_scale=0.02,
        width_min_pixels=3,
        tooltip={
            "html": f"<b>Departure:</b> [{SCHIPHOL_LON:.2f}, {SCHIPHOL_LAT:.2f}] (Schiphol)<br/>"
                    "<b>Arrival:</b> [{to_coords[0]:.2f}, {to_coords[1]:.2f}]<br/>"
                    "<b>Time:</b> {scheduleDateTime}" +
                    ("<br/><b>Destination:</b> {destination}" if "destination" in departing.columns else ""),
            "style": "background-color:steelblue; color:white; font-family: Arial;",
        },
    )

    arriving_arc_layer = pdk.Layer(
        "ArcLayer",
        data=arriving,
        get_source_position="from_coords",
        get_target_position="to",
        get_source_color=[0, 0, 255, 200],
        get_target_color=[0, 255, 0, 200],
        auto_highlight=True,
        width_scale=0.02,
        width_min_pixels=3,
        tooltip={
            "html": "<b>Departure:</b> [{from_coords[0]:.2f}, {from_coords[1]:.2f}]<br/>"
                    f"<b>Arrival:</b> [{SCHIPHOL_LON:.2f}, {SCHIPHOL_LAT:.2f}] (Schiphol)<br/>"
                    "<b>Time:</b> {scheduleDateTime}" +
                    ("<br/><b>Origin:</b> {destination}" if "destination" in arriving.columns else ""),
            "style": "background-color:steelblue; color:white; font-family: Arial;",
        },
    )

    view_state = pdk.ViewState(
        latitude=SCHIPHOL_LAT,
        longitude=SCHIPHOL_LON,
        zoom=4,
        pitch=50,
    )

    layers = [departing_arc_layer, arriving_arc_layer]

    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v10",
    )
    st.pydeck_chart(r, use_container_width=True)

# --- Load Airport Data ---
Airports = pd.read_csv('world-airports.csv')
Airports_clean = Airports.drop(columns=['home_link', 'wikipedia_link', 'scheduled_service', 'score', 'last_updated', 'elevation_ft', 'id', 'keywords'])

# --- Main App Logic ---
if options == 'Statistiek':
    st.title('Statistiek')
    tab1, tab2, tab3, tab4 = st.tabs(['Aantal vluchten', 'Vluchten per tijdstip', 'Interactieve plot', "Geplande vs. Werkelijke landingstijden"])
    with tab1:
        vlucht1(df_processed)
    with tab2:
        vlucht2(df_processed)
    with tab3:
        vlucht3(df_processed)
    with tab4:
        vlucht4(df_processed)

elif options == 'Geografische map':
    st.title("Flight Visualization with PyDeck")
    geo_tab1, geo_tab2, geo_tab3 = st.tabs(["Huidige vluchten", "Real-time vluchten", "Laatste Minuut"])

    with geo_tab1:
        st.header("Huidige vluchten")
        df_processed['scheduleTime'] = df_processed['scheduleTime'].astype(str)
        available_times = df_processed['scheduleTime'].unique()
        selected_time = st.select_slider("Select a Time:", available_times)
        visualize_flights(df_processed[df_processed["scheduleTime"] == selected_time], Airports_clean)
        with st.expander("Legenda"):
            st.markdown(
                """
                ### Legenda:
                - <span style="color:blue">Blauw</span>: Vertrekkende Vluchten
                - <span style="color:green">Groen</span>: Aankomende Vluchten
                """,
                unsafe_allow_html=True
            )

    with geo_tab2:
        st.header("Real-time vluchten (Alle actuele vluchten)")
        realtime_df_raw = get_raw_realtime_flight_data()
        if not realtime_df_raw.empty:
            realtime_df = _process_flight_data(realtime_df_raw.copy()) # Process the real-time data
            visualize_flights(realtime_df, Airports_clean, is_realtime=True)
        else:
            st.info("Momenteel geen real-time vluchtdata beschikbaar.")
        with st.expander("Legenda"):
            st.markdown(
                """
                ### Legenda:
                - <span style="color:blue">Blauw</span>: Vertrekkende Vluchten
                - <span style="color:green">Groen</span>: Aankomende Vluchten
                """,
                unsafe_allow_html=True
            )

    with geo_tab3:
        st.header("Vluchten van de laatste minuut")
        now_utc = datetime.utcnow()
        one_minute_ago_utc = now_utc - timedelta(minutes=1)

        realtime_df_raw_minute = get_raw_realtime_flight_data()

        if not realtime_df_raw_minute.empty:
            # Process the real-time data
            realtime_df_minute_processed = _process_flight_data(realtime_df_raw_minute.copy())

            # Convert scheduleTime to datetime objects (assuming UTC)
            realtime_df_minute_processed['scheduleDateTime_dt'] = pd.to_datetime(realtime_df_minute_processed['scheduleDateTime'], utc=True)

            # Filter flights scheduled within the last minute
            recent_flights = realtime_df_minute_processed[realtime_df_minute_processed['scheduleDateTime_dt'] >= one_minute_ago_utc]

            if not recent_flights.empty:
                visualize_flights(recent_flights, Airports_clean, is_realtime=True)
            else:
                st.info("Geen vluchten gevonden die in de afgelopen minuut gepland waren.")
        else:
            st.info("Momenteel geen real-time vluchtdata beschikbaar.")
        with st.expander("Legenda"):
            st.markdown(
                """
                ### Legenda:
                - <span style="color:blue">Blauw</span>: Vertrekkende Vluchten
                - <span style="color:green">Groen</span>: Aankomende Vluchten
                """,
                unsafe_allow_html=True
            )

elif options == 'Aanpassingen':
    st.title('Aanpassingen t.o.v. eerste versie')
    st.write('Als eerst hebben wij het bestand in een github repo gezet om makkelijk aanpassingen te maken en daarna is ook het kleuren thema veranderd.')
else:
    print("Ongeldige optie geselecteerd.")