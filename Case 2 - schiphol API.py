import requests
import time
import streamlit as st
import plotly.express as px
import statsmodels.api as sm
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from shapely.geometry import Point
import matplotlib.patheffects as path_effects

import plotly.graph_objects as go

import pydeck as pdk

st.set_page_config(page_title='Schiphol API',  layout='wide', page_icon=':plane:')


# Schiphol API details
url_base = "https://api.schiphol.nl/public-flights/flights"
headers = {
    'accept': 'application/json',
    'resourceversion': 'v4',
    'app_id': 'b1ff0af7',
    'app_key': '43567fff2ced7e77e947c1b71ebc5f38'
}

@st.cache_data(ttl=3600)  # Cache function for 10 minutes
def get_flight_data():
    all_flights = []  # List to store all pages of flight data
    max_pages = 20  # Limit to 5 pages

    for page in range(max_pages):
        url = f"{url_base}?includedelays=false&page={page}&sort=%2BscheduleTime"
        response = requests.get(url, headers=headers)
        data = response.json()
        all_flights.extend(data["flights"])  # Append flights
        time.sleep(1)  # Prevent rate limits

    df = pd.DataFrame(all_flights)  # Convert to DataFrame

    # Data processing
    df['destination'] = df['route'].apply(lambda x: x.get('destinations', [None])[0])
    df['eu'] = df['route'].apply(lambda x: x.get('eu', [None]))
    df['visa'] = df['route'].apply(lambda x: x.get('visa', [None]))
    df['vlucht_status'] = df['publicFlightState'].apply(lambda x: ','.join(x['flightStates']) if 'flightStates' in x else None)
    df["baggage_belt"] = df["baggageClaim"].apply(
        lambda x: int(x["belts"][0]) if isinstance(x, dict) and "belts" in x and isinstance(x["belts"], list) and x["belts"] else None
    )
    df['iataMain'] = df['aircraftType'].apply(lambda x: x.get('iataMain', [None]))
    df['iataSub'] = df['aircraftType'].apply(lambda x: x.get('iataSub', [None]))

    # Merge with airport data
    Airports = pd.read_csv('world-airports.csv')
    Airports_clean = Airports.drop(columns=['home_link', 'wikipedia_link', 'scheduled_service', 'score', 'last_updated', 'elevation_ft', 'id', 'keywords'])
    df = df.merge(Airports_clean, how='left', left_on='destination', right_on='iata_code', suffixes=['_Port', '_Flight'])

    return df

def update_data():
    while True:
        new_data = get_flight_data()
        st.session_state['realtime_flight_data'] = new_data
        time.sleep(60)

# Get cached data
df = get_flight_data()

# Sidebar Navigation
st.sidebar.title("📍 Navigatie")
options = st.sidebar.radio("Ga naar", ['Statistiek',
                                       'Geografische map', 
                                       'Aanpassingen'])

def vlucht1(dataframe):
    st.header('Aantal vluchten')
    st.plotly_chart(px.histogram(dataframe, x="isOperationalFlight", width=400))

def vlucht2(dataframe):
    st.header('Vluchten per tijdstip')
    st.plotly_chart(px.histogram(dataframe, x="scheduleTime", width=600))

def vlucht3(dataframe):
    x_axis_value = st.selectbox('Selecteer de X-as', options=dataframe.columns)
    y_axis_value = st.selectbox('Selecteer de Y-as', options=dataframe.columns)
    show_trendline = st.checkbox("Toon trendlijn")

    # Controleer of een trendlijn mogelijk is
    if show_trendline:
        try:
            # Probeer de trendlijn te berekenen
            plot = px.scatter(dataframe, x=x_axis_value, y=y_axis_value, trendline="ols")
            col = st.color_picker("Kies een kleur")
            plot.update_traces(marker=dict(color=col))
            st.plotly_chart(plot)
        except Exception as e:  # Vang eventuele fouten op tijdens de berekening van de trendlijn
            st.write("Trendlijn niet mogelijk")  # Toon de melding
            st.write(f"Oorzaak: {e}") # toon de oorzaak van de error (handig bij debuggen)
            # Toon alsnog de scatterplot zonder trendlijn (optioneel)
            plot = px.scatter(dataframe, x=x_axis_value, y=y_axis_value)
            col = st.color_picker("Kies een kleur")
            plot.update_traces(marker=dict(color=col))
            st.plotly_chart(plot)
    else:
        # Toon de scatterplot zonder trendlijn
        plot = px.scatter(dataframe, x=x_axis_value, y=y_axis_value)
        col = st.color_picker("Kies een kleur")
        plot.update_traces(marker=dict(color=col))
        st.plotly_chart(plot)

def vlucht4(dataframe):
    st.header("Vertraging per luchtvaartmaatschappij")
 
    luchtvaartmaatschappijen = ["Alle maatschappijen"] + list(dataframe['prefixICAO'].unique())
    opties = st.selectbox("Selecteer een luchtvaartmaatschappij:", luchtvaartmaatschappijen)
 
    if opties == "Alle maatschappijen":
        df_filtered = dataframe.copy()
    else:
        df_filtered = dataframe[dataframe['prefixICAO'] == opties].copy()
 
    df_filtered['scheduleDateTime'] = pd.to_datetime(df_filtered['scheduleDateTime'])
    df_filtered['actualLandingTime'] = pd.to_datetime(df_filtered['actualLandingTime'])
 
    df_filtered['vertraging (min)'] = (df_filtered['actualLandingTime'] - df_filtered['scheduleDateTime']).dt.total_seconds() / 60
 
    # Verwijder rijen met NaN-waarden
    df_filtered = df_filtered.dropna(subset=['scheduleDateTime', 'vertraging (min)'])
 
    # Sorteer data
    df_filtered = df_filtered.sort_values(by='scheduleDateTime')
 
    # Zet tijd om naar numerieke waarden voor de regressie
    df_filtered['timestamp'] = df_filtered['scheduleDateTime'].astype('int64') // 10**9  
 
 
    # **Scatterplot maken**
    fig = px.scatter(
        df_filtered,
        x='scheduleDateTime',
        y='vertraging (min)',
        color='prefixICAO' if opties == "Alle maatschappijen" else 'vertraging (min)',  
        title=f'Vertraging in minuten per vlucht ({opties})',
        labels={'scheduleDateTime': 'Geplande landingstijd', 'vertraging (min)': 'Vertraging (min)', 'prefixICAO': 'Maatschappij'},
        hover_data=['flightNumber', 'scheduleDateTime', 'actualLandingTime'],
        opacity=0.7
    )
 
 
    # Polynomiale trendlijn (indien voldoende punten)**
    if len(df_filtered) > 10:  # Alleen toepassen als er genoeg data is
        coeffs = np.polyfit(df_filtered['timestamp'], df_filtered['vertraging (min)'], deg=3)
        poly = np.poly1d(coeffs)
        df_filtered['poly_trend'] = poly(df_filtered['timestamp'])
 
        fig.add_trace(go.Scatter(
            x=df_filtered['scheduleDateTime'],
            y=df_filtered['poly_trend'],
            mode='lines',
            name='Polynomiale Trendlijn',
            line=dict(color='blue', width=2, dash='dash')
        ))
 
    st.plotly_chart(fig)
    st.write(df)


# Voeg een nieuwe kolom toe die de vluchten nummerd op basis van 'scheduleDateTime'
df["flight_number"] = df["scheduleDateTime"].rank(method="first").astype(int)

# Filter nan waarde bij cordinaten
df = df.dropna(subset=["latitude_deg", "longitude_deg"])


# Schiphol Airport Coordinates
SCHIPHOL_LON = 4.763889
SCHIPHOL_LAT = 52.308611

def visualize_flights_from_schiphol(df, selected_time):
    """
    Visualizes flight paths to and from Schiphol using separate ArcLayers
    for departures (Blue fading out) and arrivals (Origin Green fading to Schiphol Green)
    using pydeck in Streamlit.

    Args:
        df (pd.DataFrame): DataFrame containing flight data with
                           'longitude_deg', 'latitude_deg', 'scheduleDateTime',
                           and 'flightDirection' ('A' or 'D'), and optionally
                           'destination' (for departures) or 'origin' (for arrivals).
        selected_time (str): The specific scheduleDateTime to visualize.
    """
    selected_flights = df[df["scheduleTime"] == selected_time].copy()
    if selected_flights.empty:
        st.warning(f"No flights found for the selected time: {selected_time}")
        return

    # Separate departing and arriving flights
    departing = selected_flights[selected_flights['flightDirection'] == 'D'].copy()
    arriving = selected_flights[selected_flights['flightDirection'] == 'A'].copy()

    # Prepare data for Departing ArcLayer (Blue to Transparent)
    departing['from'] = [[SCHIPHOL_LON, SCHIPHOL_LAT]] * len(departing)
    departing['to'] = departing.apply(
        lambda row: [row['longitude_deg'], row['latitude_deg']], axis=1
    )
    departing_arc_layer = pdk.Layer(
        "ArcLayer",
        data=departing,
        get_source_position="from",
        get_target_position="to",
        get_source_color=[0, 0, 255, 200],  # Blue for departing source (Schiphol)
        get_target_color=[0, 255, 0, 200],      # Transparent target for departing (Destination)
        auto_highlight=True,
        get_width=5
    )


    # Prepare data for Arriving ArcLayer (Origin Green to Schiphol Green)
    arriving['from'] = arriving.apply(
        lambda row: [row['longitude_deg'], row['latitude_deg']], axis=1
    )
    arriving['to'] = [[SCHIPHOL_LON, SCHIPHOL_LAT]] * len(arriving)
    arriving_arc_layer = pdk.Layer(
        "ArcLayer",
        data=arriving,
        get_source_position="from",
        get_target_position="to",
        get_source_color=[0, 0, 255, 200],  # Green for arriving source (Origin)
        get_target_color=[0, 255, 0, 200],  # Green target for arriving (Schiphol)
        auto_highlight=True,
        get_width=5
    )

    view_state = pdk.ViewState(
        latitude=SCHIPHOL_LAT,
        longitude=SCHIPHOL_LON,
        zoom=4,
        pitch=50,
    )

    # Create the PyDeck chart with both ArcLayers
    layers = []
    if not departing.empty:
        layers.append(departing_arc_layer)
    if not arriving.empty:
        layers.append(arriving_arc_layer)

    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip={
        "html": "<b>Arrival at Schiphol</b><br/>"
                "<b>Time:</b> {scheduleTime}<br/>"
                "<b>Airline:</b> {prefixICAO} {flightNumber}<br/>"
                "<b>Origin:</b> {origin}",
        "style": "background-color:steelblue; color:white; font-family: Arial;",
    },
        map_style="mapbox://styles/mapbox/dark-v10"
    )

    # Display the PyDeck chart in Streamlit
    st.pydeck_chart(r)

if options == 'Statistiek':

    st.title('Statistiek')
    tab1, tab2, tab3, tab4 = st.tabs(['Aantal vluchten', 'Vluchten per tijdstip', 'Interactieve plot', "Geplande vs. Werkelijke landingstijden"])
    with tab1:
        vlucht1(df)

    with tab2:

        vlucht2(df)
    with tab3:

        vlucht3(df)
    with tab4:

        vlucht4(df)

elif options == 'Geografische map':
    st.title("Flight Visualization with PyDeck")
    df['scheduleTime'] = df['scheduleTime'].astype(str)
    available_times = df['scheduleTime'].unique()

    selected_time = st.select_slider("Select a Time:", available_times)

    # Convert 'scheduleDateTime' to datetime objects if it's not already
    df['scheduleDateTime'] = pd.to_datetime(df['scheduleDateTime'])

    # Filter the DataFrame for the selected time to get the corresponding date
    selected_datetime = df[df['scheduleTime'] == selected_time]['scheduleDateTime'].iloc[0]
    selected_date = selected_datetime.strftime('%A, %B %d, %Y')

    st.subheader(f"Flights scheduled for: {selected_date} at {selected_time}")

    container = st.container()

    with container:
        col1, col2 = st.columns([1,0.3])  # Adjust the ratio of widths as needed

        with col1:
            flight_deck = visualize_flights_from_schiphol(df, selected_time)

        with col2:
            st.markdown(
                    """
                    ### Legend:
                    - <span style="color:blue">Blue</span>:     Departing Flights
                    - <span style="color:green">Green</span>:   Arriving Flights
                    """,
                    unsafe_allow_html=True
                )


elif options == 'Aanpassingen':
    st.title('Aanpassingen t.o.v. eerste versie')
    st.write('Als eerst hebben wij het bestand in een github repo gezet om makkelijk aanpassingen te maken en daarna is ook het kleuren thema veranderd.')
else:
    print("Ongeldige optie geselecteerd.")

# bronnen:
# https://www.naturalearthdata.com/downloads/110m-cultural-vectors/
# https://plotly.com/python/linear-fits/\
# https://plotly.com/python/histograms/
# https://docs.streamlit.io/develop/api-reference
