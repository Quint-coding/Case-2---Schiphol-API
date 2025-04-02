import streamlit as st
import pandas as pd
import pydeck as pdk
import time
import threading
import requests
import plotly.express as px
from datetime import datetime, timedelta, timezone

st.set_page_config(page_title='Schiphol API',  layout='wide', page_icon=':plane:')

# Schiphol API details
url_base = "https://api.schiphol.nl/public-flights/flights"
headers = {
    'accept': 'application/json',
    'resourceversion': 'v4',
    'app_id': 'b1ff0af7',
    'app_key': '43567fff2ced7e77e947c1b71ebc5f38'
}

@st.cache_data(ttl=60)  # Cache function for 1 minute
def get_latest_flight_data():
    all_flights = []  # List to store all flight data from the latest page

    # First, determine the total number of pages (assuming this info is available)
    # You might need to adjust this part based on how the API exposes total pages
    try:
        response_for_total = requests.get(f"{url_base}?includedelays=false&page=0&sort=%2BscheduleTime", headers=headers)
        response_for_total.raise_for_status()  # Raise an exception for bad status codes
        total_data = response_for_total.json()
        # Assuming the total number of pages is available in a key like 'totalPages'
        total_pages = total_data.get('totalPages', 1) # Default to 1 if not found
        latest_page = max(0, total_pages - 1) # Get the index of the last page
    except requests.exceptions.RequestException as e:
        print(f"Error fetching total page count: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of an error
    except KeyError:
        print("Could not determine total number of pages from the API response.")
        return pd.DataFrame()

    url = f"{url_base}?includedelays=false&page={latest_page}&sort=%2BscheduleTime"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        if "flights" in data:
            all_flights.extend(data["flights"])  # Append flights from the latest page
        else:
            print(f"No 'flights' data found on page {latest_page}.")
            return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from page {latest_page}: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pd.DataFrame()

    if not all_flights:
        return pd.DataFrame()  # Return an empty DataFrame if no flights were fetched

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

# Initialize session state for realtime data
if 'realtime_flight_data' not in st.session_state:
    st.session_state['realtime_flight_data'] = get_latest_flight_data()

# Get cached data (initial load)
df = st.session_state['realtime_flight_data']

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
    if show_trendline:
        try:
            plot = px.scatter(dataframe, x=x_axis_value, y=y_axis_value, trendline="ols")
            col = st.color_picker("Kies een kleur")
            plot.update_traces(marker=dict(color=col))
            st.plotly_chart(plot)
        except Exception as e:
            st.write("Trendlijn niet mogelijk")
            st.write(f"Oorzaak: {e}")
            plot = px.scatter(dataframe, x=x_axis_value, y=y_axis_value)
            col = st.color_picker("Kies een kleur")
            plot.update_traces(marker=dict(color=col))
            st.plotly_chart(plot)
    else:
        plot = px.scatter(dataframe, x=x_axis_value, y=y_axis_value)
        col = st.color_picker("Kies een kleur")
        plot.update_traces(marker=dict(color=col))
        st.plotly_chart(plot)

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
        hover_name='flightNumber', # Of een andere unieke ID
        hover_data=['scheduleDateTime', 'actualLandingTime']))

# Schiphol Airport Coordinates
SCHIPHOL_LON = 4.763889
SCHIPHOL_LAT = 52.308611

def visualize_flights_from_schiphol(df_vis, current_time_str):
    selected_flights = df_vis[df_vis["scheduleTime"] == current_time_str].copy()
    if selected_flights.empty:
        st.warning(f"No flights found for the selected time: {current_time_str}")
        return

    departing = selected_flights[selected_flights['flightDirection'] == 'D'].copy()
    arriving = selected_flights[selected_flights['flightDirection'] == 'A'].copy()

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
        get_target_color=[0, 255, 0, 50],      # Transparent target for departing (Destination)
        auto_highlight=True,
        width_scale=0.02,
        width_min_pixels=3,
        tooltip={
            "html": f"<b>Departure:</b> [{SCHIPHOL_LON:.2f}, {SCHIPHOL_LAT:.2f}] (Schiphol)<br/>"
                    "<b>Arrival:</b> [{to[0]:.2f}, {to[1]:.2f}]<br/>"
                    "<b>Time:</b> {scheduleDateTime}" +
                    ("<br/><b>Destination:</b> {destination}" if "destination" in departing.columns else ""),
            "style": "background-color:steelblue; color:white; font-family: Arial;",
        },
    )

    arriving['from'] = arriving.apply(
        lambda row: [row['longitude_deg'], row['latitude_deg']], axis=1
    )
    arriving['to'] = [[SCHIPHOL_LON, SCHIPHOL_LAT]] * len(arriving)
    arriving_arc_layer = pdk.Layer(
        "ArcLayer",
        data=arriving,
        get_source_position="from",
        get_target_position="to",
        get_source_color=[0, 255, 0, 200],  # Green for arriving source (Origin)
        get_target_color=[0, 0, 255, 50],  # Blue target for arriving (Schiphol)
        auto_highlight=True,
        width_scale=0.02,
        width_min_pixels=3,
        tooltip={
            "html": "<b>Departure:</b> [{from[0]:.2f}, {from[1]:.2f}]<br/>"
                    f"<b>Arrival:</b> [{SCHIPHOL_LON:.2f}, {SCHIPHOL_LAT:.2f}] (Schiphol)<br/>"
                    "<b>Time:</b> {scheduleDateTime}" +
                    ("<br/><b>Origin:</b> {origin}" if "origin" in arriving.columns else ""),
            "style": "background-color:steelblue; color:white; font-family: Arial;",
        },
    )

    view_state = pdk.ViewState(
        latitude=SCHIPHOL_LAT,
        longitude=SCHIPHOL_LON,
        zoom=4,
        pitch=50,
    )

    layers = []
    if not departing.empty:
        layers.append(departing_arc_layer)
    if not arriving.empty:
        layers.append(arriving_arc_layer)

    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v10",
    )
    return r

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
    st.title("Real-time Flight Visualization")
    if not df.empty:
        df_map = df.dropna(subset=["latitude_deg", "longitude_deg"]).copy()
        map_placeholder = st.empty()
        legend_placeholder = st.empty()

        while True:
            current_minute = datetime.now(timezone(timedelta(hours=2))).strftime("%H:%M") # Get current Amsterdam time in HH:MM
            updated_df = get_latest_flight_data()
            st.session_state['realtime_flight_data'] = updated_df

            # --- DEBUGGING ---
            st.subheader("Debugging Updated DataFrame (Inspect in Streamlit)")
            st.dataframe(updated_df)
            # --- END DEBUGGING ---

            if 'latitude_deg' in updated_df.columns and 'longitude_deg' in updated_df.columns:
                df_map_updated = updated_df.dropna(subset=["latitude_deg", "longitude_deg"]).copy()

                with map_placeholder.container():
                    st.subheader(f"Flights Scheduled Around {current_minute}")
                    flight_deck = visualize_flights_from_schiphol(df_map_updated, current_minute)
                    if flight_deck:
                        st.pydeck_chart(flight_deck)
                    else:
                        st.info(f"No flight data available for {current_minute}.")

                with legend_placeholder.container():
                    st.markdown(
                        """
                        ### Legend:
                        - <span style="color:blue">Blue</span>:     Departing Flights
                        - <span style="color:green">Green</span>:   Arriving Flights
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.error("Error: 'latitude_deg' or 'longitude_deg' columns are missing in the updated data.")

            time.sleep(60)
    else:
        st.info("No initial flight data available to display the map.")

elif options == 'Aanpassingen':
    st.title('Aanpassingen t.o.v. eerste versie')
    st.write('Als eerst hebben wij het bestand in een github repo gezet om makkelijk aanpassingen te maken en daarna is ook het kleuren thema veranderd.')
else:
    print("Ongeldige optie geselecteerd.")