import requests
import time
import streamlit as st
import plotly.express as px
import statsmodels.api as sm
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from shapely.geometry import Point
import matplotlib.patheffects as path_effects

import pydeck as pdk

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
    all_flights = []  # List to store all pages of flight data
    total_pages = 1

    # First, attempt to get the total number of pages from the API
    try:
        response_for_total = requests.get(f"{url_base}?includedelays=false&page=0&sort=%2BscheduleTime", headers=headers)
        response_for_total.raise_for_status()
        total_data = response_for_total.json()
        total_pages = total_data.get('totalPages', 1)
        print(f"Total pages found: {total_pages}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching total page count: {e}")
        return pd.DataFrame()
    except KeyError:
        print("Could not find 'totalPages' key in the API response. Will attempt to go through pages sequentially.")
        # In this case, we'll proceed without knowing the exact number of pages
        pass

    # Iterate through all the pages
    for page in range(total_pages if 'totalPages' in locals() else 50): # Set a reasonable upper limit if totalPages is not found
        url = f"{url_base}?includedelays=false&page={page}&sort=%2BscheduleTime"
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            if "flights" in data:
                all_flights.extend(data["flights"])
            else:
                print(f"No 'flights' data found on page {page}. Stopping page iteration.")
                break # Stop if no flights are found on a page
            if 'totalPages' not in locals() and len(data.get('flights', [])) == 0:
                print(f"No flights found on page {page}, assuming this is the end.")
                break # Stop if we didn't get totalPages and find an empty page
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from page {page}: {e}")
            break # Stop on error
        except Exception as e:
            print(f"An unexpected error occurred on page {page}: {e}")
            break # Stop on error
        time.sleep(0.1) # Be gentler with the API

        # If we didn't get totalPages initially, check if the current page seems to be the last
        if 'totalPages' not in locals() and len(data.get('flights', [])) < data.get('pageSize', 50): # Adjust 'pageSize' if your API uses a different default
            print(f"Assuming page {page} is the last page based on the number of results.")
            break

    if not all_flights:
        return pd.DataFrame()

    df = pd.DataFrame(all_flights)

    # Sort the DataFrame by 'scheduleTime' to ensure the last entries are indeed the latest
    df = df.sort_values(by='scheduleTime', ascending=False)

    # Take the first 5 rows (which are now the last 5 based on scheduleTime)
    df_latest_5 = df.head(5).copy() # Use .copy() to avoid SettingWithCopyWarning

    # Data processing for the final 5 (same as before)
    df_latest_5['destination'] = df_latest_5['route'].apply(lambda x: x.get('destinations', [None])[0])
    df_latest_5['eu'] = df_latest_5['route'].apply(lambda x: x.get('eu', [None]))
    df_latest_5['visa'] = df_latest_5['route'].apply(lambda x: x.get('visa', [None]))
    df_latest_5['vlucht_status'] = df_latest_5['publicFlightState'].apply(lambda x: ','.join(x['flightStates']) if 'flightStates' in x else None)
    df_latest_5["baggage_belt"] = df_latest_5["baggageClaim"].apply(
        lambda x: int(x["belts"][0]) if isinstance(x, dict) and "belts" in x and isinstance(x["belts"], list) and x["belts"] else None
    )
    df_latest_5['iataMain'] = df_latest_5['aircraftType'].apply(lambda x: x.get('iataMain', [None]))
    df_latest_5['iataSub'] = df_latest_5['aircraftType'].apply(lambda x: x.get('iataSub', [None]))

    # Merge with airport data
    Airports = pd.read_csv('world-airports.csv')
    Airports_clean = Airports.drop(columns=['home_link', 'wikipedia_link', 'scheduled_service', 'score', 'last_updated', 'elevation_ft', 'id', 'keywords'])
    df = df_latest_5.merge(Airports_clean, how='left', left_on='destination', right_on='iata_code', suffixes=['_Port', '_Flight'])

    return df

def update_data():
    while True:
        new_data = get_latest_flight_data()
        st.session_state['realtime_flight_data'] = new_data
        time.sleep(60)

# Get cached data
df = get_latest_flight_data()

st.set_page_config(page_title='Schiphol API',  layout='wide', page_icon=':plane:')

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

    # Create the PyDeck chart with both ArcLayers
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
