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

@st.cache_data(ttl=600)  # Cache function for 10 minutes
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

# Get cached data
df = get_flight_data()


# Sidebar Navigation
st.sidebar.title("📍 Navigatie")
options = st.sidebar.radio("Ga naar", ['Statistiek',
                                       'vluchten per tijdstip geografische map (pydeck)', 
                                       "Geplande vs. Werkelijke landingstijden per vluchtmaatschappij",
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
                    "<b>Time:</b> {scheduleTime}" +
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
    tab1, tab2, tab3 = st.tabs(['Aantal vluchten', 'Vluchten per tijdstip', 'Interactieve plot'])
    with tab1:
        st.header('Aantal vluchten')
        vlucht1(df)
    with tab2:
        st.header('Vluchten per tijdstip')
        vlucht2(df)
    with tab3:
        st.header('Interactieve plot')
        vlucht3(df)
    
elif options == "Geplande vs. Werkelijke landingstijden per vluchtmaatschappij":
    vlucht4(df)
elif options == 'vluchten per tijdstip geografische map':
   # Definieer selected_time HIER met st.select_slider
    selected_time = st.select_slider("Kies een tijdstip", options=gdf["scheduleDateTime"].dropna().unique())
    vlucht5(gdf, selected_time)  # Nu is selected_time gedefinieerd


elif options == 'vluchten per tijdstip geografische map (pydeck)':
    # seperate page into 2 columns
    st.title("Flight Visualization with PyDeck")

    flight_deck = visualize_flights_from_schiphol(df, selected_time)

    # Create a container to hold the chart and legend side by side
    container = st.container()

    with container:
        col1, col2 = st.columns([3, 1])  # Adjust the ratio of widths as needed

        with col1:
            st.pydeck_chart(flight_deck)

        with col2:
            st.markdown(
                """
                ### Legend:
                - <span style="color:blue">Blue to Transparent</span>: Flights Departing from Schiphol
                - <span style="color:green">Green to Green</span>: Flights Arriving at Schiphol
                """,
                unsafe_allow_html=True,
                ow_html=True
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
