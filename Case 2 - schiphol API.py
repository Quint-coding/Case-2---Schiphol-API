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
options = st.sidebar.radio("Ga naar", ['Aantal vluchten', 
                                       'Vluchten per tijdstip',
                                       'vluchten per tijdstip geografische map (pydeck)', 
                                       'Interactieve plot', 
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

def visualize_flights_schiphol_direction_with_legend(df, selected_time):
    """
    Visualizes flight paths to and from Schiphol with color-coded arcs
    based on flight direction (A for arrival, D for departure) using pydeck
    in Streamlit, and includes a legend.

    Args:
        df (pd.DataFrame): DataFrame containing flight data with
                           'longitude_deg', 'latitude_deg', 'scheduleDateTime',
                           and 'flightDirection' ('A' or 'D'), and optionally
                           'destination' (for departures) or 'origin' (for arrivals).
        selected_time (str): The specific scheduleDateTime to visualize.
    """
    selected_flights = df[df["scheduleDateTime"] == selected_time].copy()
    if selected_flights.empty:
        st.warning(f"No flights found for the selected time: {selected_time}")
        return

    # Separate departing and arriving flights
    departing = selected_flights[selected_flights['flightDirection'] == 'D'].copy()
    arriving = selected_flights[selected_flights['flightDirection'] == 'A'].copy()

    # Prepare data for ArcLayer - Departures (Blue)
    departing['from'] = [[SCHIPHOL_LON, SCHIPHOL_LAT]] * len(departing)
    departing['to'] = departing.apply(
        lambda row: [row['longitude_deg'], row['latitude_deg']], axis=1
    )

    # Prepare data for ArcLayer - Arrivals (Green)
    arriving['from'] = arriving.apply(
        lambda row: [row['longitude_deg'], row['latitude_deg']], axis=1
    )
    arriving['to'] = [[SCHIPHOL_LON, SCHIPHOL_LAT]] * len(arriving)

    arc_layer_departures = pdk.Layer(
        "ArcLayer",
        data=departing,
        get_source_position="from",
        get_target_position="to",
        get_source_color=[0, 0, 255, 200],  # Blue for departure from Schiphol
        get_target_color=[0, 0, 255, 200],
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

    arc_layer_arrivals = pdk.Layer(
        "ArcLayer",
        data=arriving,
        get_source_position="from",
        get_target_position="to",
        get_source_color=[0, 255, 0, 200],  # Green for arrival at Schiphol
        get_target_color=[0, 255, 0, 200],
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

    # Marker Layer for Destinations (Departures)
    marker_layer_departures = pdk.Layer(
        "ScatterplotLayer",
        data=departing,
        get_position="to",
        get_radius=10000,
        get_fill_color=[0, 0, 255, 200],  # Blue for departure destinations
        pickable=True,
        opacity=0.8,
        tooltip={
            "html": "<b>Destination:</b> [{to[0]:.2f}, {to[1]:.2f}]<br/>" +
                    ("<b>Name:</b> {destination}<br/>" if "destination" in departing.columns else "") +
                    "<b>Time:</b> {scheduleDateTime}",
            "style": "background-color:steelblue; color:white; font-family: Arial;",
        },
    )

    # Marker Layer for Origins (Arrivals)
    marker_layer_arrivals = pdk.Layer(
        "ScatterplotLayer",
        data=arriving,
        get_position="from",
        get_radius=10000,
        get_fill_color=[0, 255, 0, 200],  # Green for arrival origins
        pickable=True,
        opacity=0.8,
        tooltip={
            "html": "<b>Origin:</b> [{from[0]:.2f}, {from[1]:.2f}]<br/>" +
                    ("<b>Name:</b> {origin}<br/>" if "origin" in arriving.columns else "") +
                    "<b>Time:</b> {scheduleDateTime}",
            "style": "background-color:steelblue; color:white; font-family: Arial;",
        },
    )

    # Calculate bounds for initial view state
    all_locations = []
    if not departing.empty:
        all_locations.extend(departing['to'])
        all_locations.append([SCHIPHOL_LON, SCHIPHOL_LAT])
    if not arriving.empty:
        all_locations.extend(arriving['from'])
        all_locations.append([SCHIPHOL_LON, SCHIPHOL_LAT])

    if all_locations:
        lons = [loc[0] for loc in all_locations]
        lats = [loc[1] for loc in all_locations]
        min_lon = min(lons)
        max_lon = max(lons)
        min_lat = min(lats)
        max_lat = max(lats)
    else:
        min_lon, max_lon = SCHIPHOL_LON - 1, SCHIPHOL_LON + 1
        min_lat, max_lat = SCHIPHOL_LAT - 1, SCHIPHOL_LAT + 1

    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2
    initial_zoom = 4

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=initial_zoom,
        pitch=0,
    )

    # Create the PyDeck chart with all layers
    layers = []
    if not departing.empty:
        layers.extend([arc_layer_departures, marker_layer_departures])
    if not arriving.empty:
        layers.extend([arc_layer_arrivals, marker_layer_arrivals])

    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v10",
    )

    # Display the PyDeck chart in Streamlit
    st.pydeck_chart(r)

    # Add a legend using Streamlit's markdown
    st.markdown(
        """
        ### Legend:
        - <span style="color:blue">Blue</span>: Flights Departing from Schiphol
        - <span style="color:green">Green</span>: Flights Arriving at Schiphol
        """,
        unsafe_allow_html=True,
    )



if options == 'Aantal vluchten':
    vlucht1(df)
elif options == 'Vluchten per tijdstip':
    vlucht2(df)
elif options == 'Interactieve plot':
    vlucht3(df)
elif options == "Geplande vs. Werkelijke landingstijden per vluchtmaatschappij":
    vlucht4(df)
elif options == 'vluchten per tijdstip geografische map':
   # Definieer selected_time HIER met st.select_slider
    selected_time = st.select_slider("Kies een tijdstip", options=gdf["scheduleDateTime"].dropna().unique())
    vlucht5(gdf, selected_time)  # Nu is selected_time gedefinieerd


elif options == 'vluchten per tijdstip geografische map (pydeck)':
    # Convert scheduleDateTime to string for easy selection
    st.write(df)
    
    df['scheduleDateTime'] = df['scheduleDateTime'].astype(str)
    available_times = df['scheduleDateTime'].unique()

    st.title("Flight Visualization with PyDeck")
    selected_time = st.select_slider("Select a Time:", available_times)

    visualize_flights_from_schiphol(df, selected_time)
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
