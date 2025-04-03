import requests
import time
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import pandas as pd
import numpy as np
import uuid 

# import geopandas as gpd
# import matplotlib.pyplot as plt
# import ipywidgets as widgets
# from shapely.geometry import Point
# import matplotlib.patheffects as path_effects
# import statsmodels.api as sm


st.set_page_config(page_title='Schiphol API',  layout='wide', page_icon=':plane:')


# Schiphol API details
url_base = "https://api.schiphol.nl/public-flights/flights"
headers = {
    'accept': 'application/json',
    'resourceversion': 'v4',
    'app_id': 'b1ff0af7',
    'app_key': '43567fff2ced7e77e947c1b71ebc5f38'
}

@st.cache_data(ttl=86400)  # Cache function for 1 uur minutes
def get_flight_data():
    all_flights = []  # List to store all pages of flight data
    max_pages = 20  # Limit to 200 pages

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

def vlucht3(df_filtered):
    # **Relevante kolommen filteren**
    possible_columns = [
        'scheduleDateTime', 'actualLandingTime', 'estimatedLandingTime',
        'prefixICAO',
        'flightNumber', 'flightDirection', 'serviceType',
        'baggage_belt', 'pier', 'gate', 'destination',
    ]
 
    # **Check welke kolommen echt bestaan in df_filtered**
    relevant_columns = [col for col in possible_columns if col in df_filtered.columns]
   
    if not relevant_columns:
        st.error("Geen relevante kolommen gevonden in de dataset!")
        return
   
    df_plot = df_filtered[relevant_columns].copy()
 
 
    # **Converteer datetime kolommen**
    datetime_cols = ['scheduleDateTime', 'actualLandingTime', 'estimatedLandingTime']
    for col in datetime_cols:
        if col in df_plot.columns:
            df_plot[col] = pd.to_datetime(df_plot[col], errors='coerce')
 
    # **Vertraging berekenen**
    if 'scheduleDateTime' in df_plot.columns and 'actualLandingTime' in df_plot.columns:
        df_plot['vertraging (min)'] = (df_plot['actualLandingTime'] - df_plot['scheduleDateTime']).dt.total_seconds() / 60
 
 
    # **Selecteer X- en Y-as**
    x_axis_value = st.selectbox('Selecteer de X-as', options=df_plot.columns)
    y_axis_value = st.selectbox('Selecteer de Y-as', options=df_plot.columns)
 
    # **Controle of kolommen numeriek zijn**
    x_is_numeric = pd.api.types.is_numeric_dtype(df_plot[x_axis_value])
    y_is_numeric = pd.api.types.is_numeric_dtype(df_plot[y_axis_value])
 
    # **Trendlijn optie**
    show_trendlijn = False
    if x_is_numeric and y_is_numeric:
        show_trendlijn = st.checkbox("Toon trendlijn")
 
 
    # **Maak de scatterplot**
    plot = px.scatter(df_plot, x=x_axis_value, y=y_axis_value, title="Interactieve Vlucht Plot")
 
    # **Kleur aanpassen**
    col = st.color_picker("Kies een kleur")
    plot.update_traces(marker=dict(color=col))
 
    st.plotly_chart(plot)

def vlucht4(dataframe):
    st.header("Aankomst afwijking per luchtvaartmaatschappij")
 
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
        title=f'Aankomst afwijking in minuten per vlucht ({opties})',
        labels={'scheduleDateTime': 'Geplande landingstijd', 'Afwijking (min)': 'Vertraging (min)', 'prefixICAO': 'Maatschappij'},
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

def vlucht_pertijdstip(dataframe):
    df_grouped = dataframe.groupby(['scheduleTime', 'flightDirection']).size().reset_index(name='count')
    st.plotly_chart(px.bar(df_grouped, x='scheduleTime', y='count', color='flightDirection', 
                            labels={'scheduleTime': 'Land', 'count': 'Aantal'}, width=600, height=400))

def vlucht_operationeel(dataframe):
    st.subheader('Is operationeel')
    df_grouped = dataframe.groupby(['isOperationalFlight', 'flightDirection']).size().reset_index(name='count')
    st.plotly_chart(px.bar(df_grouped, x='isOperationalFlight', y='count', color='flightDirection', 
                            labels={'isOperationalFlight': 'Land', 'count': 'Aantal'}, width=600, height=400))

def vlucht_land(dataframe):
    st.subheader("Aantal vluchten per land")
    df_grouped = dataframe.groupby(['country_name', 'flightDirection']).size().reset_index(name='count')
    st.plotly_chart(px.bar(df_grouped, x='country_name', y='count', color='flightDirection', 
                            labels={'country_name': 'Land', 'count': 'Aantal'}, width=600, height=400))

def vlucht_continent(dataframe):
    st.subheader("Aantal vluchten per continent")
    df_grouped = dataframe.groupby(['continent', 'flightDirection']).size().reset_index(name='count')
    st.plotly_chart(px.bar(df_grouped, x='continent', y='count', color='flightDirection', 
                            labels={'continent': 'Continent', 'count': 'Aantal'}, width=600, height=400))

def vlucht_bagageband(dataframe):
    # Count of baggage belts used
    st.subheader("Gebruik van bagagebanden")
    df_grouped = dataframe.groupby(['baggage_belt', 'flightDirection']).size().reset_index(name='count')
    st.plotly_chart(px.bar(df_grouped, x='baggage_belt', y='count', color='flightDirection', 
                           labels={'baggage_belt': 'Bagageband', 'count': 'Aantal'}, width=600, height=400))

def vlucht_statussen(dataframe):
    st.subheader("Aantal vluchten per status")
    df_grouped = dataframe.groupby(['vlucht_status', 'flightDirection']).size().reset_index(name='count')
    st.plotly_chart(px.bar(df_grouped, x='vlucht_status', y='count', color='flightDirection', 
                            labels={'vlucht_status': 'Status', 'count': 'Aantal'}, width=600, height=400))
    
def vlucht_statussen_info():
    st.subheader("Legenda: Vluchtstatussen")

    # Define flight statuses separately for Departing and Arriving flights
    departing_flights = {
        "SCH": "Flight scheduled",
        "DEL": "Delayed",
        "WIL": "Wait in Lounge",
        "GTO": "Gate Open",
        "BRD": "Boarding",
        "GCL": "Gate Closing",
        "GTD": "Gate Closed",
        "DEP": "Departed",
        "CNX": "Cancelled",
        "GCH": "Gate Change",
        "TOM": "Tomorrow"
    }

    arriving_flights = {
        "SCH": "Flight scheduled",
        "AIR": "Airborne",
        "EXP": "Expected Landing",
        "FIR": "Flight in Dutch airspace",
        "LND": "Landed",
        "FIB": "FIBAG",
        "ARR": "Arrived - Flight completely handled",
        "DIV": "Diverted",
        "CNX": "Cancelled",
        "TOM": "Tomorrow"
    }

    # Convert to DataFrames for Streamlit tables
    df_departing = pd.DataFrame(list(departing_flights.items()), columns=["Status", "Betekenis"])
    df_arriving = pd.DataFrame(list(arriving_flights.items()), columns=["Status", "Betekenis"])

    # Display in two columns for better readability
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("✈️ Vertrekkende Vluchten")
        st.dataframe(df_departing, hide_index=True, height=300, use_container_width=True)

    with col2:
        st.subheader("🛬 Aankomende Vluchten")
        st.dataframe(df_arriving, hide_index=True, height=300, use_container_width=True)

def vlucht_haventype(dataframe):
    # Count of flights per aircraft type
    st.subheader("Aantal vluchten per vliegtuigtype")
    df_grouby = dataframe.groupby(['type', 'flightdirection']).size().reset_index(name='count')
    st.plotly_chart(px.bar(df_grouby, x='type', y='count', color='flightDirection', 
                           labels={'type': 'Vliegtuigtype', 'count': 'Aantal'}, width=600, height=400))
    
def vlucht_pier(dataframe):
    # Count of flights per aircraft type
    st.subheader("Aantal vluchten per Schipholpier")
    df_grouby = dataframe.groupby(['pier', 'flightdirection']).size().reset_index(name='count')
    st.plotly_chart(px.bar(df_grouby, x='pier', y='count', color='flightDirection', 
                           labels={'pier': 'Schipholpier', 'count': 'Aantal'}, width=600, height=400))
    

def filter_flights(dataframe):
    flight_filter = st.radio("Selecteer vlucht type:", ["Alle", "Vertrekkend", "Aankomend"], index=0)
    
    if flight_filter == "Vertrekkend":
        return dataframe[dataframe['flightDirection'] == 'D']
    elif flight_filter == "Aankomend":
        return dataframe[dataframe['flightDirection'] == 'A']
    return dataframe


# Voeg een nieuwe kolom toe die de vluchten nummerd op basis van 'scheduleDateTime'
df["flight_number"] = df["scheduleDateTime"].rank(method="first").astype(int)

# Filter nan waarde bij cordinaten
df = df.dropna(subset=["latitude_deg", "longitude_deg"])


# Schiphol Airport Coordinates
SCHIPHOL_LON = 4.763889
SCHIPHOL_LAT = 52.308611

def visualize_flights_from_schiphol(df, selected_time=None):
    if selected_time:
        selected_flights = df[df["scheduleTime"] == selected_time].copy()
        if selected_flights.empty:
            st.warning(f"No flights found for the selected time: {selected_time}")
            return
    else:
        selected_flights = df.copy()

    # Separate departing and arriving flights
    departing = selected_flights[selected_flights['flightDirection'] == 'D'].copy()
    arriving = selected_flights[selected_flights['flightDirection'] == 'A'].copy()

    # Prepare data for Departing ArcLayer
    departing['from'] = [[SCHIPHOL_LON, SCHIPHOL_LAT]] * len(departing)
    departing['to'] = departing.apply(lambda row: [row['longitude_deg'], row['latitude_deg']], axis=1)
    departing_arc_layer = pdk.Layer(
        "ArcLayer",
        data=departing,
        get_source_position="from",
        get_target_position="to",
        get_source_color=[0, 0, 255, 100],
        get_target_color=[0, 255, 0, 100],
        auto_highlight=True,
        get_width=5,
        pickable=True,
    )

    # Prepare data for Arriving ArcLayer
    arriving['from'] = arriving.apply(lambda row: [row['longitude_deg'], row['latitude_deg']], axis=1)
    arriving['to'] = [[SCHIPHOL_LON, SCHIPHOL_LAT]] * len(arriving)
    arriving_arc_layer = pdk.Layer(
        "ArcLayer",
        data=arriving,
        get_source_position="from",
        get_target_position="to",
        get_source_color=[0, 0, 255, 100],
        get_target_color=[0, 255, 0, 100],
        auto_highlight=True,
        get_width=5,
        pickable=True,
    )

    # View state for PyDeck
    view_state = pdk.ViewState(
        latitude=SCHIPHOL_LAT,
        longitude=SCHIPHOL_LON,
        zoom=4,
        pitch=50,
    )

    # Tooltip configuration
    tooltip = {
        "html": """
            <b>Flight Information</b><br>
            <b>Time:</b> {scheduleTime}<br>
            <b>Airline:</b> {prefixICAO}<br>
            <b>Direction:</b> {flightDirection}<br>
            <b>To / From:</b> {country_name}
        """,
        "style": {
            "backgroundColor": "grey",
            "color": "white",
            "fontFamily": "Arial"
        }
    }

    # Create PyDeck chart
    layers = []
    if not departing.empty:
        layers.append(departing_arc_layer)
    if not arriving.empty:
        layers.append(arriving_arc_layer)

    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/dark-v10"
    )

    # Display in Streamlit
    st.pydeck_chart(r)

# Sidebar Navigation
st.sidebar.title("📍 Navigatie")
options = st.sidebar.radio("Ga naar", ['Home',
                                       'Statistiek',
                                       'Geografische map', 
                                       'Aanpassingen'])


if options == 'Home':
    st.title('Verbeterd API dashboard')

    st.subheader('Door meer streamlit functies, pydeck charts en statistieke analyse het dashboard verbeterd.')

    st.write(df)

if options == 'Statistiek':
    st.title('Statistiek')
    st.divider()

    df_filtered = filter_flights(df)

    tab1, tab2, tab3 = st.tabs(['Vluchten', "Geplande vs. Werkelijke landingstijden", 'Interactieve plot'])
    with tab1:
        container = st.container()
        with container:
            col1, col2 = st.columns([1,1])
            with col1:
                vlucht_pertijdstip(df_filtered)
            with col2:
                vlucht_operationeel(df_filtered)
            
            col3, col4 = st.columns([1,1])
            with col3:
                vlucht_land(df_filtered)
            with col4:
                vlucht_continent(df_filtered)
            
            col5, col6 = st.columns([1,1])
            with col5:
                vlucht_statussen(df_filtered)
            with col6:
                vlucht_statussen_info()
            col7, col8 = st.columns(2)
            with col7:
                vlucht_pier(df_filtered)
            with col8:
                vlucht_haventype(df_filtered)

elif options == 'Geografische map':
    st.title("Flight Visualization with PyDeck")

    st.divider()

    df['scheduleTime'] = df['scheduleTime'].astype(str)
    df['scheduleDateTime'] = pd.to_datetime(df['scheduleDateTime'])
    available_times = df['scheduleTime'].unique()

    display_option = st.segmented_control("Select Flight Display Mode:", ["All Flights", "By Time"])

    st.divider()

    container = st.container()

    with container:
        col1, col2 = st.columns([1,0.3])  # Adjust the ratio of widths as needed

        with col1:
            if display_option == "By Time":
                selected_time = st.select_slider("Select a Time:", available_times)
                selected_datetime = df[df['scheduleTime'] == selected_time]['scheduleDateTime'].iloc[0]
                selected_date = selected_datetime.strftime('%A, %B %d, %Y')
                st.write(f"**Selected Time:** {selected_date} at {selected_time}")
                visualize_flights_from_schiphol(df, selected_time)
            else:
                start_time = df['scheduleTime'].min()
                end_time = df['scheduleTime'].max()
                st.write(f"**Showing all flights from {start_time} to {end_time}.**")
                visualize_flights_from_schiphol(df)

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
    st.subheader("""Als eerst hebben wij het bestand in een github repo gezet om makkelijk aanpassingen te maken en daarna is ook het kleuren thema veranderd.\nDaarna hebben wij gekeken naar nieuwe manieren om data te visualiseren en welke data wij ter beschikking hadden.\nVan de geopandas plot een Pydeck Arcplot gemaakt. \nAndere data gevisualiseerd.\nNieuwe technieken gebruikt voor de layout, zoals tabs, containers en dividers.\nTekst groter gemaakt waar kon."""
                 )
else:
    print("Ongeldige optie geselecteerd.")

# bronnen:
# https://www.naturalearthdata.com/downloads/110m-cultural-vectors/
# https://plotly.com/python/linear-fits/\
# https://plotly.com/python/histograms/
# https://docs.streamlit.io/develop/api-reference
