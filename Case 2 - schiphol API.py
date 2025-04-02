import requests
import time
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import pandas as pd
import numpy as np

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


if options == 'Statistiek':

    st.title('Statistiek')
    tab1, tab2, tab3 = st.tabs(['Vluchten', "Geplande vs. Werkelijke landingstijden", 'Interactieve plot'])
    with tab1:
        container = st.container()

    with container:
        col1, col2 = st.columns([1,1])  # Adjust the ratio of widths as needed

        with col1:
            vlucht2(df)

        with col2:
            vlucht1(df)

    with tab2:
        vlucht4(df)

    with tab3:
        vlucht3(df)

elif options == 'Geografische map':
    st.title("Flight Visualization with PyDeck")

    df['scheduleTime'] = df['scheduleTime'].astype(str)
    available_times = df['scheduleTime'].unique()

    display_option = st.radio("Select Flight Display Mode:", ["All Flights", "By Time"], horizontal=True)

    # selected_time = st.select_slider("Select a Time:", available_times)

    # # Convert 'scheduleDateTime' to datetime objects if it's not already
    # df['scheduleDateTime'] = pd.to_datetime(df['scheduleDateTime'])

    # # Filter the DataFrame for the selected time to get the corresponding date
    # selected_datetime = df[df['scheduleTime'] == selected_time]['scheduleDateTime'].iloc[0]
    # selected_date = selected_datetime.strftime('%A, %B %d, %Y')

    # st.subheader(f"Flights scheduled for: {selected_date} at {selected_time}")

    container = st.container()

    with container:
        col1, col2 = st.columns([1,0.3])  # Adjust the ratio of widths as needed

        with col1:
            if display_option == "By Time":
                selected_time = st.select_slider("Select a Time:", available_times)
                df['scheduleDateTime'] = pd.to_datetime(df['scheduleDateTime'])
                selected_datetime = df[df['scheduleTime'] == selected_time]['scheduleDateTime'].iloc[0]
                selected_date = selected_datetime.strftime('%A, %B %d, %Y')
                st.write(f"**Selected Time:** {selected_date} at {selected_time}")
                visualize_flights_from_schiphol(df, selected_time)
            else:
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
    st.write('Als eerst hebben wij het bestand in een github repo gezet om makkelijk aanpassingen te maken en daarna is ook het kleuren thema veranderd.')
else:
    print("Ongeldige optie geselecteerd.")

# bronnen:
# https://www.naturalearthdata.com/downloads/110m-cultural-vectors/
# https://plotly.com/python/linear-fits/\
# https://plotly.com/python/histograms/
# https://docs.streamlit.io/develop/api-reference
