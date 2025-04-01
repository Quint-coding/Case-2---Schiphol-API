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

st.title("Schiphol vluchten")
options = st.sidebar.radio('pages', options=['Aantal vluchten', 'Vluchten per tijdstip','vluchten per tijdstip geografische map', 'Interactieve plot', "Geplande vs. Werkelijke landingstijden per vluchtmaatschappij"])

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

#  GeoDataFrame rij maken van cordinaten en er een gpd van maken
df["geometry"] = [Point(xy) for xy in zip(df["longitude_deg"], df["latitude_deg"])]
gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

# Basis wereldkaart inladen bron: https://www.naturalearthdata.com/downloads/110m-cultural-vectors/
world = gpd.read_file('ne_50m_admin_0_countries.shp')
def vlucht5(gdf, selected_time):
    # Filter de vluchten op basis van de geselecteerde tijd
    selected_flights = gdf[gdf["scheduleDateTime"] == selected_time] # Gebruik gdf, niet dataframe

    # Maak een plot
    fig, ax = plt.subplots(figsize=(12, 6)) # fig en ax moeten binnen de functie worden aangemaakt
    world.plot(ax=ax, color="lightgray")  # Wereldkaart plotten

    # Fixeer assenlimieten
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect('equal')

    # Plot de geselecteerde vluchten
    selected_flights.plot(ax=ax, markersize=50, color="blue", alpha=0.6)

    # Voeg labels toe voor bestemmingen
    for _, row in selected_flights.iterrows():
        try: # try except toegevoegd voor het geval er geen 'destination' is
            text = ax.text(row.geometry.x, row.geometry.y, row['destination'], fontsize=9, ha='right', color='white', fontweight='bold')
            text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()])
        except KeyError:
            pass # doe niets als 'destination' niet bestaat

    # Toon de plot in Streamlit
    st.pyplot(fig)  # Correcte manier om de plot in Streamlit weer te geven






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
else:
    print("Ongeldige optie geselecteerd.")

# bronnen:
# https://www.naturalearthdata.com/downloads/110m-cultural-vectors/
# https://plotly.com/python/linear-fits/\
# https://plotly.com/python/histograms/
# https://docs.streamlit.io/develop/api-reference
