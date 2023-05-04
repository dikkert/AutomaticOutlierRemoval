import requests
import folium

# Set the URL of the web feature server
url = "https://geodata.nationaalgeoregister.nl/wijkenbuurten2021/wfs"

# Define the parameters for the request
params = {
    "service": "WFS",
    "version": "2.0.0",
    "request": "GetFeature",
    "typeNames": "buurt:buurt",
    "outputFormat": "json",
    "cql_filter": "buurt:buurttype='PARK'",
}

# Send the request and get the response
response = requests.get(url, params=params)

# Convert the response to JSON
data = response.json()

# Create a map centered on Rotterdam
m = folium.Map(location=[51.92, 4.48], zoom_start=12, tiles="Stamen Toner")

# Add each park to the map as a black and white circle
for feature in data["features"]:
    coords = feature["geometry"]["coordinates"]
    folium.CircleMarker(location=[coords[1], coords[0]], radius=5, color="black", fill_color="white").add_to(m)

# Save the map as an HTML file
m.save("parks.html")