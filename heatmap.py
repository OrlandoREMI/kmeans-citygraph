import pandas as pd
import folium
from folium.plugins import HeatMap
import webbrowser

# 🔹 1. Cargar datos
df = pd.read_csv("iieg_2023.csv")

df["x"] = pd.to_numeric(df["x"], errors="coerce")
df["y"] = pd.to_numeric(df["y"], errors="coerce")
df = df.dropna(subset=["x", "y"])

print("Puntos:", len(df))

# 🔹 2. Crear mapa
mapa = folium.Map(
    location=[20.67, -103.35],
    zoom_start=12,
    tiles=None
)

folium.TileLayer("CartoDB positron").add_to(mapa)

# 🔥 3. Heatmap
HeatMap(
    df[["y", "x"]].values.tolist(),
    radius=8
).add_to(mapa)

# 🔹 4. Guardar
mapa.save("mapa.html")
webbrowser.open("mapa.html")