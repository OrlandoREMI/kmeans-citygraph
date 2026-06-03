import matplotlib
matplotlib.use('Agg')
import pandas as pd
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ox.settings.use_cache = True

# ── 1. CARGAR DATOS ──────────────────────────────────────────────
print("Cargando base de datos...")
df = pd.read_csv('iieg_2023.csv')
df = df.dropna(subset=['x', 'y'])
print(f"Total incidentes originales: {len(df)}")

# ── 2. OBTENER POLÍGONO REAL DE ZAPOPAN ──────────────────────────
print("Descargando límite de Zapopan...")
zapopan_gdf = ox.geocode_to_gdf("Zapopan, Jalisco, Mexico")
poligono_zapopan = zapopan_gdf.geometry.iloc[0]

# ── 3. FILTRAR PUNTOS DENTRO DEL POLÍGONO ────────────────────────
print("Filtrando incidentes dentro de Zapopan...")
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df['x'], df['y']),
    crs='EPSG:4326'
)

df_zap = gdf[gdf.geometry.within(poligono_zapopan)].copy()
print(f"Incidentes dentro de Zapopan: {len(df_zap)}")
print(df_zap['delito'].value_counts())