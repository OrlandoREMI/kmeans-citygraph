import matplotlib
matplotlib.use('Agg')

import pandas as pd
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

ox.settings.use_cache = True

# ── 1. CARGAR DATOS ──────────────────────────────────────────────
print("Cargando base de datos...")
df = pd.read_csv('iieg_2023.csv')
df = df.dropna(subset=['x', 'y'])
print(f"Total incidentes: {len(df):,}")

# DEBUG rápido
print("Rango X:", df['x'].min(), df['x'].max())
print("Rango Y:", df['y'].min(), df['y'].max())

# ── 2. CARGAR MAPAS ──────────────────────────────────────────────
print("Cargando red vial...")
G_gdl = ox.graph_from_place("Guadalajara, Jalisco, Mexico", network_type='drive')
G_zap = ox.graph_from_place("Zapopan, Jalisco, Mexico", network_type='drive')
G = nx.compose(G_gdl, G_zap)

# ── 3. LÍMITES ───────────────────────────────────────────────────
print("Descargando límites...")
gdl_gdf = ox.geocode_to_gdf("Guadalajara, Jalisco, Mexico")
zap_gdf = ox.geocode_to_gdf("Zapopan, Jalisco, Mexico")

# ── 4. COLORES ───────────────────────────────────────────────────
colores_delito = {
    'homicidio doloso':              '#FF0000',
    'feminicidio':                   '#FF00FF',
    'violacion':                     '#FF6600',
    'abuso sexual infantil':         '#FF9900',
    'violencia familiar':            '#FFD700',
    'lesiones dolosas':              '#FF4444',
    'robo a persona':                '#00BFFF',
    'robo a negocio':                '#1E90FF',
    'robo a casa habitacion':        '#4169E1',
    'robo a vehiculos particulares': '#00CED1',
    'robo de motocicleta':           '#20B2AA',
    'robo de autopartes':            '#3CB371',
    'robo a int de vehiculos':       '#90EE90',
    'robo a bancos':                 '#8B0000',
    'robo a carga pesada':           '#A0522D',
    'robo a cuentahabientes':        '#DDA0DD',
}

# ── 5. FIGURA ────────────────────────────────────────────────────
print("Generando mapa...")
fig, ax = plt.subplots(figsize=(26, 26), facecolor='#1a1a2e')
ax.set_facecolor('#1a1a2e')

# Red vial
ox.plot_graph(
    G, ax=ax, show=False, close=False,
    bgcolor='none',
    edge_color='#444466',
    edge_linewidth=0.3,
    node_size=0
)

# Límites
gdl_gdf.boundary.plot(ax=ax, color='#FFFFFF', linewidth=1.2,
                      linestyle='--', alpha=0.5, zorder=4)
zap_gdf.boundary.plot(ax=ax, color='#AAAAFF', linewidth=1.2,
                      linestyle='--', alpha=0.5, zorder=4)

# ── 6. FORZAR LÍMITES (CLAVE) ────────────────────────────────────
combined = gpd.GeoSeries(
    pd.concat([gdl_gdf.geometry, zap_gdf.geometry], ignore_index=True)
)
minx, miny, maxx, maxy = combined.total_bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)

# ── 7. PUNTOS ────────────────────────────────────────────────────
delitos_ordenados = df['delito'].value_counts().index

for delito in delitos_ordenados:
    subset = df[df['delito'] == delito]
    color  = colores_delito.get(delito, '#FFFFFF')

    # ⚠️ CAMBIA AQUÍ SI ESTÁN INVERTIDOS
    ax.scatter(
        subset['x'],   # prueba cambiar a subset['y']
        subset['y'],   # prueba cambiar a subset['x']
        c=color,
        s=5,
        alpha=0.6,
        linewidths=0,
        zorder=5
    )

# ── 8. LEYENDA ───────────────────────────────────────────────────
handles = []
for delito in delitos_ordenados:
    count = len(df[df['delito'] == delito])
    color = colores_delito.get(delito, '#FFFFFF')
    handles.append(mpatches.Patch(color=color,
                   label=f"{delito.title()}  ({count:,})"))

handles += [
    mpatches.Patch(color='none', label=''),
    mpatches.Patch(facecolor='none', edgecolor='#FFFFFF',
                   linestyle='--', linewidth=1.5, label='Guadalajara'),
    mpatches.Patch(facecolor='none', edgecolor='#AAAAFF',
                   linestyle='--', linewidth=1.5, label='Zapopan'),
]

legend = ax.legend(handles=handles, loc='lower left', fontsize=10,
                   framealpha=0.85, facecolor='#0d0d1a',
                   edgecolor='#555577', labelcolor='white',
                   title='Tipo de delito', title_fontsize=12)
legend.get_title().set_color('white')

# ── 9. ETIQUETAS ─────────────────────────────────────────────────
ax.text(gdl_gdf.geometry.iloc[0].centroid.x,
        gdl_gdf.geometry.iloc[0].centroid.y,
        'GUADALAJARA', fontsize=13, fontweight='bold',
        color='white', alpha=0.5, ha='center', zorder=6)

ax.text(zap_gdf.geometry.iloc[0].centroid.x,
        zap_gdf.geometry.iloc[0].centroid.y,
        'ZAPOPAN', fontsize=13, fontweight='bold',
        color='#AAAAFF', alpha=0.5, ha='center', zorder=6)

# ── 10. TÍTULO ───────────────────────────────────────────────────
ax.set_title("Incidentes Delictivos — Guadalajara & Zapopan 2023",
             fontsize=18, fontweight='bold', color='white', pad=14)

ax.annotate(f"Total incidentes: {len(df):,}  |  Fuente: IIEG Jalisco 2023",
            xy=(0.5, 0.01), xycoords='axes fraction',
            ha='center', fontsize=9, color='#AAAACC')

ax.set_axis_off()

fig.savefig("mapa_delitos_gdl_zapopan.png", dpi=150,
            bbox_inches='tight', facecolor='#1a1a2e')

plt.close(fig)
print("¡Listo!")