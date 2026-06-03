import matplotlib
matplotlib.use('Agg')
import pandas as pd
import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ox.settings.use_cache = True

# ── 1. CARGAR DATOS ──────────────────────────────────────────────
print("Cargando base de datos...")
df = pd.read_csv('iieg_2023.csv')
df = df.dropna(subset=['x', 'y'])
print(f"Total de incidentes: {len(df)}")
print("Delitos disponibles:", df['delito'].unique())

# ── 2. CARGAR MAPA DE GUADALAJARA ────────────────────────────────
print("Cargando mapa de Guadalajara...")
G = ox.graph_from_place("Guadalajara, Jalisco, Mexico", network_type='drive')

# ── 3. PALETA DE COLORES POR TIPO DE DELITO ──────────────────────
colores_delito = {
    'homicidio doloso':              '#FF0000',   # rojo fuerte
    'feminicidio':                   '#FF00FF',   # magenta
    'violacion':                     '#FF6600',   # naranja
    'abuso sexual infantil':         '#FF9900',   # ámbar
    'violencia familiar':            '#FFD700',   # amarillo
    'lesiones dolosas':              '#FF4444',   # rojo claro
    'robo a persona':                '#00BFFF',   # azul cielo
    'robo a negocio':                '#1E90FF',   # azul
    'robo a casa habitacion':        '#4169E1',   # azul oscuro
    'robo a vehiculos particulares': '#00CED1',   # turquesa
    'robo de motocicleta':           '#20B2AA',   # verde agua
    'robo de autopartes':            '#3CB371',   # verde
    'robo a int de vehiculos':       '#90EE90',   # verde claro
    'robo a bancos':                 '#8B0000',   # rojo vino
    'robo a carga pesada':           '#A0522D',   # café
    'robo a cuentahabientes':        '#DDA0DD',   # lavanda
}

# ── 4. DIBUJAR MAPA BASE ─────────────────────────────────────────
print("Generando mapa...")
fig, ax = plt.subplots(figsize=(22, 22), facecolor='#1a1a2e')
ax.set_facecolor('#1a1a2e')

ox.plot_graph(G, ax=ax, show=False, close=False,
              bgcolor='none',
              edge_color='#444466',
              edge_linewidth=0.3,
              node_size=0)

# ── 5. PLOTEAR CADA TIPO DE DELITO ───────────────────────────────
delitos_ordenados = df['delito'].value_counts().index  # del más al menos frecuente

for delito in delitos_ordenados:
    subset = df[df['delito'] == delito]
    color  = colores_delito.get(delito, '#FFFFFF')
    ax.scatter(subset['x'], subset['y'],
               c=color, s=4, alpha=0.5,
               linewidths=0, zorder=3)

# ── 6. LEYENDA CON CONTEOS ───────────────────────────────────────
handles = []
for delito in delitos_ordenados:
    count = len(df[df['delito'] == delito])
    color = colores_delito.get(delito, '#FFFFFF')
    handles.append(
        mpatches.Patch(color=color, label=f"{delito.title()}  ({count:,})")
    )

legend = ax.legend(
    handles=handles,
    loc='lower left',
    fontsize=10,
    framealpha=0.85,
    facecolor='#0d0d1a',
    edgecolor='#555577',
    labelcolor='white',
    title='Tipo de delito',
    title_fontsize=12
)
legend.get_title().set_color('white')

# ── 7. TÍTULO Y ANOTACIONES ──────────────────────────────────────
ax.set_title("Incidentes Delictivos — Guadalajara 2023",
             fontsize=18, fontweight='bold', color='white', pad=14)

ax.annotate(f"Total de incidentes: {len(df):,}  |  Fuente: IIEG Jalisco 2023",
            xy=(0.5, 0.01), xycoords='axes fraction',
            ha='center', fontsize=9, color='#AAAACC')

ax.set_axis_off()

fig.savefig("mapa_delitos_gdl.png", dpi=150,
            bbox_inches='tight', facecolor='#1a1a2e')
plt.close(fig)
print("¡Listo! Revisa 'mapa_delitos_gdl.png'")