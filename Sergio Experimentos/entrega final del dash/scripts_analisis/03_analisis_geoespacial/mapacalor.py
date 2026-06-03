import matplotlib
matplotlib.use('Agg')
import pandas as pd
import osmnx as ox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from scipy.ndimage import gaussian_filter

ox.settings.use_cache = True

# ── 1. CARGAR DATOS ──────────────────────────────────────────────
print("Cargando datos...")
df = pd.read_csv('iieg_2023.csv')
df = df.dropna(subset=['x', 'y'])

# Conteo por colonia y merge con coordenadas (centroide de cada colonia)
colonias = df.groupby('colonia').agg(
    total=('delito', 'count'),
    x=('x', 'mean'),       # centroide aproximado
    y=('y', 'mean')
).reset_index().sort_values('total', ascending=False)

print(f"Total colonias: {len(colonias)}")
print(colonias.head(10))

# ── 2. CARGAR MAPA ───────────────────────────────────────────────
print("Cargando mapa...")
G = ox.graph_from_place("Guadalajara, Jalisco, Mexico", network_type='drive')

# ── 3. HEATMAP CON GAUSSIAN FILTER ──────────────────────────────
print("Generando heatmap...")
fig, axes = plt.subplots(1, 2, figsize=(28, 16), facecolor='#0d0d1a')
fig.suptitle("Mapa de calor de incidentes delictivos — Guadalajara 2023",
             fontsize=20, fontweight='bold', color='white', y=0.98)

# ─── Panel izquierdo: heatmap sobre mapa ─────────────────────────
ax1 = axes[0]
ax1.set_facecolor('#0d0d1a')

ox.plot_graph(G, ax=ax1, show=False, close=False,
              bgcolor='none', edge_color='#2a2a4a',
              edge_linewidth=0.25, node_size=0)

# Crear grilla para el gaussian heatmap
resolucion = 500
x_min, x_max = df['x'].min() - 0.002, df['x'].max() + 0.002
y_min, y_max = df['y'].min() - 0.002, df['y'].max() + 0.002

xi = np.linspace(x_min, x_max, resolucion)
yi = np.linspace(y_min, y_max, resolucion)
xi_grid, yi_grid = np.meshgrid(xi, yi)

# Acumular puntos en la grilla
heatmap = np.zeros((resolucion, resolucion))
for _, row in df.iterrows():
    ix = int((row['x'] - x_min) / (x_max - x_min) * (resolucion - 1))
    iy = int((row['y'] - y_min) / (y_max - y_min) * (resolucion - 1))
    if 0 <= ix < resolucion and 0 <= iy < resolucion:
        heatmap[iy, ix] += 1

# Suavizar con filtro gaussiano
heatmap_suave = gaussian_filter(heatmap, sigma=8)

# Dibujar heatmap
cmap = plt.cm.get_cmap('YlOrRd')
hm = ax1.imshow(
    heatmap_suave,
    extent=[x_min, x_max, y_min, y_max],
    origin='lower',
    cmap=cmap,
    alpha=0.75,
    aspect='auto',
    zorder=3
)

# Colorbar
cbar = plt.colorbar(hm, ax=ax1, fraction=0.03, pad=0.02)
cbar.set_label('Densidad de incidentes', color='white', fontsize=11)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

# Etiquetas top 10 colonias
top10 = colonias.head(10)
for _, row in top10.iterrows():
    ax1.annotate(
        row['colonia'].title(),
        xy=(row['x'], row['y']),
        fontsize=7,
        color='white',
        fontweight='bold',
        ha='center',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5, edgecolor='none'),
        zorder=6
    )

ax1.set_title("Densidad de incidentes por zona", color='white', fontsize=13, pad=10)
ax1.set_axis_off()

# ─── Panel derecho: ranking de colonias ──────────────────────────
ax2 = axes[1]
ax2.set_facecolor('#0d0d1a')

top25 = colonias.head(25)

# Colorear barras según nivel de riesgo
norm = mcolors.Normalize(vmin=top25['total'].min(), vmax=top25['total'].max())
colores_barras = [cmap(norm(v)) for v in top25['total']]

bars = ax2.barh(
    range(len(top25)),
    top25['total'],
    color=colores_barras,
    edgecolor='none',
    height=0.7
)

# Etiquetas de valor al final de cada barra
for i, (bar, val) in enumerate(zip(bars, top25['total'])):
    ax2.text(val + 5, i, str(val),
             va='center', color='white', fontsize=9, fontweight='bold')

ax2.set_yticks(range(len(top25)))
ax2.set_yticklabels(
    [c.title() for c in top25['colonia']],
    color='white', fontsize=9
)
ax2.set_xlabel('Número de incidentes', color='white', fontsize=11)
ax2.set_title("Top 25 colonias por incidentes", color='white', fontsize=13, pad=10)
ax2.tick_params(colors='white')
ax2.spines['bottom'].set_color('#444466')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.set_facecolor('#0d0d1a')
ax2.xaxis.label.set_color('white')

# Línea de referencia promedio
promedio = colonias['total'].mean()
ax2.axvline(x=promedio, color='#AAAAFF', linewidth=1,
            linestyle='--', alpha=0.6)
ax2.text(promedio + 3, len(top25) - 0.5,
         f'Promedio: {promedio:.0f}',
         color='#AAAAFF', fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("heatmap_colonias.png", dpi=150,
            bbox_inches='tight', facecolor='#0d0d1a')
plt.close(fig)
print("¡Listo! Revisa 'heatmap_colonias.png'")