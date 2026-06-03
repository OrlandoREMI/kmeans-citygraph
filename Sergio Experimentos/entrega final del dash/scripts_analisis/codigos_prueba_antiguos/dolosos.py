import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

ox.settings.use_cache = True

# ── 1. CARGAR Y FILTRAR SOLO DELITOS CRÍTICOS ────────────────────
print("Cargando datos...")
df = pd.read_csv('iieg_2023.csv')
df = df.dropna(subset=['x', 'y'])

delitos_criticos = ['homicidio doloso', 'feminicidio', 'violacion', 'abuso sexual infantil']
df_critico = df[df['delito'].isin(delitos_criticos)].copy()

print(f"Total delitos críticos: {len(df_critico)}")
print(df_critico['delito'].value_counts())

# ── 2. CARGAR MAPA ───────────────────────────────────────────────
print("Cargando mapa...")
G = ox.graph_from_place("Guadalajara, Jalisco, Mexico", network_type='drive')

# ── 3. COLORMAP PERSONALIZADO: transparente → rojo intenso ───────
# Negro cuando no hay nada, rojo vivo en zonas de concentración
colores_critico = [
    (0.0, (0.0,  0.0,  0.0,  0.0)),   # transparente
    (0.2, (0.4,  0.0,  0.1,  0.6)),   # vino oscuro
    (0.5, (0.8,  0.0,  0.0,  0.8)),   # rojo
    (0.8, (1.0,  0.2,  0.0,  0.9)),   # rojo naranja
    (1.0, (1.0,  1.0,  0.2,  1.0)),   # amarillo (punto más caliente)
]
cmap_critico = LinearSegmentedColormap.from_list(
    'critico',
    [(v, c) for v, c in colores_critico]
)

# ── 4. CONSTRUIR GRILLA HEATMAP ──────────────────────────────────
resolucion = 600
x_min = df['x'].min() - 0.003
x_max = df['x'].max() + 0.003
y_min = df['y'].min() - 0.003
y_max = df['y'].max() + 0.003

heatmap = np.zeros((resolucion, resolucion))

for _, row in df_critico.iterrows():
    ix = int((row['x'] - x_min) / (x_max - x_min) * (resolucion - 1))
    iy = int((row['y'] - y_min) / (y_max - y_min) * (resolucion - 1))
    if 0 <= ix < resolucion and 0 <= iy < resolucion:
        heatmap[iy, ix] += 1

# Suavizar — sigma más bajo que el heatmap general
# para preservar la precisión geográfica de estos delitos
heatmap_suave = gaussian_filter(heatmap, sigma=6)

# ── 5. FIGURA PRINCIPAL ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(28, 16), facecolor='#0d0d1a')
fig.suptitle("Delitos de alta gravedad — Guadalajara 2023\nHomicidio doloso · Feminicidio · Violación · Abuso sexual infantil",
             fontsize=18, fontweight='bold', color='white', y=0.98)

# ─── Panel izquierdo: heatmap ────────────────────────────────────
ax1 = axes[0]
ax1.set_facecolor('#0d0d1a')

ox.plot_graph(G, ax=ax1, show=False, close=False,
              bgcolor='none', edge_color='#2a2a4a',
              edge_linewidth=0.3, node_size=0)

hm = ax1.imshow(
    heatmap_suave,
    extent=[x_min, x_max, y_min, y_max],
    origin='lower',
    cmap=cmap_critico,
    alpha=0.85,
    aspect='auto',
    zorder=3
)

# Colorbar
cbar = plt.colorbar(hm, ax=ax1, fraction=0.03, pad=0.02)
cbar.set_label('Concentración de delitos críticos', color='white', fontsize=11)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

# Top 8 colonias con etiqueta
top_cols = (df_critico.groupby('colonia')
            .agg(total=('delito','count'), x=('x','mean'), y=('y','mean'))
            .sort_values('total', ascending=False)
            .head(8))

for i, (colonia, row) in enumerate(top_cols.iterrows()):
    ax1.annotate(
        f"#{i+1} {colonia.title()} ({row['total']})",
        xy=(row['x'], row['y']),
        fontsize=7.5, color='white', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#1a0000',
                  alpha=0.9, edgecolor='#FF4444', linewidth=0.8),
        zorder=9
    )
    ax1.plot(row['x'], row['y'], 'x', color='#FF4444',
             markersize=6, markeredgewidth=1.2, zorder=8)

ax1.set_title("Mapa de calor — zonas críticas", color='white', fontsize=13, pad=10)
ax1.set_axis_off()

# ─── Panel derecho: desglose por delito y colonia ────────────────
ax2 = axes[1]
ax2.set_facecolor('#0d0d1a')

# Top 20 colonias, barras apiladas por tipo de delito
top20 = (df_critico.groupby('colonia')
         .size().sort_values(ascending=False).head(20).index)

df_top = df_critico[df_critico['colonia'].isin(top20)]
pivot = (df_top.groupby(['colonia', 'delito'])
         .size().unstack(fill_value=0)
         .loc[top20])

colores_delito = {
    'homicidio doloso':      '#FF0000',
    'abuso sexual infantil': '#FF8C00',
    'violacion':             '#FF4500',
    'feminicidio':           '#FF00FF',
}

bottom = np.zeros(len(pivot))
for delito in delitos_criticos:
    if delito in pivot.columns:
        valores = pivot[delito].values
        bars = ax2.barh(
            range(len(pivot)),
            valores,
            left=bottom,
            color=colores_delito[delito],
            edgecolor='none',
            height=0.65,
            label=delito.title()
        )
        bottom += valores

# Total al final de cada barra
totales = pivot.sum(axis=1).values
for i, total in enumerate(totales):
    ax2.text(total + 0.3, i, str(total),
             va='center', color='white', fontsize=9, fontweight='bold')

ax2.set_yticks(range(len(pivot)))
ax2.set_yticklabels([c.title() for c in pivot.index],
                    color='white', fontsize=8.5)
ax2.set_xlabel('Número de incidentes', color='white', fontsize=10)
ax2.set_title("Top 20 colonias — desglose por delito crítico",
              color='white', fontsize=12, pad=10)
ax2.tick_params(colors='white')

legend = ax2.legend(loc='lower right', fontsize=10,
                    facecolor='#0d0d1a', edgecolor='#444466',
                    labelcolor='white')
for spine in ax2.spines.values():
    spine.set_edgecolor('#444466')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_facecolor('#0d0d1a')

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("heatmap_critico.png", dpi=150,
            bbox_inches='tight', facecolor='#0d0d1a')
plt.close(fig)
print("¡Listo! Revisa 'heatmap_critico.png'")