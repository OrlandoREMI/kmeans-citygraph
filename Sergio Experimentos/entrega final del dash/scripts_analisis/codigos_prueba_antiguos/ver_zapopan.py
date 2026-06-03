import matplotlib
matplotlib.use('Agg')
import osmnx as ox
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

ox.settings.use_cache = True
lugar = "Zapopan, Jalisco, Mexico"

print("Descargando red vial...")
G = ox.graph_from_place(lugar, network_type='drive')

print("Descargando zonas verdes...")
parques = ox.features_from_place(lugar, tags={'leisure': 'park'})
verde   = ox.features_from_place(lugar, tags={'landuse': ['grass', 'forest', 'meadow']})

print("Descargando agua...")
agua = ox.features_from_place(lugar, tags={'natural': 'water'})
rios = ox.features_from_place(lugar, tags={'waterway': 'river'})

print("Descargando semáforos...")
semaforos = ox.features_from_place(lugar, tags={'highway': 'traffic_signals'})

print("Dibujando mapa...")
fig, ax = plt.subplots(figsize=(22, 22), facecolor='#E8E0D0')
ax.set_facecolor('#E8E0D0')

# Zonas verdes
if not verde.empty:
    verde.plot(ax=ax, color='#5A8C3C', alpha=0.6, linewidth=0)
if not parques.empty:
    parques.plot(ax=ax, color='#7AB84A', edgecolor='#5A8C3C', linewidth=0.4, alpha=0.75)

# Agua
if not agua.empty:
    agua.plot(ax=ax, color='#4A90C4', edgecolor='#2E6A9A', linewidth=0.4, alpha=0.85)
if not rios.empty:
    rios.plot(ax=ax, color='#4A90C4', linewidth=1.2, alpha=0.85)

# Red vial
ox.plot_graph(G, ax=ax, show=False, close=False,
              bgcolor='none',
              edge_color='#888888',
              edge_linewidth=0.35,
              node_size=0)

# Semáforos
if not semaforos.empty:
    sem_pts = semaforos[semaforos.geometry.geom_type == 'Point']
    if not sem_pts.empty:
        sem_pts.plot(ax=ax, color='#FF3300', markersize=4,
                     marker='o', zorder=6, label='_nolegend_')
        # Punto representativo para leyenda
        ax.plot([], [], 'o', color='#FF3300', markersize=8, label=f'Semáforos ({len(sem_pts)})')

# Parches para leyenda de zonas
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

leyenda = [
    Patch(facecolor='#7AB84A', edgecolor='#5A8C3C', label='Parques / Zonas verdes'),
    Patch(facecolor='#5A8C3C', edgecolor='#5A8C3C', label='Bosques / Pastizales'),
    Patch(facecolor='#4A90C4', edgecolor='#2E6A9A', label='Cuerpos de agua / Ríos'),
    Line2D([0],[0], color='#888888', linewidth=1.5,  label='Red vial'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#FF3300',
           markersize=9, label=f'Semáforos ({len(semaforos) if not semaforos.empty else 0})'),
]

ax.legend(handles=leyenda, loc='lower right', fontsize=13,
          framealpha=0.9, facecolor='white', edgecolor='#AAAAAA',
          title='Referencias', title_fontsize=14)

ax.set_title("Zapopan, Jalisco — Zonas verdes, agua y semáforos",
             fontsize=17, fontweight='bold', pad=14)
ax.set_axis_off()

fig.savefig("zapopan_verde_agua_semaforos.png", dpi=150,
            bbox_inches='tight', facecolor='#E8E0D0')
plt.close(fig)
print("¡Listo! Revisa 'zapopan_verde_agua_semaforos.png'")