import matplotlib
matplotlib.use('Agg')
import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

# Configuración
ox.settings.use_cache = True
PLACE_NAME = "Guadalajara, Jalisco, Mexico"

print("1/3 Descargando red vial...")
G = ox.graph_from_place(PLACE_NAME, network_type='drive')

print("2/3 Procesando velocidades...")
G = ox.add_edge_speeds(G)

print("3/3 Generando mapa visible...")
fig, ax = plt.subplots(figsize=(25, 20), facecolor='#0d0d1a')
ax.set_facecolor('#0d0d1a')

lines = []
colors = []
widths = []

cmap = plt.colormaps['RdYlGn'] 
norm = plt.Normalize(vmin=20, vmax=90)

for u, v, data in G.edges(data=True):
    # Obtener geometría
    if 'geometry' in data:
        xs, ys = data['geometry'].xy
        lines.append(list(zip(xs, ys)))
    else:
        x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
        x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
        lines.append([(x1, y1), (x2, y2)])
    
    # Velocidad para el color
    speed = data.get('speed_kph', 30)
    if isinstance(speed, list): speed = speed[0]
    speed = float(speed)
    colors.append(cmap(norm(speed)))
    
    # AJUSTE CRÍTICO: Grosor según velocidad
    # Las vías rápidas (verdes) serán más gruesas para resaltar
    if speed > 70:
        widths.append(2.0)
    elif speed > 40:
        widths.append(1.0)
    else:
        widths.append(0.4)

# Crear la colección con grosores y colores mejorados
lc = LineCollection(lines, colors=colors, linewidths=widths, alpha=0.9, zorder=1)
ax.add_collection(lc)

# Auto-ajustar los límites del mapa para que no se vea "lejos"
ax.autoscale()

ax.set_title("Mapa de Calor de Velocidades: Guadalajara (Corregido)", 
             color='white', fontsize=25, pad=30, fontweight='bold')
ax.axis('off')

# Barra de color más grande y legible
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.03)
cbar.set_label('Velocidad Estimada (km/h)', color='white', fontsize=15)
cbar.ax.yaxis.set_tick_params(color='white', labelsize=12)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

plt.savefig("mapa_calor_visible.png", dpi=200, bbox_inches='tight', facecolor='#0d0d1a')
print("¡Listo! Ahora deberías ver la red vial claramente en 'mapa_calor_visible.png'")