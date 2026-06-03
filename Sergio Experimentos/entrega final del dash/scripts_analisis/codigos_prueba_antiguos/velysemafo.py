import matplotlib
matplotlib.use('Agg')
import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

# Configuración
ox.settings.use_cache = True
PLACE_NAME = "Guadalajara, Jalisco, Mexico"

print("1/4 Descargando red vial y semáforos...")
G = ox.graph_from_place(PLACE_NAME, network_type='drive')
# Obtener semáforos reales de OSM
semaforos = ox.features_from_place(PLACE_NAME, tags={'highway': 'traffic_signals'})
semaforos = semaforos[semaforos.geometry.type == 'Point']

print("2/4 Calculando velocidades promedio con penalización...")
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)

# Localizar nodos con semáforo
nodos_semaforo = set(ox.distance.nearest_nodes(G, X=semaforos.geometry.x, Y=semaforos.geometry.y))

# PENALIZACIÓN: Segundos promedio que un semáforo quita al flujo
COSTO_SEMAFORO_SEG = 40 

for u, v, k, data in G.edges(data=True, keys=True):
    distancia = data.get('length', 1)
    tiempo_legal = data.get('travel_time', 1)
    
    # Si hay semáforo al final del tramo, sumamos la penalización al tiempo
    if v in nodos_semaforo:
        tiempo_promedio = tiempo_legal + COSTO_SEMAFORO_SEG
    else:
        tiempo_promedio = tiempo_legal
    
    # Recalculamos la velocidad promedio real (v = d / t) convertida a km/h
    # Multiplicamos por 3.6 para pasar de m/s a km/h
    vel_promedio = (distancia / tiempo_promedio) * 3.6
    data['vel_promedio'] = vel_promedio

print("3/4 Generando mapa de calor...")
fig, ax = plt.subplots(figsize=(25, 20), facecolor='#0d0d1a')
ax.set_facecolor('#0d0d1a')

lines, colors, widths = [], [], []
cmap = plt.colormaps['RdYlGn'] 
norm = plt.Normalize(vmin=10, vmax=70) # Escala de velocidad promedio real

for u, v, data in G.edges(data=True):
    if 'geometry' in data:
        xs, ys = data['geometry'].xy
        lines.append(list(zip(xs, ys)))
    else:
        lines.append([(G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])])
    
    v_prom = data.get('vel_promedio', 20)
    colors.append(cmap(norm(v_prom)))
    
    # Grosor: las vías con mejor velocidad promedio se ven más gruesas
    widths.append(1.8 if v_prom > 40 else 0.4)

# Dibujar vialidades
lc = LineCollection(lines, colors=colors, linewidths=widths, alpha=0.9, zorder=1)
ax.add_collection(lc)

# Dibujar semáforos como puntos de advertencia (pequeños destellos)
ax.scatter(semaforos.geometry.x, semaforos.geometry.y, 
           c='#FFD700', s=8, alpha=0.4, edgecolors='none', zorder=2)

ax.autoscale()
ax.set_title("Velocidades Promedio Reales: Guadalajara\n(Incluye penalización por Semáforos | Rojo=Lento, Verde=Rápido)", 
             color='white', fontsize=24, pad=30, fontweight='bold')
ax.axis('off')

# Barra de color
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.03)
cbar.set_label('Velocidad Promedio Calculada (km/h)', color='white', fontsize=16)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

# 4/4 Guardar
output = "mapa_velocidad_promedio_semaforos.png"
plt.savefig(output, dpi=200, bbox_inches='tight', facecolor='#0d0d1a')
print(f"4/4 ¡Listo! Archivo guardado como '{output}'")