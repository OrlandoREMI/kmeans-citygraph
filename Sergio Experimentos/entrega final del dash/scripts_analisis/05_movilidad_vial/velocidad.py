import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy.ndimage import gaussian_filter

ox.settings.use_cache = True

# ── 1. DATOS ─────────────────────────────────────────────────────
print("Cargando datos...")
df = pd.read_csv('iieg_2023.csv')
df = df.dropna(subset=['x', 'y'])

colonias_stats = df.groupby('colonia').agg(
    delitos=('delito', 'count'),
    x=('x', 'mean'),
    y=('y', 'mean')
).reset_index().sort_values('delitos', ascending=False)

top = colonias_stats.head(8)

# ── 2. RED VIAL + VELOCIDADES ─────────────────────────────────────
print("Descargando red vial...")
G = ox.graph_from_place("Guadalajara, Jalisco, Mexico", network_type='drive')

# Añadir velocidades: usa maxspeed de OSM donde existe,
# infiere por tipo de vía donde no hay dato
G = ox.add_edge_speeds(G)          # km/h en atributo 'speed_kph'
G = ox.add_edge_travel_times(G)    # segundos en atributo 'travel_time'

# Verificar cobertura
edges = ox.graph_to_gdfs(G, nodes=False)
con_speed = edges['speed_kph'].notna().sum()
print(f"Aristas con velocidad asignada: {con_speed}/{len(edges)} ({100*con_speed/len(edges):.1f}%)")

# ── 3. SEMÁFOROS ──────────────────────────────────────────────────
print("Descargando semáforos...")
semaforos = ox.features_from_place(
    "Guadalajara, Jalisco, Mexico",
    tags={'highway': 'traffic_signals'}
)
semaforos = semaforos[semaforos.geometry.type == 'Point']
nodos_semaforo = set(ox.distance.nearest_nodes(
    G, semaforos.geometry.x, semaforos.geometry.y))
print(f"Semáforos: {len(nodos_semaforo)} nodos")

# ── 4. PESOS POR ESCENARIO HORARIO ───────────────────────────────
# Penalización en SEGUNDOS según hora del día
escenarios = {
    'madrugada (0-6h)':  30,    # semáforo ~30 seg
    'día normal (9-17h)': 60,   # semáforo ~60 seg
    'hora pico (7-9h)':  120,   # semáforo ~120 seg
}

def aplicar_pesos(G, penalizacion_seg):
    for u, v, k, data in G.edges(data=True, keys=True):
        tiempo_base = data.get('travel_time', data.get('length', 1) / (30/3.6))
        data['tiempo_total'] = tiempo_base + (penalizacion_seg if v in nodos_semaforo else 0)
    return G

# ── 5. NODOS DE LAS COLONIAS ──────────────────────────────────────
nodos = []
for _, row in top.iterrows():
    nodo = ox.distance.nearest_nodes(G, row['x'], row['y'])
    nodos.append({
        'nombre':  str(row['colonia']).title(),
        'delitos': row['delitos'],
        'x': row['x'], 'y': row['y'],
        'nodo': nodo
    })

# ── 6. TSP GREEDY ─────────────────────────────────────────────────
def calcular_circuito(G, nodos, weight='tiempo_total'):
    K = len(nodos)
    dist_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i != j:
                try:
                    path = nx.shortest_path(G, nodos[i]['nodo'], nodos[j]['nodo'], weight=weight)
                    dist_matrix[i][j] = sum(
                        G[path[k]][path[k+1]][0].get(weight, 0)
                        for k in range(len(path)-1)
                    )
                except:
                    dist_matrix[i][j] = np.inf

    visitados = [0]
    resto = list(range(1, K))
    while resto:
        ultimo = visitados[-1]
        sig = min(resto, key=lambda x: dist_matrix[ultimo][x])
        visitados.append(sig)
        resto.remove(sig)
    visitados.append(visitados[0])

    rutas, tiempos, distancias, semaforos_cnt = [], [], [], []
    for i in range(len(visitados)-1):
        try:
            r = nx.shortest_path(G, nodos[visitados[i]]['nodo'],
                                    nodos[visitados[i+1]]['nodo'], weight=weight)
            t = sum(G[r[k]][r[k+1]][0].get('travel_time', 0) for k in range(len(r)-1))
            d = sum(G[r[k]][r[k+1]][0].get('length', 0)      for k in range(len(r)-1))
            s = sum(1 for n in r if n in nodos_semaforo)
            rutas.append(r)
            tiempos.append(t)
            distancias.append(d)
            semaforos_cnt.append(s)
        except:
            pass
    return rutas, tiempos, distancias, semaforos_cnt, visitados

# Calcular para los 3 escenarios
resultados = {}
for nombre_esc, pen in escenarios.items():
    G = aplicar_pesos(G, pen)
    rutas, tiempos, distancias, sems, orden = calcular_circuito(G, nodos)
    resultados[nombre_esc] = {
        'rutas': rutas, 'tiempos': tiempos,
        'distancias': distancias, 'semaforos': sems,
        'orden': orden,
        'tiempo_total_min': sum(tiempos) / 60,
        'distancia_total_km': sum(distancias) / 1000,
    }
    print(f"{nombre_esc}: {sum(tiempos)/60:.1f} min | {sum(distancias)/1000:.2f} km")

# ── 7. FIGURA: MAPA + TABLA COMPARATIVA ──────────────────────────
print("Generando mapa...")
fig = plt.figure(figsize=(26, 20), facecolor='#0d0d1a')
fig.suptitle("Circuito de patrullaje — Tiempos estimados por escenario horario\nGuadalajara 2023",
             fontsize=18, fontweight='bold', color='white', y=0.99)

# ─── Panel izquierdo: mapa con ruta de día normal ────────────────
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_facecolor('#0d0d1a')

ox.plot_graph(G, ax=ax1, show=False, close=False,
              bgcolor='none', edge_color='#1e1e3a',
              edge_linewidth=0.2, node_size=0)

# Heatmap de fondo
nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
xmin, ymin, xmax, ymax = edges_gdf.total_bounds
xs, ys = df['x'].values, df['y'].values
heat, _, _ = np.histogram2d(xs, ys, bins=400,
                             range=[[xmin, xmax], [ymin, ymax]])
heat = gaussian_filter(heat.T, sigma=6)
ax1.imshow(heat, extent=[xmin, xmax, ymin, ymax],
           origin='lower', cmap='YlOrRd', alpha=0.4, zorder=2)

# Semáforos
ax1.scatter(semaforos.geometry.x, semaforos.geometry.y,
            c='#FFD700', s=2, alpha=0.4, zorder=3)

# Colormap para velocidades (CORREGIDO DEPRECATION WARNING)
speed_cmap = plt.colormaps['RdYlGn']
speed_norm = plt.Normalize(vmin=20, vmax=80)

for u, v, data in G.edges(data=True):
    spd = data.get('speed_kph', 30)
    if isinstance(spd, list):
        spd = float(spd[0])
    color = speed_cmap(speed_norm(spd))
    x_coords = [G.nodes[u]['x'], G.nodes[v]['x']]
    y_coords = [G.nodes[u]['y'], G.nodes[v]['y']]
    ax1.plot(x_coords, y_coords, color=color,
             linewidth=0.4, alpha=0.5, zorder=1)

# Ruta del escenario "día normal"
esc_dia = resultados['día normal (9-17h)']
for i, ruta in enumerate(esc_dia['rutas']):
    xs_r = [G.nodes[n]['x'] for n in ruta]
    ys_r = [G.nodes[n]['y'] for n in ruta]
    t_seg = esc_dia['tiempos'][i]
    ax1.plot(xs_r, ys_r, color='#00FFAA', linewidth=2.5,
             alpha=0.9, zorder=5, solid_capstyle='round')
    # Tiempo del segmento en el punto medio
    mx = np.mean([xs_r[0], xs_r[-1]])
    my = np.mean([ys_r[0], ys_r[-1]])
    ax1.annotate(f"{t_seg/60:.1f} min",
                 xy=(mx, my), fontsize=7, color='#00FFAA',
                 ha='center', fontweight='bold',
                 path_effects=[pe.withStroke(linewidth=2, foreground='#0d0d1a')],
                 zorder=7)

# Nodos de colonias
for i, n in enumerate(nodos):
    ax1.scatter(n['x'], n['y'], c='#FF2A2A', s=180,
                edgecolors='white', linewidths=1.2, zorder=8)
    ax1.annotate(
        f"{n['nombre']}\n{n['delitos']} delitos",
        xy=(n['x'], n['y']), xytext=(12, 12),
        textcoords='offset points', fontsize=8.5,
        color='white', fontweight='bold',
        path_effects=[pe.withStroke(linewidth=3, foreground='#000000')],
        arrowprops=dict(arrowstyle='->', color='#FFD700', lw=1.2),
        zorder=9
    )

# Colorbar de velocidades
sm = plt.cm.ScalarMappable(cmap=speed_cmap, norm=speed_norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax1, fraction=0.025, pad=0.02)
cbar.set_label('Velocidad límite (km/h)', color='white', fontsize=9)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

ax1.set_title("Límites de velocidad + circuito (día normal)",
              color='white', fontsize=12, pad=8)
ax1.set_axis_off()

# ─── Panel derecho: tabla comparativa por escenario ──────────────
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_facecolor('#0d0d1a')
ax2.set_axis_off()

colores_esc = {
    'madrugada (0-6h)':   '#00BFFF',
    'día normal (9-17h)': '#00FFAA',
    'hora pico (7-9h)':   '#FF6B35',
}

# Encabezado
ax2.set_title("Comparativa de tiempos por escenario horario",
              color='white', fontsize=13, pad=10)

# Resumen por escenario
y = 0.92
for esc, res in resultados.items():
    color = colores_esc[esc]
    ax2.text(0.03, y, esc.upper(), transform=ax2.transAxes,
             color=color, fontsize=10, fontweight='bold')
    ax2.text(0.55, y, f"{res['tiempo_total_min']:.1f} min totales",
             transform=ax2.transAxes, color='white', fontsize=10)
    ax2.text(0.80, y, f"{res['distancia_total_km']:.2f} km",
             transform=ax2.transAxes,
             color=color, fontsize=10, fontweight='bold')
    y -= 0.04

# CORREGIDO: axhline cambiado por plot para evitar conflictos con transform
ax2.plot([0.02, 0.98], [y + 0.02, y + 0.02],
         color='#444466', linewidth=0.8, transform=ax2.transAxes)
y -= 0.02

# Tabla segmento a segmento
headers = ['Segmento', 'Dist.', 'Mdrg.', 'Día', 'H.Pico', 'Semáf.']
col_x   = [0.03, 0.30, 0.46, 0.57, 0.68, 0.82]
for cx, h in zip(col_x, headers):
    ax2.text(cx, y, h, transform=ax2.transAxes,
             color='#AAAAFF', fontsize=8.5, fontweight='bold')
y -= 0.025

# CORREGIDO: axhline cambiado por plot
ax2.plot([0.02, 0.98], [y + 0.01, y + 0.01],
         color='#444466', linewidth=0.5, transform=ax2.transAxes)

# Usar orden del escenario día normal
orden = resultados['día normal (9-17h)']['orden']
for i in range(len(orden) - 1):
    seg_label = f"{nodos[orden[i]]['nombre'][:12]} → {nodos[orden[i+1]]['nombre'][:12]}"
    dist_m = resultados['día normal (9-17h)']['distancias'][i]
    t_mdrg = resultados['madrugada (0-6h)']['tiempos'][i] / 60
    t_dia  = resultados['día normal (9-17h)']['tiempos'][i] / 60
    t_pico = resultados['hora pico (7-9h)']['tiempos'][i] / 60
    sems   = resultados['día normal (9-17h)']['semaforos'][i]

    bg = '#1a1a2e' if i % 2 == 0 else '#0d0d1a'
    ax2.add_patch(plt.Rectangle((0.02, y - 0.025), 0.96, 0.038,
                                transform=ax2.transAxes,
                                facecolor=bg, alpha=0.5))
    valores = [seg_label, f"{dist_m/1000:.2f}km",
               f"{t_mdrg:.1f}m", f"{t_dia:.1f}m",
               f"{t_pico:.1f}m", str(sems)]
    colores_fila = ['white', 'white',
                    colores_esc['madrugada (0-6h)'],
                    colores_esc['día normal (9-17h)'],
                    colores_esc['hora pico (7-9h)'],
                    '#FFD700']
    for cx, val, col in zip(col_x, valores, colores_fila):
        ax2.text(cx, y, val, transform=ax2.transAxes,
                 color=col, fontsize=7.5, va='center')
    y -= 0.062

# Fila de totales
# CORREGIDO: axhline cambiado por plot
ax2.plot([0.02, 0.98], [y + 0.04, y + 0.04],
         color='#444466', linewidth=0.8, transform=ax2.transAxes)

ax2.text(0.03, y + 0.01,
         f"TOTAL DEL CIRCUITO:   "
         f"Mdrg {resultados['madrugada (0-6h)']['tiempo_total_min']:.0f} min   "
         f"Día {resultados['día normal (9-17h)']['tiempo_total_min']:.0f} min   "
         f"Pico {resultados['hora pico (7-9h)']['tiempo_total_min']:.0f} min",
         transform=ax2.transAxes, color='#FFD700',
         fontsize=9, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("circuito_tiempos.png", dpi=150,
            bbox_inches='tight', facecolor='#0d0d1a')
plt.close(fig)
print("¡Listo! Revisa 'circuito_tiempos.png'")