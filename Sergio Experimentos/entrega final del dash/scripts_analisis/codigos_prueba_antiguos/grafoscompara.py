import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter

# Optimización de caché para OSMnx
ox.settings.use_cache = True

# ── 1. DATOS ORIGINALES (EL SECRETO ESTABA AQUÍ) ─────────
print("Cargando datos y agrupando por colonia exacta...")
df = pd.read_csv('iieg_2023.csv')
df = df.dropna(subset=['x', 'y'])

# Agrupamos usando la columna oficial del CSV (exactamente como en tu otro código)
colonias_stats = df.groupby('colonia').agg(
    delitos=('delito', 'count'),
    x=('x', 'mean'), # Centroide matemático basado en la concentración de delitos
    y=('y', 'mean')
).reset_index().sort_values('delitos', ascending=False)

# Tomamos el Top 8 real
top = colonias_stats.head(8)
print("\n🔥 Top 8 colonias a rutear:")
print(top[['colonia', 'delitos']])
print("-" * 30)

# ── 2. DESCARGAR RED VIAL DE GUADALAJARA ────────────────
print("Descargando polígono y red vial de Guadalajara...")
G = ox.graph_from_place("Guadalajara, Jalisco, Mexico", network_type='drive')

# ── 3. SEMÁFOROS Y PESOS ────────────────────────────────
print("Calculando semáforos e impedancias...")
semaforos = ox.features_from_place(
    "Guadalajara, Jalisco, Mexico",
    tags={'highway': 'traffic_signals'}
)
semaforos = semaforos[semaforos.geometry.type == 'Point']

nodos_semaforo = set(
    ox.distance.nearest_nodes(G,
        semaforos.geometry.x,
        semaforos.geometry.y)
)

for u, v, k, data in G.edges(data=True, keys=True):
    data['impedancia'] = data.get('length', 1) + (
        400 if v in nodos_semaforo else 0
    )

# ── 4. CENTROIDES (COLONIAS) ───────────────────────────
print("Preparando nodos para la ruta...")
nodos = []
for _, row in top.iterrows():
    # Buscamos el nodo de calle más cercano al centroide de los delitos
    nodo = ox.distance.nearest_nodes(G, row['x'], row['y'])
    
    nodos.append({
        'nombre': str(row['colonia']).title(), # Capitalizamos para que se vea elegante
        'delitos': row['delitos'],
        'x': row['x'],
        'y': row['y'],
        'nodo': nodo
    })

# ── 5. RUTA TIPO TSP ───────────────────────────────────
print("Calculando ruta óptima (TSP)...")

def distancia(ruta):
    d = 0
    for u, v in zip(ruta[:-1], ruta[1:]):
        data = G.get_edge_data(u, v)
        if data:
            d += list(data.values())[0].get('length', 0)
    return d

K = len(nodos)
dist_matrix = np.zeros((K, K))

for i in range(K):
    for j in range(K):
        if i != j:
            try:
                r = nx.shortest_path(G,
                                     nodos[i]['nodo'],
                                     nodos[j]['nodo'],
                                     weight='impedancia')
                dist_matrix[i][j] = distancia(r)
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

rutas = []
for i in range(len(visitados)-1):
    try:
        r = nx.shortest_path(
            G,
            nodos[visitados[i]]['nodo'],
            nodos[visitados[i+1]]['nodo'],
            weight='impedancia'
        )
        rutas.append(r)
    except:
        pass

# ── 6. MAPA HD ──────────────────────────────────────────
print("Generando mapa HD...")
fig, ax = plt.subplots(figsize=(20, 20), facecolor='#0d0d1a')
ax.set_facecolor('#0d0d1a')

# Red vial
ox.plot_graph(G, ax=ax, show=False, close=False,
              bgcolor='none',
              edge_color='#1e1e3a',
              edge_linewidth=0.2,
              node_size=0)

nodes_gdf, edges = ox.graph_to_gdfs(G)
xmin, ymin, xmax, ymax = edges.total_bounds

# Heatmap Vectorizado de Alta Velocidad
xs = df['x'].values
ys = df['y'].values

heat, _, _ = np.histogram2d(xs, ys, bins=400, range=[[xmin, xmax], [ymin, ymax]])
heat = heat.T
heat = gaussian_filter(heat, sigma=6)

ax.imshow(heat, extent=[xmin, xmax, ymin, ymax],
          origin='lower', cmap='YlOrRd', alpha=0.4)

# Semáforos
ax.scatter(semaforos.geometry.x,
           semaforos.geometry.y,
           c='#FFD700', s=3, alpha=0.5)

# Rutas
for ruta in rutas:
    xs_ruta = [G.nodes[n]['x'] for n in ruta]
    ys_ruta = [G.nodes[n]['y'] for n in ruta]
    ax.plot(xs_ruta, ys_ruta, color='#00FFAA', linewidth=2.5)

# Colonias TOP con Anotaciones Exactas
for n in nodos:
    ax.scatter(n['x'], n['y'], c='#ff2a2a', s=200, zorder=6, edgecolors='white', linewidths=1.5)

    ax.annotate(
        f"{n['nombre']}\n({int(n['delitos'])} delitos)",
        xy=(n['x'], n['y']),
        xytext=(15, 15),
        textcoords="offset points",
        fontsize=11,
        color='white',
        fontweight='bold',
        path_effects=[pe.withStroke(linewidth=3, foreground="#000000")],
        arrowprops=dict(arrowstyle='->', color='#FFD700', lw=1.5),
        zorder=7
    )

ax.set_title("Rutas Críticas y Colonias de Mayor Índice Delictivo — Guadalajara",
             color='white', fontsize=22, fontweight='bold', pad=20)

ax.set_axis_off()

fig.savefig("colonias_peligrosas_gdl_final.png",
            dpi=150,
            bbox_inches='tight',
            facecolor='#0d0d1a')

plt.close(fig)

print("🔥 ¡Listo! Revisa la imagen 'colonias_peligrosas_gdl_final.png'. Números 100% exactos.")