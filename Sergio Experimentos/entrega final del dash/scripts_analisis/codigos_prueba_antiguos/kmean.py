import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ox.settings.use_cache = True

# ── 1. CARGAR Y PREPARAR DATOS ───────────────────────────────────
print("Cargando datos...")
df = pd.read_csv('iieg_2023.csv')
df = df.dropna(subset=['x', 'y'])

# Features para clustering: coordenadas solamente
# KMeans agrupa por proximidad geográfica
coords = df[['x', 'y']].values

# Escalar coordenadas (KMeans es sensible a escala)
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)

# ── 2. MÉTODO DEL CODO — encontrar K óptimo ──────────────────────
# Prueba K de 2 a 12 y mide la inercia (suma de distancias al centroide)
print("Calculando método del codo...")
inercias = []
rango_k = range(2, 13)

for k in rango_k:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(coords_scaled)
    inercias.append(km.inertia_)

# ── 3. ELEGIR K=6 Y ENTRENAR ─────────────────────────────────────
# En el codo verás que ~6 zonas es donde la curva se aplana
# para Guadalajara: Norte, Sur, Centro, Oriente, Poniente, Centro-Sur
K = 6
print(f"Entrenando KMeans con K={K}...")
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(coords_scaled)

# Estadísticas por cluster
resumen = df.groupby('cluster').agg(
    total=('delito', 'count'),
    delito_top=('delito', lambda x: x.value_counts().index[0]),
    colonias=('colonia', 'nunique'),
    x_centro=('x', 'mean'),
    y_centro=('y', 'mean')
).reset_index().sort_values('total', ascending=False)

print("\n=== RESUMEN POR CLUSTER ===")
print(resumen.to_string())

# ── 4. COLORES Y NOMBRES POR CLUSTER ────────────────────────────
colores_cluster = {
    0: '#FF4444',   # rojo
    1: '#FFD700',   # amarillo
    2: '#00BFFF',   # azul cielo
    3: '#FF8C00',   # naranja
    4: '#7CFC00',   # verde
    5: '#DA70D6',   # orquídea
}

# Nombrar clusters según su posición geográfica
def nombrar_cluster(row):
    x, y = row['x_centro'], row['y_centro']
    cx = df['x'].mean()
    cy = df['y'].mean()
    ns = 'Norte' if y > cy else 'Sur'
    eo = 'Poniente' if x < cx else 'Oriente'
    if abs(x - cx) < 0.02 and abs(y - cy) < 0.02:
        return 'Centro'
    return f'{ns}-{eo}'

resumen['nombre'] = resumen.apply(nombrar_cluster, axis=1)
# Mapear nombre al df principal
nombre_map = resumen.set_index('cluster')['nombre'].to_dict()
df['zona'] = df['cluster'].map(nombre_map)

# ── 5. FIGURA PRINCIPAL: 3 paneles ──────────────────────────────
print("Cargando mapa de Guadalajara...")
G = ox.graph_from_place("Guadalajara, Jalisco, Mexico", network_type='drive')

fig = plt.figure(figsize=(28, 18), facecolor='#0d0d1a')
fig.suptitle("Clustering de zonas de crimen — Guadalajara 2023",
             fontsize=22, fontweight='bold', color='white', y=0.98)

# ─── Panel 1 (izq): Mapa con clusters ───────────────────────────
ax1 = fig.add_subplot(1, 3, (1, 2))
ax1.set_facecolor('#0d0d1a')

ox.plot_graph(G, ax=ax1, show=False, close=False,
              bgcolor='none', edge_color='#2a2a4a',
              edge_linewidth=0.25, node_size=0)

# Dibujar puntos por cluster
for cluster_id, grupo in df.groupby('cluster'):
    color = colores_cluster[cluster_id]
    zona  = nombre_map[cluster_id]
    ax1.scatter(grupo['x'], grupo['y'],
                c=color, s=4, alpha=0.45,
                linewidths=0, zorder=4,
                label=f'Zona {zona} ({len(grupo):,})')

# Centroides con X grande
centroides = scaler.inverse_transform(kmeans.cluster_centers_)
for i, (cx, cy) in enumerate(centroides):
    color = colores_cluster[i]
    zona  = nombre_map[i]
    count = resumen[resumen['cluster'] == i]['total'].values[0]
    ax1.scatter(cx, cy, c=color, s=200, marker='*',
                edgecolors='white', linewidths=0.8, zorder=7)
    ax1.annotate(
        f"{zona}\n{count:,} casos",
        xy=(cx, cy), xytext=(cx + 0.008, cy + 0.004),
        fontsize=8, color='white', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d0d1a',
                  alpha=0.8, edgecolor=color, linewidth=1),
        zorder=8
    )

# Leyenda
handles = [mpatches.Patch(color=colores_cluster[r['cluster']],
           label=f"Zona {r['nombre']}  ({r['total']:,} casos)  —  top: {r['delito_top'].title()}")
           for _, r in resumen.iterrows()]
legend = ax1.legend(handles=handles, loc='lower left', fontsize=9,
                    facecolor='#0d0d1a', edgecolor='#444466',
                    labelcolor='white', title='Clusters (★ = centroide)',
                    title_fontsize=10)
legend.get_title().set_color('white')
ax1.set_title("Mapa de zonas de riesgo", color='white', fontsize=14, pad=10)
ax1.set_axis_off()

# ─── Panel 2 (der arriba): Método del codo ──────────────────────
ax2 = fig.add_subplot(2, 3, 3)
ax2.set_facecolor('#0d0d1a')

ax2.plot(list(rango_k), inercias, color='#00BFFF',
         linewidth=2, marker='o', markersize=6,
         markerfacecolor='white', markeredgecolor='#00BFFF')
ax2.axvline(x=K, color='#FFD700', linewidth=1.5,
            linestyle='--', alpha=0.8)
ax2.text(K + 0.2, max(inercias) * 0.95,
         f'K={K} elegido', color='#FFD700', fontsize=9)

ax2.set_title("Método del codo", color='white', fontsize=12)
ax2.set_xlabel("Número de clusters (K)", color='white', fontsize=10)
ax2.set_ylabel("Inercia", color='white', fontsize=10)
ax2.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_edgecolor('#444466')
ax2.set_facecolor('#0d0d1a')

# ─── Panel 3 (der abajo): Barras por cluster ────────────────────
ax3 = fig.add_subplot(2, 3, 6)
ax3.set_facecolor('#0d0d1a')

nombres  = [nombre_map[r['cluster']] for _, r in resumen.iterrows()]
totales  = resumen['total'].values
colores  = [colores_cluster[r['cluster']] for _, r in resumen.iterrows()]

bars = ax3.bar(nombres, totales, color=colores,
               edgecolor='none', width=0.6)
for bar, val in zip(bars, totales):
    ax3.text(bar.get_x() + bar.get_width() / 2,
             val + 20, str(val),
             ha='center', color='white', fontsize=9, fontweight='bold')

ax3.set_title("Incidentes por zona", color='white', fontsize=12)
ax3.set_ylabel("Total incidentes", color='white', fontsize=10)
ax3.tick_params(colors='white', axis='both')
ax3.tick_params(axis='x', labelsize=8, rotation=15)
for spine in ax3.spines.values():
    spine.set_edgecolor('#444466')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.set_facecolor('#0d0d1a')

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("clustering_zonas.png", dpi=150,
            bbox_inches='tight', facecolor='#0d0d1a')
plt.close(fig)
print("¡Listo! Revisa 'clustering_zonas.png'")