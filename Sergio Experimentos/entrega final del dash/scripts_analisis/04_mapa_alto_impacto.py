"""
SCRIPT 4 — Mapa 2x2 Delitos de Alto Impacto
===========================================
Genera un análisis geoespacial dividido en 4 categorías de 
delitos de alto impacto, presentadas en una cuadrícula de 2x2.
"""
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import osmnx as ox
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

ox.settings.use_cache = True

print("Cargando datos...")
df_iieg = pd.read_csv('iieg_2023.csv').dropna(subset=['x', 'y'])

# 4 categorías propuestas de alto impacto
categorias = {
    'Homicidios y Feminicidios': ['homicidio doloso', 'feminicidio'],
    'Lesiones Dolosas': ['lesiones dolosas'],
    'Robo a Persona': ['robo a persona', 'robo a cuentahabientes'],
    'Robo a Negocio': ['robo a negocio', 'robo a bancos']
}

def clasificar_delito(delito):
    for cat, lista in categorias.items():
        if delito in lista:
            return cat
    return 'Otro'

df_iieg['Categoria'] = df_iieg['delito'].apply(clasificar_delito)

print("Descargando mapa base OSM...")
lugar = "Guadalajara, Jalisco, Mexico"
G = ox.graph_from_place(lugar, network_type='drive')
parques = ox.features_from_place(lugar, tags={'leisure': 'park'})

x_min, x_max = df_iieg['x'].min() - 0.002, df_iieg['x'].max() + 0.002
y_min, y_max = df_iieg['y'].min() - 0.002, df_iieg['y'].max() + 0.002

fig, axes = plt.subplots(2, 2, figsize=(24, 24), facecolor='#0d0d1a')
axes = axes.flatten()

# Estilos de mapa de calor para cada panel
cmaps_heat = ['inferno', 'magma', 'plasma', 'viridis']

for i, (cat, lista) in enumerate(categorias.items()):
    print(f"Procesando: {cat}...")
    ax = axes[i]
    ax.set_facecolor('#0d0d1a')
    
    if not parques.empty:
        parques.plot(ax=ax, color='#1B4332', edgecolor='#0D2B1D', linewidth=0.2, alpha=0.5)
        
    ox.plot_graph(G, ax=ax, show=False, close=False,
                  bgcolor='none', edge_color='#2a2a4a',
                  edge_linewidth=0.2, node_size=0)
                  
    subset = df_iieg[df_iieg['Categoria'] == cat]
    
    # Generar heatmap
    res = 500
    heatmap = np.zeros((res, res))
    for _, row in subset.iterrows():
        ix = int((row['x'] - x_min) / (x_max - x_min) * (res - 1))
        iy = int((row['y'] - y_min) / (y_max - y_min) * (res - 1))
        if 0 <= ix < res and 0 <= iy < res:
            heatmap[iy, ix] += 1
            
    # Filtro gaussiano para suavizar
    heatmap_s = gaussian_filter(heatmap, sigma=8)
    heatmap_m = np.ma.masked_where(heatmap_s < heatmap_s.max() * 0.05, heatmap_s)
    
    try:
        cmap_heat = matplotlib.colormaps[cmaps_heat[i]]
    except AttributeError:
        cmap_heat = plt.cm.get_cmap(cmaps_heat[i])

    ax.imshow(heatmap_m,
              extent=[x_min, x_max, y_min, y_max],
              origin='lower', cmap=cmap_heat,
              alpha=0.6, aspect='auto', zorder=3)
              
    # Graficar los puntos individuales con poca opacidad si no son excesivos
    ax.scatter(subset['x'], subset['y'], c='white', s=8, alpha=0.4, edgecolors='none', zorder=4)
        
    ax.set_title(f"{cat} (N={len(subset):,})", fontsize=24, fontweight='bold', color='white', pad=15)
    ax.set_axis_off()

fig.suptitle("Guadalajara — Análisis de Delitos de Alto Impacto (IIEG 2023)", 
             fontsize=32, fontweight='bold', color='white', y=0.96)

plt.tight_layout(rect=[0, 0, 1, 0.94])
salida = 'mapa_alto_impacto_4paneles.png'
fig.savefig(salida, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
plt.close(fig)
print(f"\n✅ Guardado: '{salida}'")
