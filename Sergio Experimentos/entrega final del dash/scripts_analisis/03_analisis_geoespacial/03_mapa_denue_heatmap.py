"""
SCRIPT 3 — Mapa combinado: DENUE real + heatmap ENVIPE + POIs OSM
==================================================================
Inputs:
  - iieg_2023.csv        → delitos con coordenadas (heatmap)
  - gdl_denue.csv        → establecimientos DENUE clasificados
  - gdl_percepcion.csv   → percepción de seguridad (opcional, sin coordenadas exactas)

Diferencia con versión anterior:
  - Los POIs ya NO vienen de OSM: vienen del DENUE (datos reales INEGI)
  - Mucho más preciso y completo para Guadalajara
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import osmnx as ox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

ox.settings.use_cache = True

# ══════════════════════════════════════════════════════════════════════════════
# 1. CARGAR DATOS
# ══════════════════════════════════════════════════════════════════════════════
print("Cargando datos...")
df_heat = pd.read_csv('iieg_2023.csv').dropna(subset=['x', 'y'])
denue   = pd.read_csv('gdl_denue.csv')

# Separar categorías de riesgo y disuasores
RIESGO = ['Bar / Cantina', 'Antro / Discoteca', 'Cajero ATM',
          'Joyería / Relojería', 'Gasolinera', 'Hotel / Motel',
          'Casa de empeño', 'Licorería', 'Casino / Apuestas',
          'Estacionamiento', 'Tienda conveniencia']

DISUASORES = ['Hospital / Clínica', 'Escuela', 'Policía / Seguridad']

# Configuración visual por categoría: (color, marker, tamaño, zorder)
CONFIG_CAT = {
    'Bar / Cantina':        ('#FF6B6B', 'o', 8,  6),
    'Antro / Discoteca':    ('#FF0055', 'o', 14, 7),
    'Cajero ATM':           ('#FFD700', 'D', 9,  6),
    'Banco / Financiero':   ('#FFA500', 's', 9,  6),
    'Joyería / Relojería':  ('#FF69B4', '*', 13, 7),
    'Gasolinera':           ('#FF4500', '^', 10, 6),
    'Hotel / Motel':        ('#CD853F', 'h', 10, 6),
    'Casa de empeño':       ('#FF1493', 'v', 11, 7),
    'Licorería':            ('#DA70D6', 'o', 9,  6),
    'Casino / Apuestas':    ('#FF6600', 'P', 12, 7),
    'Estacionamiento':      ('#6A6ABA', 's', 6,  4),
    'Tienda conveniencia':  ('#87CEEB', 'o', 7,  5),
    'Supermercado':         ('#98D8C8', 's', 8,  5),
    'Hospital / Clínica':   ('#00CED1', 'P', 13, 8),
    'Escuela':              ('#F1C40F', '^', 8,  5),
    'Policía / Seguridad':  ('#00FF7F', 's', 14, 9),
}

# ══════════════════════════════════════════════════════════════════════════════
# 2. MAPA BASE OSM
# ══════════════════════════════════════════════════════════════════════════════
print("Descargando mapa base OSM...")
lugar = "Guadalajara, Jalisco, Mexico"
G       = ox.graph_from_place(lugar, network_type='drive')
parques = ox.features_from_place(lugar, tags={'leisure': 'park'})
agua    = ox.features_from_place(lugar, tags={'natural': 'water'})

# ══════════════════════════════════════════════════════════════════════════════
# 3. HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
print("Generando heatmap...")
res = 600
x_min, x_max = df_heat['x'].min() - 0.002, df_heat['x'].max() + 0.002
y_min, y_max = df_heat['y'].min() - 0.002, df_heat['y'].max() + 0.002

heatmap = np.zeros((res, res))
for _, row in df_heat.iterrows():
    ix = int((row['x'] - x_min) / (x_max - x_min) * (res - 1))
    iy = int((row['y'] - y_min) / (y_max - y_min) * (res - 1))
    if 0 <= ix < res and 0 <= iy < res:
        heatmap[iy, ix] += 1

heatmap_s = gaussian_filter(heatmap, sigma=10)
heatmap_m = np.ma.masked_where(heatmap_s < heatmap_s.max() * 0.08, heatmap_s)

# ══════════════════════════════════════════════════════════════════════════════
# 4. DIBUJAR
# ══════════════════════════════════════════════════════════════════════════════
print("Dibujando mapa...")
fig, ax = plt.subplots(figsize=(26, 26), facecolor='#0d0d1a')
ax.set_facecolor('#0d0d1a')

# Capas base
if not parques.empty:
    parques.plot(ax=ax, color='#1B4332', edgecolor='#0D2B1D', linewidth=0.3, alpha=0.65)
if not agua.empty:
    agua.plot(ax=ax, color='#0A2540', edgecolor='#0A1F33', linewidth=0.3, alpha=0.85)

ox.plot_graph(G, ax=ax, show=False, close=False,
              bgcolor='none', edge_color='#2a2a4a',
              edge_linewidth=0.3, node_size=0)

# Heatmap
cmap_heat = plt.cm.get_cmap('inferno')
ax.imshow(heatmap_m,
          extent=[x_min, x_max, y_min, y_max],
          origin='lower', cmap=cmap_heat,
          alpha=0.38, aspect='auto', zorder=3)

# ── Pintar POIs desde DENUE ───────────────────────────────────────────────────
conteos = {}
# Primero estacionamientos (fondo)
for cat in ['Estacionamiento', 'Tienda conveniencia', 'Supermercado', 'Escuela']:
    subset = denue[denue['categoria'] == cat].dropna(subset=['latitud','longitud'])
    if subset.empty:
        conteos[cat] = 0
        continue
    cfg = CONFIG_CAT.get(cat, ('#AAAAAA', 'o', 6, 4))
    ax.scatter(subset['longitud'], subset['latitud'],
               c=cfg[0], marker=cfg[1], s=cfg[2]**2 * 0.6,
               zorder=cfg[3], alpha=0.5, edgecolors='none')
    conteos[cat] = len(subset)

# Luego riesgo principal
for cat in ['Bar / Cantina', 'Licorería', 'Gasolinera', 'Hotel / Motel',
            'Banco / Financiero', 'Cajero ATM']:
    subset = denue[denue['categoria'] == cat].dropna(subset=['latitud','longitud'])
    if subset.empty:
        conteos[cat] = 0
        continue
    cfg = CONFIG_CAT.get(cat, ('#AAAAAA', 'o', 8, 6))
    ax.scatter(subset['longitud'], subset['latitud'],
               c=cfg[0], marker=cfg[1], s=cfg[2]**2,
               zorder=cfg[3], alpha=0.85, edgecolors='white', linewidths=0.4)
    conteos[cat] = len(subset)

# Alto valor / alto riesgo encima
for cat in ['Joyería / Relojería', 'Casa de empeño', 'Antro / Discoteca',
            'Casino / Apuestas']:
    subset = denue[denue['categoria'] == cat].dropna(subset=['latitud','longitud'])
    if subset.empty:
        conteos[cat] = 0
        continue
    cfg = CONFIG_CAT.get(cat, ('#AAAAAA', '*', 12, 7))
    ax.scatter(subset['longitud'], subset['latitud'],
               c=cfg[0], marker=cfg[1], s=cfg[2]**2 * 1.2,
               zorder=cfg[3], alpha=0.9, edgecolors='white', linewidths=0.5)
    conteos[cat] = len(subset)

# Disuasores al frente
for cat in ['Hospital / Clínica', 'Policía / Seguridad']:
    subset = denue[denue['categoria'] == cat].dropna(subset=['latitud','longitud'])
    if subset.empty:
        conteos[cat] = 0
        continue
    cfg = CONFIG_CAT.get(cat, ('#00FF7F', 's', 14, 9))
    ax.scatter(subset['longitud'], subset['latitud'],
               c=cfg[0], marker=cfg[1], s=cfg[2]**2 * 1.5,
               zorder=cfg[3], alpha=0.95, edgecolors='white', linewidths=0.6)
    conteos[cat] = len(subset)

# ── Leyenda ───────────────────────────────────────────────────────────────────
def entry(marker, color, label, ms=9):
    return Line2D([0],[0], marker=marker, color='w',
                  markerfacecolor=color, markersize=ms,
                  markeredgecolor='white', markeredgewidth=0.3, label=label)

leg_r = [
    entry('o',  '#FF6B6B', f"Bares / Cantinas ({conteos.get('Bar / Cantina', 0):,})"),
    entry('o',  '#FF0055', f"Antros / Discotecas ({conteos.get('Antro / Discoteca', 0):,})"),
    entry('D',  '#FFD700', f"Cajeros ATM ({conteos.get('Cajero ATM', 0):,})"),
    entry('s',  '#FFA500', f"Bancos ({conteos.get('Banco / Financiero', 0):,})"),
    entry('*',  '#FF69B4', f"Joyerías ({conteos.get('Joyería / Relojería', 0):,})", ms=12),
    entry('^',  '#FF4500', f"Gasolineras ({conteos.get('Gasolinera', 0):,})"),
    entry('h',  '#CD853F', f"Hoteles / Moteles ({conteos.get('Hotel / Motel', 0):,})"),
    entry('v',  '#FF1493', f"Casas de empeño ({conteos.get('Casa de empeño', 0):,})"),
    entry('o',  '#DA70D6', f"Licorerías ({conteos.get('Licorería', 0):,})"),
    entry('P',  '#FF6600', f"Casinos / Apuestas ({conteos.get('Casino / Apuestas', 0):,})"),
    entry('o',  '#87CEEB', f"Tiendas conv. ({conteos.get('Tienda conveniencia', 0):,})"),
    entry('s',  '#6A6ABA', f"Estacionamientos ({conteos.get('Estacionamiento', 0):,})"),
]

leg_d = [
    entry('P',  '#00CED1', f"Hospitales / Clínicas ({conteos.get('Hospital / Clínica', 0):,})"),
    entry('^',  '#F1C40F', f"Escuelas ({conteos.get('Escuela', 0):,})"),
    entry('s',  '#00FF7F', f"Seguridad privada ({conteos.get('Policía / Seguridad', 0):,})"),
]

leg_b = [
    Patch(facecolor='#1B4332', edgecolor='#0D2B1D', label='Parques'),
    Patch(facecolor='#0A2540', edgecolor='#0A1F33', label='Agua'),
    Line2D([0],[0], color='#2a2a4a', linewidth=1.5, label='Red vial'),
    Patch(facecolor='#FF4500', alpha=0.4, label='Densidad delictiva'),
]

kw = dict(fontsize=9, framealpha=0.9, facecolor='#0d0d1a',
          labelcolor='white', borderpad=0.8)

l1 = ax.legend(handles=leg_r, loc='lower left',
               title='⚠  Factores de riesgo (DENUE)', title_fontsize=10, **kw)
l1.get_title().set_color('#FF6B6B')

l2 = ax.legend(handles=leg_d, loc='lower center',
               title='🛡  Disuasores', title_fontsize=10, **kw)
l2.get_title().set_color('#00FF7F')

l3 = ax.legend(handles=leg_b, loc='lower right',
               title='Capas base', title_fontsize=10, **kw)
l3.get_title().set_color('#AAAAFF')

ax.add_artist(l1)
ax.add_artist(l2)

ax.set_title("Guadalajara — Establecimientos DENUE + Densidad Delictiva (IIEG 2023)",
             fontsize=18, fontweight='bold', color='white', pad=14)
ax.set_axis_off()

salida = 'guadalajara_denue_heatmap.png'
fig.savefig(salida, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
plt.close(fig)
print(f"\n✅ Guardado: '{salida}'")
print(f"\nEstablecimientos pintados:")
for k, v in sorted(conteos.items(), key=lambda x: -x[1]):
    print(f"  {k:<30}: {v:,}")