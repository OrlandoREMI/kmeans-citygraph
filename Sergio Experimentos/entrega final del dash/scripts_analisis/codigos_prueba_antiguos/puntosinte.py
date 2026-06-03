import matplotlib
matplotlib.use('Agg')
import pandas as pd
import osmnx as ox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

ox.settings.use_cache = True

# ══════════════════════════════════════════════════════════════════
# 1. CARGAR DATOS DELICTIVOS
# ══════════════════════════════════════════════════════════════════
print("Cargando datos delictivos...")
df = pd.read_csv('iieg_2023.csv')
df = df.dropna(subset=['x', 'y'])

colonias = df.groupby('colonia').agg(
    total=('delito', 'count'),
    x=('x', 'mean'),
    y=('y', 'mean')
).reset_index().sort_values('total', ascending=False)

print(f"  → {len(colonias)} colonias cargadas")

# ══════════════════════════════════════════════════════════════════
# 2. DESCARGAR CAPAS BASE OSM
# ══════════════════════════════════════════════════════════════════
lugar = "Guadalajara, Jalisco, Mexico"

print("Descargando red vial...")
G = ox.graph_from_place(lugar, network_type='drive')

print("Descargando capas base...")
parques   = ox.features_from_place(lugar, tags={'leisure': 'park'})
agua      = ox.features_from_place(lugar, tags={'natural': 'water'})
comercial = ox.features_from_place(lugar, tags={'landuse': 'commercial'})

# ══════════════════════════════════════════════════════════════════
# 3. DESCARGAR PUNTOS DE INTERÉS
# ══════════════════════════════════════════════════════════════════

# ── Helper para descargar sin romper si no hay datos ─────────────
import geopandas as gpd

def descargar(lugar, tags, nombre):
    try:
        gdf = ox.features_from_place(lugar, tags=tags)
        print(f"  ✓ {nombre}: {len(gdf)} features")
        return gdf
    except Exception:
        print(f"  ✗ {nombre}: sin datos en OSM, se omite")
        return gpd.GeoDataFrame()

# ── Factores de RIESGO ────────────────────────────────────────────
print("Descargando POIs de riesgo...")
bares        = descargar(lugar, {'amenity': ['bar', 'pub', 'nightclub']},  'Bares / Antros')
atms         = descargar(lugar, {'amenity': 'atm'},                        'Cajeros ATM')
bancos       = descargar(lugar, {'amenity': 'bank'},                       'Bancos')
joyerias     = descargar(lugar, {'shop': 'jewelry'},                       'Joyerías')
gasolineras  = descargar(lugar, {'amenity': 'fuel'},                       'Gasolineras')
estac        = descargar(lugar, {'amenity': 'parking'},                    'Estacionamientos')
casas_empen  = descargar(lugar, {'shop': 'pawnbroker'},                    'Casas de empeño')
licorerias   = descargar(lugar, {'shop': ['alcohol', 'wine']},             'Licoreras')
tiendas_conv = descargar(lugar, {'shop': 'convenience'},                   'Tiendas conveniencia')
farmacias    = descargar(lugar, {'amenity': 'pharmacy'},                   'Farmacias')
hoteles      = descargar(lugar, {'tourism': ['hotel', 'motel']},           'Hoteles / Moteles')

# ── Factores DISUASORES ───────────────────────────────────────────
print("Descargando POIs disuasores...")
policia    = descargar(lugar, {'amenity': 'police'},              'Policía')
hospitales = descargar(lugar, {'amenity': ['hospital', 'clinic']},'Hospitales / Clínicas')
semaforos  = descargar(lugar, {'highway': 'traffic_signals'},     'Semáforos')
bomberos   = descargar(lugar, {'amenity': 'fire_station'},        'Bomberos')
camaras    = descargar(lugar, {'man_made': 'surveillance'},       'Cámaras vigilancia')

print("  → Descarga completada")

# ══════════════════════════════════════════════════════════════════
# 4. HELPER: EXTRAER CENTROIDES
# ══════════════════════════════════════════════════════════════════
def solo_puntos(gdf):
    if gdf.empty:
        return gdf
    pts = gdf.copy()
    pts['geometry'] = pts['geometry'].centroid
    return pts[pts.geometry.geom_type == 'Point']

def plotear_poi(gdf, ax, color, marker, size, zorder, alpha=0.88, edge='white', lw=0.5):
    if gdf.empty:
        return 0
    pts = solo_puntos(gdf)
    if pts.empty:
        return 0
    pts.plot(ax=ax, color=color, markersize=size, marker=marker,
             zorder=zorder, alpha=alpha, edgecolors=edge, linewidths=lw)
    return len(pts)

# ══════════════════════════════════════════════════════════════════
# 5. DIBUJAR
# ══════════════════════════════════════════════════════════════════
print("Dibujando mapa...")
fig, ax = plt.subplots(figsize=(26, 26), facecolor='#0d0d1a')
ax.set_facecolor('#0d0d1a')

# ── Capas base ────────────────────────────────────────────────────
if not comercial.empty:
    comercial.plot(ax=ax, color='#25254A', alpha=0.45, linewidth=0)
if not parques.empty:
    parques.plot(ax=ax, color='#1B4332', edgecolor='#0D2B1D', linewidth=0.3, alpha=0.65)
if not agua.empty:
    agua.plot(ax=ax, color='#0A2540', edgecolor='#0A1F33', linewidth=0.3, alpha=0.85)

# Red vial
ox.plot_graph(G, ax=ax, show=False, close=False,
              bgcolor='none', edge_color='#2a2a4a',
              edge_linewidth=0.3, node_size=0)

# ── Heatmap delictivo ─────────────────────────────────────────────
resolucion = 600
x_min, x_max = df['x'].min() - 0.002, df['x'].max() + 0.002
y_min, y_max = df['y'].min() - 0.002, df['y'].max() + 0.002

heatmap = np.zeros((resolucion, resolucion))
for _, row in df.iterrows():
    ix = int((row['x'] - x_min) / (x_max - x_min) * (resolucion - 1))
    iy = int((row['y'] - y_min) / (y_max - y_min) * (resolucion - 1))
    if 0 <= ix < resolucion and 0 <= iy < resolucion:
        heatmap[iy, ix] += 1

heatmap_suave = gaussian_filter(heatmap, sigma=10)

# Enmascarar zonas sin datos para que el fondo no se tape
heatmap_masked = np.ma.masked_where(heatmap_suave < heatmap_suave.max() * 0.08, heatmap_suave)

cmap_heat = plt.cm.get_cmap('inferno')
hm = ax.imshow(
    heatmap_masked,
    extent=[x_min, x_max, y_min, y_max],
    origin='lower',
    cmap=cmap_heat,
    alpha=0.38,
    aspect='auto',
    zorder=3
)

cbar = plt.colorbar(hm, ax=ax, fraction=0.025, pad=0.01, shrink=0.4)
cbar.set_label('Densidad delictiva', color='white', fontsize=11)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
cbar.outline.set_edgecolor('#444466')

# Etiquetas top 8 colonias
for _, row in colonias.head(8).iterrows():
    ax.annotate(
        row['colonia'].title(),
        xy=(row['x'], row['y']),
        fontsize=7.5, color='white', fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#000000', alpha=0.65, edgecolor='none'),
        zorder=10
    )

# ── POIs: Factores de RIESGO (sobre el heatmap) ───────────────────
n_bares   = plotear_poi(bares,       ax, '#FF6B6B', 'o', 14, 6)
n_atms    = plotear_poi(atms,        ax, '#FFD700', 'D', 12, 6)
n_bancos  = plotear_poi(bancos,      ax, '#FFA500', 's', 13, 6)
n_gas     = plotear_poi(gasolineras, ax, '#FF4500', '^', 13, 6)
n_joy     = plotear_poi(joyerias,    ax, '#FF69B4', '*', 18, 7)
n_estac   = plotear_poi(estac,       ax, '#6A6ABA', 's',  8, 4, alpha=0.50)
n_licore  = plotear_poi(licorerias,  ax, '#DA70D6', 'o', 12, 6)
n_conv    = plotear_poi(tiendas_conv,ax, '#87CEEB', 'o',  9, 5, alpha=0.65)
n_farm    = plotear_poi(farmacias,   ax, '#90EE90', '+', 12, 5, alpha=0.7, edge='none', lw=0)
n_hotel   = plotear_poi(hoteles,     ax, '#CD853F', 'h', 14, 6)
n_empen   = plotear_poi(casas_empen, ax, '#FF1493', 'v', 14, 7)

# ── POIs: Factores DISUASORES (encima de todo) ────────────────────
n_pol     = plotear_poi(policia,     ax, '#00FF7F', 's', 20, 9)
n_hosp    = plotear_poi(hospitales,  ax, '#00CED1', 'P', 18, 8)
n_bomb    = plotear_poi(bomberos,    ax, '#FF6347', 'D', 18, 8)
n_cam     = plotear_poi(camaras,     ax, '#E0E0FF', '^',  9, 8, alpha=0.7)

# ══════════════════════════════════════════════════════════════════
# 6. LEYENDA (tres bloques)
# ══════════════════════════════════════════════════════════════════
def entry(marker, color, label, ms=9):
    return Line2D([0],[0], marker=marker, color='w',
                  markerfacecolor=color, markersize=ms,
                  markeredgecolor='white', markeredgewidth=0.4, label=label)

leg_riesgo = [
    entry('o', '#FF6B6B', f'Bares / Antros ({n_bares})'),
    entry('D', '#FFD700', f'Cajeros ATM ({n_atms})'),
    entry('s', '#FFA500', f'Bancos ({n_bancos})'),
    entry('^', '#FF4500', f'Gasolineras ({n_gas})'),
    entry('*', '#FF69B4', f'Joyerías ({n_joy})', ms=12),
    entry('o', '#DA70D6', f'Licoreras ({n_licore})'),
    entry('v', '#FF1493', f'Casas de empeño ({n_empen})'),
    entry('h', '#CD853F', f'Hoteles / Moteles ({n_hotel})'),
    entry('o', '#87CEEB', f'Tiendas conv. ({n_conv})'),
    entry('+', '#90EE90', f'Farmacias ({n_farm})'),
    entry('s', '#6A6ABA', f'Estacionamientos ({n_estac})'),
]

leg_dis = [
    entry('s', '#00FF7F', f'Policía ({n_pol})'),
    entry('P', '#00CED1', f'Hospitales / Clínicas ({n_hosp})'),
    entry('D', '#FF6347', f'Bomberos ({n_bomb})'),
    entry('^', '#E0E0FF', f'Cámaras vigilancia ({n_cam})'),
]

leg_base = [
    Patch(facecolor='#1B4332', edgecolor='#0D2B1D', label='Parques'),
    Patch(facecolor='#0A2540', edgecolor='#0A1F33', label='Agua'),
    Patch(facecolor='#25254A', label='Zona comercial'),
    Line2D([0],[0], color='#2a2a4a', linewidth=1.5, label='Red vial'),
    Patch(facecolor='#FF4500', alpha=0.5, label='Densidad delictiva (heatmap)'),
]

kw = dict(fontsize=9.5, framealpha=0.88, facecolor='#0d0d1a',
          labelcolor='white', borderpad=0.8)

l1 = ax.legend(handles=leg_riesgo, loc='lower left',
               title='⚠  Factores de riesgo', title_fontsize=11, **kw)
l1.get_title().set_color('#FF6B6B')

l2 = ax.legend(handles=leg_dis, loc='lower center',
               title='🛡  Disuasores', title_fontsize=11, **kw)
l2.get_title().set_color('#00FF7F')

l3 = ax.legend(handles=leg_base, loc='lower right',
               title='Capas base', title_fontsize=11, **kw)
l3.get_title().set_color('#AAAAFF')

ax.add_artist(l1)
ax.add_artist(l2)

ax.set_title("Guadalajara, Jalisco — Puntos de Interés + Densidad Delictiva 2023",
             fontsize=19, fontweight='bold', pad=16, color='white')
ax.set_axis_off()

salida = "guadalajara_poi_heatmap.png"
fig.savefig(salida, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
plt.close(fig)
print(f"\n✅ Guardado como '{salida}'")
print(f"\nPOIs encontrados:")
print(f"  Riesgo   → bares:{n_bares} | ATMs:{n_atms} | bancos:{n_bancos} | gasolineras:{n_gas} | joyerías:{n_joy}")
print(f"             licoreras:{n_licore} | empeños:{n_empen} | hoteles:{n_hotel} | conv:{n_conv} | farmacias:{n_farm} | estac:{n_estac}")
print(f"  Disuasor → policía:{n_pol} | hospitales:{n_hosp} | bomberos:{n_bomb} | cámaras:{n_cam} | semáforos:{n_sem}")