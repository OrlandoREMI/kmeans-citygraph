"""
SCRIPT 5 — Geoestadística y Entorno Urbano: Infraestructura y Hotspots
======================================================================
Este script utiliza la base del Inventario Nacional de Viviendas (Entorno Urbano - Frentes de Manzana)
para explorar las carencias en el alumbrado público (ALUMPUB_D) y su relación con
los hotspots de criminalidad en la ciudad de Guadalajara.
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from libpysal.weights import Queen
from esda.moran import Moran_Local
from splot.esda import lisa_cluster
from matplotlib.colors import ListedColormap
import warnings

warnings.filterwarnings('ignore')

# 1. CARGAR Y PREPARAR DATOS DEL ENTORNO URBANO (FRENTES)
# ----------------------------------------------------------------
print("Leyendo y procesando el inventario de frentes de manzana...")
# Leemos la base como string para no perder ceros a la izquierda en claves
df_frentes = pd.read_csv('inv_frentes.csv', dtype=str)

# Filtramos para el municipio de Guadalajara (039)
df_gdl = df_frentes[df_frentes['CVE_MUN'] == '039'].copy()

# El CVEGEO en esta base viene a nivel frente (16 dígitos).
# Extraemos los primeros 13 para obtener el AGEB (ENT+MUN+LOC+AGEB)
df_gdl['CVEGEO_AGEB'] = df_gdl['CVEGEO'].str[:13]

# Queremos analizar la falta de alumbrado público.
# Creamos una variable binaria: 1 si NO dispone de alumbrado, 0 en otro caso
df_gdl['sin_alumbrado'] = (df_gdl['ALUMPUB_D'] == 'No dispone').astype(int)

# Agrupamos por AGEB: sumamos cuántos frentes no tienen alumbrado y contamos el total de frentes
agrupado_alumbrado = df_gdl.groupby('CVEGEO_AGEB').agg(
    frentes_sin_alumbrado=('sin_alumbrado', 'sum'),
    total_frentes=('CVEGEO_AGEB', 'count')
).reset_index()

# Calculamos el porcentaje de frentes sin alumbrado por AGEB
agrupado_alumbrado['pct_sin_alumbrado'] = (agrupado_alumbrado['frentes_sin_alumbrado'] / agrupado_alumbrado['total_frentes']) * 100

# 2. CARGAR MAPA Y UNIR DATOS DE ENTORNO URBANO
# ----------------------------------------------------------------
print("Cargando mapa de AGEBs y uniendo con datos urbanos...")
agebs = gpd.read_file('guadalajara_AGEB/2025_14039_A07052026_1549.shp')

# Renombramos la columna para el merge y unimos
agrupado_alumbrado = agrupado_alumbrado.rename(columns={'CVEGEO_AGEB': 'CVEGEO'})
agebs = agebs.merge(agrupado_alumbrado, on='CVEGEO', how='inner')

# 3. ESPACIALIZAR DELITOS (IIEG 2023)
# ----------------------------------------------------------------
print("Contando delitos por cada AGEB...")
delitos = pd.read_csv('iieg_2023.csv')
gdf_delitos = gpd.GeoDataFrame(
    delitos, geometry=gpd.points_from_xy(delitos.x, delitos.y), crs="EPSG:4326"
)

# Homologar proyecciones
agebs = agebs.to_crs(gdf_delitos.crs)

# Spatial Join: ¿En qué AGEB cayó cada delito?
join_delitos = gpd.sjoin(gdf_delitos, agebs, predicate='within')
crime_counts = join_delitos.groupby('CVEGEO').size().rename('total_delitos')

# Unimos los delitos al mapa
agebs = agebs.merge(crime_counts, on='CVEGEO', how='left')
agebs['total_delitos'] = agebs['total_delitos'].fillna(0)

# 4. ANÁLISIS DE HOTSPOTS (LISA / Moran Local) SOBRE EL CRIMEN
# ----------------------------------------------------------------
print("Generando análisis de autocorrelación espacial...")
# Eliminar AGEBs aislados sin vecinos para el cálculo de pesos
agebs_clean = agebs[agebs['total_frentes'] > 0].copy()
w = Queen.from_dataframe(agebs_clean)
w.transform = 'R'

# Analizamos los hotspots de crimen
moran_loc = Moran_Local(agebs_clean['total_delitos'], w)

# 5. VISUALIZACIÓN PROFESIONAL
# ----------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 12), facecolor='#0d0d1a')

# --- MAPA A: PORCENTAJE SIN ALUMBRADO PÚBLICO ---
# Usamos un esquema de color que destaque las deficiencias urbanas (OrRd o Reds)
agebs_clean.plot(column='pct_sin_alumbrado', cmap='OrRd', legend=True, ax=ax1, 
                 scheme='quantiles', k=5,
                 legend_kwds={'title': "% de Calles sin Alumbrado", 
                              'loc': 'lower right'})
ax1.set_title("Carencia de Alumbrado Público", color='white', fontsize=18, pad=20)
ax1.set_axis_off()

# --- MAPA B: HOTSPOTS DE CRIMEN (LISA) ---
lisa_cluster(moran_loc, agebs_clean, p=0.05, ax=ax2, 
             legend_kwds={'loc': 'upper left', 'bbox_to_anchor': (1, 1)})

# Reemplazamos los textos de la leyenda generada por lisa_cluster
if ax2.get_legend():
    label_map = {
        'ns': 'No significativo',
        'HH': 'Hotspot (Mucho Crimen)',
        'LL': 'Zona Segura (Poco Crimen)',
        'LH': 'Bajo rodeado de Alto',
        'HL': 'Alto rodeado de Bajo'
    }
    for text in ax2.get_legend().get_texts():
        orig = text.get_text()
        text.set_text(label_map.get(orig, orig))

ax2.set_title("Hotspots de Criminalidad (Clusters LISA)", color='white', fontsize=18, pad=20)
ax2.set_axis_off()

# Añadir un texto explicativo global
plt.suptitle("Análisis Geoespacial Guadalajara: Alumbrado Público vs Criminalidad\n", 
             color='white', fontsize=24, fontweight='bold', y=0.98)

# Pie de página con créditos
plt.figtext(0.5, 0.02, "Análisis: Sergio Bernardo Robles | Datos: INV Frentes 2020 + IIEG 2023", 
            ha="center", color="gray", fontsize=12)

plt.savefig('analisis_entorno_urbano_gdl.png', dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
print("✅ Proceso completado. Archivo generado: analisis_entorno_urbano_gdl.png")
