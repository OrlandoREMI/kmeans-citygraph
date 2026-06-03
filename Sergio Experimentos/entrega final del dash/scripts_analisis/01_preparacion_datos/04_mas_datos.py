"""
SCRIPT 4 — Geoestadística Final: Ingresos, Normalización y Hotspots
==================================================================
Versión compatible con: RESAGEBURB_14CSV20.csv (ITER por AGEB)
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from libpysal.weights import Queen
from esda.moran import Moran_Local
from splot.esda import lisa_cluster
import warnings
warnings.filterwarnings('ignore')

# 1. CARGAR Y PREPARAR DATOS SOCIOECONÓMICOS (ITER)
# ----------------------------------------------------------------
# Línea corregida con codificación para español
df_iter = pd.read_csv('RESAGEBURB_14CSV20.csv', encoding='latin-1')

# Filtrar para Guadalajara (MUN 39) y niveles de Total AGEB
df_gdl = df_iter[(df_iter['MUN'] == 39) & (df_iter['NOM_LOC'] == 'Total AGEB urbana')].copy()

# Crear CVEGEO de 13 dígitos
df_gdl['CVEGEO'] = (
    df_gdl['ENTIDAD'].astype(str).str.zfill(2) +
    df_gdl['MUN'].astype(str).str.zfill(3) +
    df_gdl['LOC'].astype(str).str.zfill(4) +
    df_gdl['AGEB'].astype(str).str.zfill(4)
)

# Limpiar columnas numéricas (manejar el '*' de confidencialidad del INEGI)
for col in ['POBTOT', 'GRAPROES']:
    df_gdl[col] = pd.to_numeric(df_gdl[col], errors='coerce').fillna(0)

df_socio = df_gdl[['CVEGEO', 'POBTOT', 'GRAPROES']]

# 2. CARGAR MAPA Y UNIR DATOS
# ----------------------------------------------------------------
print("Cargando mapa de AGEBs y uniendo con Censo...")
# Corrección de la ruta según tu carpeta real
agebs = gpd.read_file('guadalajara_AGEB/2025_14039_A07052026_1549.shp')
# Unir geometría con datos del censo
agebs = agebs.merge(df_socio, on='CVEGEO', how='inner')

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
agebs = agebs.merge(crime_counts, on='CVEGEO', how='left').fillna(0)

# 4. CÁLCULO DE INDICADORES (Normalización e Ingresos)
# ----------------------------------------------------------------
# Delitos por cada 1,000 habitantes (para evitar sesgo de zonas muy pobladas)
agebs['delitos_1k'] = (agebs['total_delitos'] / agebs['POBTOT'] * 1000).replace([float('inf'), -float('inf')], 0).fillna(0)

# 5. ANÁLISIS DE HOTSPOTS (LISA / Moran Local)
# ----------------------------------------------------------------
print("Generando análisis de autocorrelación espacial...")
# Eliminar AGEBs aislados sin vecinos para el cálculo de pesos
agebs_clean = agebs[agebs['POBTOT'] > 0].copy()
w = Queen.from_dataframe(agebs_clean)
w.transform = 'R'

moran_loc = Moran_Local(agebs_clean['total_delitos'], w)

# 6. VISUALIZACIÓN MEJORADA CON ETIQUETAS CLARAS
# ----------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 12), facecolor='#0d0d1a')

# --- MAPA A: ESCOLARIDAD (PROXY DE INGRESOS) ---
# Usamos un esquema de color que evoque "riqueza" o estabilidad (YlGn)
agebs_clean.plot(column='GRAPROES', cmap='YlGn', legend=True, ax=ax1, 
                 scheme='quantiles', k=5,  # Divide en 5 niveles iguales
                 legend_kwds={'title': "Años de Escolaridad Promedio", 
                              'loc': 'lower right'})
ax1.set_title("Nivel Socioeconómico (Ingresos estimados)", color='white', fontsize=18, pad=20)
ax1.set_axis_off()

# --- MAPA B: HOTSPOTS DE CRIMEN (LISA) ---
# Aquí ajustamos manualmente para que la leyenda sea comprensible
from matplotlib.colors import ListedColormap

# Definimos colores estándar para LISA:
# 0: No significativo (Gris), 1: High-High (Rojo), 2: Low-Low (Azul), 
# 3: Low-High (Diamante), 4: High-Low (Diamante)
lisa_colors = ['#eeeeee', '#d7191c', '#2b83ba', '#abd9e9', '#fdae61']
lisa_labels = ['No significativo', 'Hotspot (Mucho Crimen)', 'Zona Segura (Poco Crimen)', 
               'Bajo rodeado de Alto', 'Alto rodeado de Bajo']

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

# Ajustes estéticos finales para el mapa de Hotspots
ax2.set_title("Mapa de Calidad Estadística (Hotspots)", color='white', fontsize=18, pad=20)
ax2.set_axis_off()

# Añadir un texto explicativo global
plt.suptitle("Análisis Geoespacial Guadalajara: Correlación Crimen vs Educación\n", 
             color='white', fontsize=24, fontweight='bold', y=0.98)

# Pie de página con créditos (usando tus colaboradores)
plt.figtext(0.5, 0.02, "Análisis: Sergio Bernardo Robles | Datos: INEGI 2020 + IIEG 2023", 
            ha="center", color="gray", fontsize=12)

plt.savefig('analisis_final_etiquetado.png', dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
print("✅ Mapa con etiquetas claras generado: analisis_final_etiquetado.png")