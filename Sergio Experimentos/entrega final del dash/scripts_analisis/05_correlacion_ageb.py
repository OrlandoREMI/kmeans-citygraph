"""
SCRIPT 5 — Cruce espacial (AGEB) y Correlación: Delitos Alto Impacto vs DENUE
=============================================================================
Este script:
1. Carga crímenes (IIEG) y los agrupa en 4 categorías de alto impacto.
2. Carga los establecimientos de negocios (DENUE).
3. Realiza un cruce espacial (Spatial Join) de ambos contra los polígonos AGEB.
4. Calcula la correlación de Spearman por AGEB para ver qué negocios están
   más asociados con los delitos de alto impacto.
5. Genera un Heatmap de correlaciones.
"""

import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

print("Cargando shapefile de AGEB...")
# Leemos el shapefile de las AGEBs
ageb = gpd.read_file('guadalajara_AGEB/2025_14039_A07052026_1549.shp')
if ageb.crs is None:
    ageb = ageb.set_crs("EPSG:4326")
else:
    ageb = ageb.to_crs("EPSG:4326")

print("Cargando y clasificando datos de IIEG (Delitos)...")
df_iieg = pd.read_csv('iieg_2023.csv').dropna(subset=['x', 'y'])

categorias_delito = {
    'Homicidios y Feminicidios': ['homicidio doloso', 'feminicidio'],
    'Lesiones Dolosas': ['lesiones dolosas'],
    'Robo a Persona': ['robo a persona', 'robo a cuentahabientes'],
    'Robo a Negocio': ['robo a negocio', 'robo a bancos']
}

def clasificar_delito(delito):
    for cat, lista in categorias_delito.items():
        if delito in lista:
            return cat
    return None

df_iieg['Categoria_Delito'] = df_iieg['delito'].apply(clasificar_delito)
df_iieg = df_iieg.dropna(subset=['Categoria_Delito'])

# Convertir a GeoDataFrame
gdf_iieg = gpd.GeoDataFrame(
    df_iieg, 
    geometry=gpd.points_from_xy(df_iieg.x, df_iieg.y), 
    crs="EPSG:4326"
)

print("Cargando datos DENUE (Negocios)...")
df_denue = pd.read_csv('gdl_denue.csv').dropna(subset=['longitud', 'latitud', 'categoria'])

# Excluir la categoría 'Otro' para tener un análisis más preciso
df_denue = df_denue[df_denue['categoria'] != 'Otro']

# Convertir a GeoDataFrame
gdf_denue = gpd.GeoDataFrame(
    df_denue, 
    geometry=gpd.points_from_xy(df_denue.longitud, df_denue.latitud), 
    crs="EPSG:4326"
)

print("Realizando cruces espaciales (sjoin)...")
# sjoin de delitos con AGEB (Asigna a cada delito el índice del AGEB donde ocurrió)
join_delitos = gpd.sjoin(gdf_iieg, ageb, how="inner", predicate="intersects")
# Contamos cuántos delitos de cada categoría ocurrieron en cada AGEB
conteo_delitos = pd.crosstab(join_delitos['index_right'], join_delitos['Categoria_Delito'])

# sjoin de denue con AGEB (Asigna a cada negocio el índice del AGEB)
join_denue = gpd.sjoin(gdf_denue, ageb, how="inner", predicate="intersects")
# Contamos cuántos negocios de cada categoría hay en cada AGEB
conteo_denue = pd.crosstab(join_denue['index_right'], join_denue['categoria'])

print("Calculando matriz de correlación...")
# Unimos ambos conteos por el índice del AGEB
df_corr_base = conteo_delitos.join(conteo_denue, how='inner').fillna(0)

# Calculamos correlación de Spearman (ideal para datos de conteo/no normales)
corr_matrix = df_corr_base.corr(method='spearman')

# Extraer el bloque cruzado: Filas = Categorías DENUE, Columnas = Categorías de Delito
cols_delitos = list(categorias_delito.keys())
cols_denue = list(conteo_denue.columns)

# Filtrar solo si las columnas existen en la matriz
cols_delitos = [c for c in cols_delitos if c in corr_matrix.columns]
cols_denue = [c for c in cols_denue if c in corr_matrix.columns]

corr_cruzada = corr_matrix.loc[cols_denue, cols_delitos]

# Ordenamos los negocios por su correlación promedio para mejor estética visual
corr_cruzada['promedio'] = corr_cruzada.mean(axis=1)
corr_cruzada = corr_cruzada.sort_values(by='promedio', ascending=False).drop(columns=['promedio'])

print("Generando visualización...")
# Usamos un estilo de "dark mode"
plt.figure(figsize=(12, 10), facecolor='#0d0d1a')
ax = plt.gca()
ax.set_facecolor('#0d0d1a')

# Mapa de calor (Heatmap)
sns.heatmap(corr_cruzada, annot=True, fmt=".2f", cmap="coolwarm", center=0, 
            cbar_kws={'label': 'Coeficiente de Spearman (r)'},
            linewidths=0.5, linecolor='#0d0d1a', ax=ax, 
            annot_kws={"size": 11, "weight": "bold"})

# Formato estético
ax.tick_params(colors='white', labelsize=12)
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')

# Configurar colorbar
cbar = ax.collections[0].colorbar
cbar.ax.yaxis.set_tick_params(color='white')
cbar.ax.yaxis.set_ticklabels(cbar.ax.yaxis.get_ticklabels(), color='white')
cbar.set_label('Correlación de Spearman (r)', color='white', size=12, weight='bold')

plt.title("Correlación (por AGEB): Negocios vs Delitos de Alto Impacto\nGuadalajara", 
          fontsize=18, fontweight='bold', color='white', pad=20)
plt.xticks(rotation=20, ha='right', weight='bold')
plt.yticks(rotation=0, weight='bold')
plt.tight_layout()

salida = '05_correlacion_ageb_alto_impacto.png'
plt.savefig(salida, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
plt.close()

print(f"✅ ¡Completado! El análisis ha sido guardado en: '{salida}'")
