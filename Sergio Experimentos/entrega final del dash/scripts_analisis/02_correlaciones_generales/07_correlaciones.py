"""
SCRIPT 7 — Análisis Integral: Alumbrado vs Incidencia Delictiva
==============================================================================
Genera 3 evidencias estadísticas sobre la relación entre infraestructura y crimen:
1. Matrices de Correlación Triangular (Pearson).
2. Gráficos de Dispersión y Correlación No Paramétrica (Spearman).
3. Autocorrelación Espacial Bivariada (Índice de Moran Bivariado).
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import libpysal
from libpysal.weights import Queen
from esda.moran import Moran_BV
import warnings

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------
# 1. CARGAR Y PREPARAR DATOS DEL ENTORNO URBANO (FRENTES)
# ----------------------------------------------------------------
print("1. Procesando inventario de frentes de manzana...")
df_frentes = pd.read_csv('inv_frentes.csv', dtype=str)

df_gdl = df_frentes[df_frentes['CVE_MUN'] == '039'].copy()
df_gdl['CVEGEO_AGEB'] = df_gdl['CVEGEO'].str[:13]
df_gdl['sin_alumbrado'] = (df_gdl['ALUMPUB_D'] == 'No dispone').astype(int)

agrupado_alumbrado = df_gdl.groupby('CVEGEO_AGEB').agg(
    frentes_sin_alumbrado=('sin_alumbrado', 'sum'),
    total_frentes=('CVEGEO_AGEB', 'count')
).reset_index()

agrupado_alumbrado['% Sin Alumbrado'] = (agrupado_alumbrado['frentes_sin_alumbrado'] / agrupado_alumbrado['total_frentes']) * 100
agrupado_alumbrado = agrupado_alumbrado.rename(columns={'CVEGEO_AGEB': 'CVEGEO'})

# ----------------------------------------------------------------
# 2. CARGAR MAPA Y DELITOS (IIEG 2023)
# ----------------------------------------------------------------
print("2. Cargando mapa AGEB y procesando archivo de delitos...")
agebs = gpd.read_file('guadalajara_AGEB/2025_14039_A07052026_1549.shp')
delitos = pd.read_csv('iieg_2023.csv').dropna(subset=['x', 'y'])

gdf_delitos = gpd.GeoDataFrame(
    delitos, geometry=gpd.points_from_xy(delitos.x, delitos.y), crs="EPSG:4326"
)

agebs = agebs.to_crs(gdf_delitos.crs)
join_delitos = gpd.sjoin(gdf_delitos, agebs[['CVEGEO', 'geometry']], predicate='within')

# ----------------------------------------------------------------
# 3. PIVOTEAR DELITOS POR AGEB Y LIMPIAR FORMATO
# ----------------------------------------------------------------
print("3. Estructurando matriz de datos...")
delitos_pivot = join_delitos.pivot_table(index='CVEGEO', columns='delito', aggfunc='size', fill_value=0)
delitos_pivot['Total Delitos'] = delitos_pivot.sum(axis=1)

# Unimos con los datos de alumbrado
df_final = pd.merge(agrupado_alumbrado[['CVEGEO', '% Sin Alumbrado']], delitos_pivot, on='CVEGEO', how='inner')

# Capitalizamos nombres
df_final.columns = [str(c).title() if c not in ['% Sin Alumbrado', 'Total Delitos', 'CVEGEO'] else c for c in df_final.columns]

# Creamos un GeoDataFrame final (esencial para el análisis espacial de Moran)
gdf_final = agebs[['CVEGEO', 'geometry']].drop_duplicates().merge(df_final, on='CVEGEO', how='inner')
gdf_final.set_index('CVEGEO', inplace=True)


# ================================================================
# EVIDENCIA 1: MATRICES DE CORRELACIÓN TRIANGULAR (PEARSON)
# ================================================================
print("4. Generando Matrices de Correlación Triangular...")

delitos_criticos = ['Homicidio Doloso', 'Feminicidio', 'Violacion', 'Abuso Sexual Infantil']
criticos_existentes = [d for d in delitos_criticos if d in gdf_final.columns]

cols_matriz_1 = ['% Sin Alumbrado', 'Total Delitos'] + criticos_existentes
corr_1 = gdf_final[cols_matriz_1].corr()

top_delitos = delitos_pivot.drop(columns=['Total Delitos']).sum().nlargest(12).index.tolist()
top_delitos = [d.title() for d in top_delitos]
cols_matriz_2 = ['% Sin Alumbrado'] + top_delitos
corr_2 = gdf_final[cols_matriz_2].corr()

sns.set_theme(style="white")
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 11))
cmap = sns.color_palette("vlag", as_cmap=True)

mask1 = np.triu(np.ones_like(corr_1, dtype=bool))
sns.heatmap(corr_1, mask=mask1, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, annot=True, fmt=".2f", cbar_kws={"shrink": .7}, ax=ax1)
ax1.set_title("Pearson: Alumbrado vs Delitos Dolosos", fontsize=18, fontweight='bold', pad=20)
ax1.tick_params(axis='x', rotation=45); ax1.tick_params(axis='y', rotation=0)

mask2 = np.triu(np.ones_like(corr_2, dtype=bool))
sns.heatmap(corr_2, mask=mask2, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, annot=True, fmt=".2f", cbar_kws={"shrink": .7}, ax=ax2)
ax2.set_title("Pearson: Alumbrado vs Top 12 Delitos", fontsize=18, fontweight='bold', pad=20)
ax2.tick_params(axis='x', rotation=45); ax2.tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.savefig('01_matriz_triangular.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()


# ================================================================
# EVIDENCIA 2: DISPERSIÓN Y CORRELACIÓN NO PARAMÉTRICA (SPEARMAN)
# ================================================================
print("5. Generando Gráficos de Dispersión y Spearman...")

sns.set_theme(style="whitegrid")
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(18, 7))

# Gráfico 1: Total de Delitos
sns.regplot(data=gdf_final, x='% Sin Alumbrado', y='Total Delitos', 
            scatter_kws={'alpha':0.4, 'color': '#2c3e50'}, 
            line_kws={'color': '#e74c3c', 'linewidth': 2}, ax=ax3)
coef_sp_total, p_val_total = stats.spearmanr(gdf_final['% Sin Alumbrado'], gdf_final['Total Delitos'])
ax3.set_title(f"Dispersión: Alumbrado vs Crimen Total\nSpearman: {coef_sp_total:.3f} (p-value: {p_val_total:.3f})", 
              fontsize=14, fontweight='bold', pad=15)
ax3.set_xlabel("% de Frentes sin Alumbrado")
ax3.set_ylabel("Total de Delitos")

# Gráfico 2: Homicidios Dolosos (Usa el primer delito crítico que exista)
delito_critico_top = criticos_existentes[0] if criticos_existentes else 'Total Delitos'
sns.regplot(data=gdf_final, x='% Sin Alumbrado', y=delito_critico_top, 
            scatter_kws={'alpha':0.4, 'color': '#2c3e50'}, 
            line_kws={'color': '#e74c3c', 'linewidth': 2}, ax=ax4)
coef_sp_hom, p_val_hom = stats.spearmanr(gdf_final['% Sin Alumbrado'], gdf_final[delito_critico_top])
ax4.set_title(f"Dispersión: Alumbrado vs {delito_critico_top}\nSpearman: {coef_sp_hom:.3f} (p-value: {p_val_hom:.3f})", 
              fontsize=14, fontweight='bold', pad=15)
ax4.set_xlabel("% de Frentes sin Alumbrado")
ax4.set_ylabel(f"Incidentes: {delito_critico_top}")

plt.tight_layout()
plt.savefig('02_dispersion_spearman.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()


# ================================================================
# EVIDENCIA 3: AUTOCORRELACIÓN ESPACIAL BIVARIADA (MORAN)
# ================================================================
print("6. Analizando efecto de derrame espacial (Moran Bivariado)...")

# Limpiamos AGEBs sin vecinos para que la matriz Queen no falle
gdf_clean = gdf_final[~gdf_final.geometry.is_empty].copy()
w = Queen.from_dataframe(gdf_clean)
w.transform = 'r'

x = gdf_clean['% Sin Alumbrado'].values
y = gdf_clean['Total Delitos'].values

# Calculamos el Moran Bivariado
moran_bv = Moran_BV(y, x, w)

# Calculamos el retardo espacial (lag) para el gráfico
lag_y = libpysal.weights.lag_spatial(w, y)

fig3, ax5 = plt.subplots(figsize=(8, 8))
sns.regplot(x=x, y=lag_y, scatter_kws={'alpha':0.5, 'color': '#8e44ad'}, 
            line_kws={'color': '#27ae60', 'linewidth': 2}, ax=ax5)

ax5.axvline(x.mean(), color='gray', linestyle='--')
ax5.axhline(lag_y.mean(), color='gray', linestyle='--')

ax5.set_title(f"Efecto Espacial Bivariado (Moran's I: {moran_bv.I:.3f})\n¿Calles oscuras afectan a colonias vecinas? (p-value: {moran_bv.p_sim:.3f})", 
              fontsize=14, fontweight='bold', pad=15)
ax5.set_xlabel("% de Frentes sin Alumbrado en AGEB de Origen", fontsize=12)
ax5.set_ylabel("Total de Delitos en AGEBs VECINAS (Retardo Espacial)", fontsize=12)

# Añadimos cuadrantes interpretativos
ax5.text(x.max()*0.8, lag_y.max()*0.9, "Alto-Alto", color='gray', alpha=0.6)
ax5.text(x.min(), lag_y.min(), "Bajo-Bajo", color='gray', alpha=0.6)

plt.tight_layout()
plt.savefig('03_moran_bivariado.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("\n✅ ANÁLISIS COMPLETADO. Se generaron 3 imágenes:")
print("1. 01_matriz_triangular.png (Relación Lineal Tradicional)")
print("2. 02_dispersion_spearman.png (Validación No Paramétrica)")
print("3. 03_moran_bivariado.png (Validación Geoespacial de Vecindario)")