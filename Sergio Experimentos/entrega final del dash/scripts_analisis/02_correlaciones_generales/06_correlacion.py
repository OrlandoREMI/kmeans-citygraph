"""
SCRIPT 6 — Análisis Integral de Correlación: Alumbrado vs Incidencia Delictiva
===============================================================================
Genera 4 evidencias visuales para demostrar la ausencia (o presencia) de correlación
entre el porcentaje de frentes sin alumbrado público y los distintos tipos de delitos:

  1. Matrices de Correlación Triangular (Pearson)
  2. Cuadrícula de Gráficos de Dispersión (visualmente confirma la nube de puntos)
  3. Heatmap de P-Valores (significancia estadística de cada correlación)
  4. Comparativa Pearson vs Spearman (valida que no hay correlación lineal ni monotónica)
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import scipy.stats as stats
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGAR Y PREPARAR DATOS
# ─────────────────────────────────────────────────────────────────────────────
print("1. Procesando inventario de frentes de manzana...")
df_frentes = pd.read_csv('inv_frentes.csv', dtype=str)

df_gdl = df_frentes[df_frentes['CVE_MUN'] == '039'].copy()
df_gdl['CVEGEO_AGEB'] = df_gdl['CVEGEO'].str[:13]
df_gdl['sin_alumbrado'] = (df_gdl['ALUMPUB_D'] == 'No dispone').astype(int)

agrupado_alumbrado = df_gdl.groupby('CVEGEO_AGEB').agg(
    frentes_sin_alumbrado=('sin_alumbrado', 'sum'),
    total_frentes=('CVEGEO_AGEB', 'count')
).reset_index()

agrupado_alumbrado['% Sin Alumbrado'] = (
    agrupado_alumbrado['frentes_sin_alumbrado'] / agrupado_alumbrado['total_frentes']
) * 100
agrupado_alumbrado = agrupado_alumbrado.rename(columns={'CVEGEO_AGEB': 'CVEGEO'})

# ─────────────────────────────────────────────────────────────────────────────
# 2. CARGAR MAPA Y DELITOS
# ─────────────────────────────────────────────────────────────────────────────
print("2. Cargando mapa AGEB y procesando archivo de delitos...")
agebs = gpd.read_file('guadalajara_AGEB/2025_14039_A07052026_1549.shp')
delitos = pd.read_csv('iieg_2023.csv').dropna(subset=['x', 'y'])

gdf_delitos = gpd.GeoDataFrame(
    delitos, geometry=gpd.points_from_xy(delitos.x, delitos.y), crs="EPSG:4326"
)
agebs = agebs.to_crs(gdf_delitos.crs)
join_delitos = gpd.sjoin(gdf_delitos, agebs[['CVEGEO', 'geometry']], predicate='within')

# ─────────────────────────────────────────────────────────────────────────────
# 3. ESTRUCTURAR DATOS
# ─────────────────────────────────────────────────────────────────────────────
print("3. Estructurando matriz de datos...")
delitos_pivot = join_delitos.pivot_table(index='CVEGEO', columns='delito', aggfunc='size', fill_value=0)
delitos_pivot['Total Delitos'] = delitos_pivot.sum(axis=1)

df_final = pd.merge(
    agrupado_alumbrado[['CVEGEO', '% Sin Alumbrado']],
    delitos_pivot, on='CVEGEO', how='inner'
)
df_final.set_index('CVEGEO', inplace=True)
df_final.columns = [
    str(c).title() if c not in ['% Sin Alumbrado', 'Total Delitos'] else c
    for c in df_final.columns
]

# Variables de interés
delitos_criticos = ['Homicidio Doloso', 'Feminicidio', 'Violacion', 'Abuso Sexual Infantil']
criticos_existentes = [d for d in delitos_criticos if d in df_final.columns]
top12 = delitos_pivot.drop(columns=['Total Delitos']).sum().nlargest(12).index.str.title().tolist()
top6 = delitos_pivot.drop(columns=['Total Delitos']).sum().nlargest(6).index.str.title().tolist()

cols_m1 = ['% Sin Alumbrado', 'Total Delitos'] + criticos_existentes
cols_m2 = ['% Sin Alumbrado'] + top12

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN AUXILIAR: calcular tabla de p-valores
# ─────────────────────────────────────────────────────────────────────────────
def compute_pvalue_matrix(df, cols):
    """Devuelve un DataFrame triangular inferior con los p-valores de Pearson."""
    n = len(cols)
    pval = np.ones((n, n))
    for i in range(n):
        for j in range(i):
            _, p = stats.pearsonr(df[cols[i]].fillna(0), df[cols[j]].fillna(0))
            pval[i, j] = p
            pval[j, i] = p
    return pd.DataFrame(pval, index=cols, columns=cols)


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICA 1 — MATRICES DE CORRELACIÓN TRIANGULAR (PEARSON)
# ══════════════════════════════════════════════════════════════════════════════
print("4. Generando Matriz de Correlación Triangular (Pearson)...")

corr_1 = df_final[cols_m1].corr()
corr_2 = df_final[cols_m2].corr()

sns.set_theme(style="white", font_scale=1.1)
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 11))
cmap_div = sns.color_palette("vlag", as_cmap=True)

mask1 = np.triu(np.ones_like(corr_1, dtype=bool))
sns.heatmap(corr_1, mask=mask1, cmap=cmap_div, vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, annot=True, fmt=".2f",
            cbar_kws={"shrink": .7}, ax=ax1, annot_kws={"size": 13})
ax1.set_title("Pearson: Alumbrado vs Delitos Dolosos", fontsize=18, fontweight='bold', pad=20)
ax1.tick_params(axis='x', rotation=45, labelsize=12)
ax1.tick_params(axis='y', rotation=0, labelsize=12)

mask2 = np.triu(np.ones_like(corr_2, dtype=bool))
sns.heatmap(corr_2, mask=mask2, cmap=cmap_div, vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, annot=True, fmt=".2f",
            cbar_kws={"shrink": .7}, ax=ax2, annot_kws={"size": 10})
ax2.set_title("Pearson: Alumbrado vs Top 12 Delitos Generales", fontsize=18, fontweight='bold', pad=20)
ax2.tick_params(axis='x', rotation=45, labelsize=10)
ax2.tick_params(axis='y', rotation=0, labelsize=10)

plt.tight_layout()
plt.savefig('cor_01_matriz_triangular.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✔ cor_01_matriz_triangular.png")


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICA 2 — CUADRÍCULA DE DISPERSIÓN (NUBE DE PUNTOS)
# ══════════════════════════════════════════════════════════════════════════════
print("5. Generando cuadrícula de dispersión...")

vars_scatter = ['% Sin Alumbrado', 'Total Delitos'] + top6
n_vars = len(vars_scatter)               # 8 variables → 7 gráficos vs % Sin Alumbrado

fig2, axes = plt.subplots(2, 4, figsize=(22, 11))
axes = axes.flatten()

COLOR_PUNTOS  = '#2c3e50'
COLOR_LINEA   = '#e74c3c'
COLOR_IC      = '#fadbd8'

for idx, delito in enumerate(vars_scatter[1:]):   # el eje X siempre es % Sin Alumbrado
    ax = axes[idx]
    x_data = df_final['% Sin Alumbrado'].fillna(0)
    y_data = df_final[delito].fillna(0)

    r_p, p_p = stats.pearsonr(x_data, y_data)
    r_s, p_s = stats.spearmanr(x_data, y_data)

    sns.regplot(x=x_data, y=y_data,
                scatter_kws={'alpha': 0.35, 's': 25, 'color': COLOR_PUNTOS},
                line_kws={'color': COLOR_LINEA, 'linewidth': 1.8},
                ci=95,
                ax=ax)

    # Etiqueta de significancia
    sig = "✗ No sig." if p_p > 0.05 else "✓ Sig."
    ax.set_title(
        f"{delito}\n"
        f"Pearson r={r_p:.3f}  |  Spearman ρ={r_s:.3f}\n"
        f"p-valor={p_p:.3f}  →  {sig}",
        fontsize=9.5, fontweight='bold', pad=8
    )
    ax.set_xlabel("% Frentes sin Alumbrado", fontsize=8)
    ax.set_ylabel("N° Incidentes", fontsize=8)
    ax.tick_params(labelsize=8)

# Ocultar el último eje si sobra
for idx in range(len(vars_scatter) - 1, len(axes)):
    axes[idx].set_visible(False)

fig2.suptitle(
    "Dispersión: % Sin Alumbrado vs Delitos (cada punto = 1 AGEB)\n"
    "La nube de puntos sin tendencia evidencia la ausencia de correlación",
    fontsize=14, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.savefig('cor_02_dispersion_grid.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✔ cor_02_dispersion_grid.png")


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICA 3 — HEATMAP DE P-VALORES (SIGNIFICANCIA ESTADÍSTICA)
# ══════════════════════════════════════════════════════════════════════════════
print("6. Generando heatmap de p-valores...")

cols_pval = ['% Sin Alumbrado', 'Total Delitos'] + criticos_existentes + top6[:4]
cols_pval = list(dict.fromkeys(cols_pval))   # eliminar duplicados preservando orden

pval_df  = compute_pvalue_matrix(df_final, cols_pval)
mask_pval = np.triu(np.ones_like(pval_df, dtype=bool))

# Creamos anotaciones especiales: el valor numérico + asterisco si es significativo
annot_pval = pval_df.copy().round(3).astype(str)
for i in range(len(cols_pval)):
    for j in range(i):
        p = pval_df.iloc[i, j]
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        annot_pval.iloc[i, j] = f"{p:.3f}\n{stars}"

fig3, ax3 = plt.subplots(figsize=(13, 10))

# Colormap: verde oscuro = p<0.05 (significativo), rojo claro = p>0.05 (no significativo)
cmap_pval = sns.diverging_palette(10, 145, s=80, l=55, n=256, as_cmap=True)

sns.heatmap(pval_df, mask=mask_pval, cmap=cmap_pval, vmax=1, vmin=0, center=0.05,
            square=True, linewidths=.5, annot=annot_pval, fmt="",
            cbar_kws={"shrink": .6, "label": "p-valor"}, ax=ax3, annot_kws={"size": 9})

ax3.set_title(
    "P-Valores de Correlación de Pearson\n"
    "Verde oscuro = significativo (p<0.05)  |  Rojo = no significativo (p>0.05)\n"
    "ns = no significativo  |  * p<0.05  |  ** p<0.01  |  *** p<0.001",
    fontsize=13, fontweight='bold', pad=20
)
ax3.tick_params(axis='x', rotation=45, labelsize=10)
ax3.tick_params(axis='y', rotation=0, labelsize=10)

plt.tight_layout()
plt.savefig('cor_03_pvalores.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✔ cor_03_pvalores.png")


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICA 4 — COMPARATIVA PEARSON vs SPEARMAN (BARRAS HORIZONTALES)
# ══════════════════════════════════════════════════════════════════════════════
print("7. Generando comparativa Pearson vs Spearman...")

vars_bar = ['Total Delitos'] + criticos_existentes + top6
vars_bar = list(dict.fromkeys(vars_bar))

resultados = []
for delito in vars_bar:
    x = df_final['% Sin Alumbrado'].fillna(0)
    y = df_final[delito].fillna(0)
    r_p, p_p = stats.pearsonr(x, y)
    r_s, p_s = stats.spearmanr(x, y)
    resultados.append({
        'Delito': delito,
        'Pearson r': r_p,
        'Spearman ρ': r_s,
        'p_pearson': p_p,
        'p_spearman': p_s,
    })

df_bar = pd.DataFrame(resultados).set_index('Delito')
df_bar = df_bar.reindex(df_bar['Pearson r'].abs().sort_values(ascending=True).index)

fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(18, max(6, len(vars_bar) * 0.65 + 2)),
                                   sharey=True)

# Paleta: azul si positivo, rojo si negativo
def bar_colors(series, alpha=0.8):
    return ['#2980b9' if v >= 0 else '#c0392b' for v in series]

# — Pearson —
bars_p = ax4a.barh(df_bar.index, df_bar['Pearson r'],
                   color=bar_colors(df_bar['Pearson r']), edgecolor='white', height=0.6)
ax4a.axvline(0, color='black', linewidth=1)
ax4a.axvspan(-0.1, 0.1, alpha=0.08, color='gray')   # zona "sin efecto"
ax4a.set_xlim(-0.7, 0.7)
ax4a.set_xlabel("Coeficiente de Correlación de Pearson (r)", fontsize=11)
ax4a.set_title("Pearson r\n(Correlación Lineal)", fontsize=13, fontweight='bold')
for bar, (_, row) in zip(bars_p, df_bar.iterrows()):
    sig = "*" if row['p_pearson'] < 0.05 else ""
    ax4a.text(bar.get_width() + (0.015 if bar.get_width() >= 0 else -0.015),
              bar.get_y() + bar.get_height() / 2,
              f"{bar.get_width():.3f}{sig}", va='center', ha='left' if bar.get_width() >= 0 else 'right',
              fontsize=8.5)

# — Spearman —
bars_s = ax4b.barh(df_bar.index, df_bar['Spearman ρ'],
                   color=bar_colors(df_bar['Spearman ρ']), edgecolor='white', height=0.6)
ax4b.axvline(0, color='black', linewidth=1)
ax4b.axvspan(-0.1, 0.1, alpha=0.08, color='gray')
ax4b.set_xlim(-0.7, 0.7)
ax4b.set_xlabel("Coeficiente de Correlación de Spearman (ρ)", fontsize=11)
ax4b.set_title("Spearman ρ\n(Correlación Monotónica — robusta a outliers)", fontsize=13, fontweight='bold')
for bar, (_, row) in zip(bars_s, df_bar.iterrows()):
    sig = "*" if row['p_spearman'] < 0.05 else ""
    ax4b.text(bar.get_width() + (0.015 if bar.get_width() >= 0 else -0.015),
              bar.get_y() + bar.get_height() / 2,
              f"{bar.get_width():.3f}{sig}", va='center', ha='left' if bar.get_width() >= 0 else 'right',
              fontsize=8.5)

fig4.suptitle(
    "Comparativa Pearson vs Spearman: % Sin Alumbrado vs Delitos por AGEB\n"
    "Zona gris = efecto despreciable (|r| < 0.10)  |  * = p<0.05 (estadísticamente significativo)",
    fontsize=13, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.savefig('cor_04_pearson_vs_spearman.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✔ cor_04_pearson_vs_spearman.png")


# ─────────────────────────────────────────────────────────────────────────────
print("\n✅ ANÁLISIS COMPLETADO. Se generaron 4 archivos:")
print("   cor_01_matriz_triangular.png    → Mapa de calor triangular (Pearson)")
print("   cor_02_dispersion_grid.png      → Nube de puntos por delito")
print("   cor_03_pvalores.png             → Heatmap de significancia estadística")
print("   cor_04_pearson_vs_spearman.png  → Barras comparativas (lineal vs monotónico)")