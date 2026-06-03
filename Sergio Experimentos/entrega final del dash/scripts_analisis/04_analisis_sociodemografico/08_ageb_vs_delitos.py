"""
SCRIPT 8 — Correlación: Tipos de Negocios (DENUE) vs Tipos de Delitos (IIEG 2023)
===================================================================================
Analiza si la densidad de ciertos tipos de establecimientos comerciales por AGEB
está asociada con la incidencia de delitos específicos, usando 4 evidencias visuales:

  1. Heatmap de correlación Pearson (negocios x delitos)
  2. Heatmap de p-valores (qué correlaciones son estadísticamente significativas)
  3. Cuadrícula de scatter plots de los pares con mayor correlación
  4. Ranking de correlaciones más fuertes (positivas y negativas)
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import scipy.stats as stats
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGAR DATOS
# ─────────────────────────────────────────────────────────────────────────────
print("1. Cargando bases de datos (IIEG 2023 + DENUE + AGEB)...")
agebs    = gpd.read_file('guadalajara_AGEB/2025_14039_A07052026_1549.shp')
delitos  = pd.read_csv('iieg_2023.csv').dropna(subset=['x', 'y'])
denue    = pd.read_csv('gdl_denue.csv').dropna(subset=['latitud', 'longitud'])

# ─────────────────────────────────────────────────────────────────────────────
# 2. FILTRAR CATEGORÍAS RELEVANTES (excluir "Otro" — demasiado genérico)
# ─────────────────────────────────────────────────────────────────────────────
print("2. Filtrando categorías de negocios con mayor poder explicativo...")

CATEGORIAS_INTERES = [
    'Bar / Cantina',
    'Restaurante',
    'Antro / Discoteca',
    'Licorería',
    'Hotel / Motel',
    'Joyería / Relojería',
    'Banco / Financiero',
    'Casa de empeño',
    'Escuela',
    'Farmacia',
    'Hospital / Clínica',
    'Tienda conveniencia',
    'Supermercado',
    'Gasolinera',
    'Estacionamiento',
    'Policía / Seguridad',
    'Telefonía / Electrónica',
]

denue_filtrado = denue[denue['categoria'].isin(CATEGORIAS_INTERES)].copy()
print(f"   → {len(denue_filtrado):,} establecimientos en {len(CATEGORIAS_INTERES)} categorías")

# ─────────────────────────────────────────────────────────────────────────────
# 3. PROCESAMIENTO ESPACIAL
# ─────────────────────────────────────────────────────────────────────────────
print("3. Ejecutando cruce espacial (spatial join) por AGEB...")

gdf_delitos  = gpd.GeoDataFrame(
    delitos,
    geometry=gpd.points_from_xy(delitos.x, delitos.y),
    crs="EPSG:4326"
)
gdf_negocios = gpd.GeoDataFrame(
    denue_filtrado,
    geometry=gpd.points_from_xy(denue_filtrado.longitud, denue_filtrado.latitud),
    crs="EPSG:4326"
)
agebs = agebs.to_crs(gdf_delitos.crs)

# --- Delitos: pivot por tipo de delito por AGEB ---
join_delitos = gpd.sjoin(gdf_delitos, agebs[['CVEGEO', 'geometry']], predicate='within')

# Tomamos los top 10 delitos más frecuentes + Total
top_delitos_raw = join_delitos['delito'].value_counts().nlargest(10).index.tolist()
counts_delitos  = join_delitos.pivot_table(
    index='CVEGEO', columns='delito', aggfunc='size', fill_value=0
)
# Capitalizar
counts_delitos.columns = counts_delitos.columns.str.title()
top_delitos = [d.title() for d in top_delitos_raw]

counts_delitos['Total Delitos'] = counts_delitos.sum(axis=1)

# --- Negocios: pivot por categoría por AGEB ---
join_negocios   = gpd.sjoin(gdf_negocios, agebs[['CVEGEO', 'geometry']], predicate='within')
counts_negocios = join_negocios.pivot_table(
    index='CVEGEO', columns='categoria', aggfunc='size', fill_value=0
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. UNIFICACIÓN Y CÁLCULO DE CORRELACIONES
# ─────────────────────────────────────────────────────────────────────────────
print("4. Calculando correlaciones Pearson y Spearman...")

df_final = pd.merge(
    counts_delitos[['Total Delitos'] + top_delitos],
    counts_negocios,
    left_index=True, right_index=True, how='outer'
).fillna(0)

# Columnas finales
cols_negocios = [c for c in counts_negocios.columns if c in df_final.columns]
cols_delitos  = ['Total Delitos'] + [d for d in top_delitos if d in df_final.columns]

# Matriz de correlación cruzada (negocios como filas, delitos como columnas)
def cross_corr_matrix(df, rows, cols, method='pearson'):
    """Calcula una matriz de correlación cruzada entre dos grupos de variables."""
    result = pd.DataFrame(index=rows, columns=cols, dtype=float)
    for r in rows:
        for c in cols:
            if method == 'pearson':
                val, _ = stats.pearsonr(df[r].fillna(0), df[c].fillna(0))
            else:
                val, _ = stats.spearmanr(df[r].fillna(0), df[c].fillna(0))
            result.loc[r, c] = val
    return result

def cross_pval_matrix(df, rows, cols):
    """Calcula la matriz de p-valores (Pearson) entre dos grupos de variables."""
    result = pd.DataFrame(index=rows, columns=cols, dtype=float)
    for r in rows:
        for c in cols:
            _, p = stats.pearsonr(df[r].fillna(0), df[c].fillna(0))
            result.loc[r, c] = p
    return result

corr_pearson  = cross_corr_matrix(df_final, cols_negocios, cols_delitos, 'pearson')
corr_spearman = cross_corr_matrix(df_final, cols_negocios, cols_delitos, 'spearman')
pval_matrix   = cross_pval_matrix(df_final, cols_negocios, cols_delitos)

print(f"   → Matriz {len(cols_negocios)} categorías × {len(cols_delitos)} tipos de delito")


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICA 1 — HEATMAP DE CORRELACIÓN (PEARSON)
# ══════════════════════════════════════════════════════════════════════════════
print("5. Generando Heatmap de Correlación Pearson...")

# Anotaciones que muestran el valor + asterisco de significancia
def make_annot(corr_df, pval_df):
    annot = corr_df.copy().astype(object)
    for r in corr_df.index:
        for c in corr_df.columns:
            v = corr_df.loc[r, c]
            p = pval_df.loc[r, c]
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            annot.loc[r, c] = f"{v:.2f}{stars}"
    return annot

annot_pearson = make_annot(corr_pearson, pval_matrix)

sns.set_theme(style="white", font_scale=1.0)
cmap_div = sns.color_palette("vlag", as_cmap=True)

fig1, ax = plt.subplots(figsize=(len(cols_delitos) * 1.5 + 2, len(cols_negocios) * 0.65 + 3))

sns.heatmap(
    corr_pearson.astype(float),
    annot=annot_pearson, fmt="", cmap=cmap_div,
    vmin=-1, vmax=1, center=0,
    linewidths=0.4, linecolor='#e0e0e0',
    cbar_kws={"shrink": 0.7, "label": "Pearson r"},
    ax=ax, annot_kws={"size": 9}
)

ax.set_title(
    "Correlación Pearson: Densidad de Establecimientos vs Incidencia Delictiva por AGEB\n"
    "* p<0.05  |  ** p<0.01  |  *** p<0.001",
    fontsize=14, fontweight='bold', pad=18
)
ax.set_xlabel("Tipo de Delito", fontsize=11, labelpad=10)
ax.set_ylabel("Categoría de Establecimiento (DENUE)", fontsize=11, labelpad=10)
ax.tick_params(axis='x', rotation=40, labelsize=9)
ax.tick_params(axis='y', rotation=0, labelsize=9)

plt.tight_layout()
plt.savefig('neg_01_heatmap_pearson.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✔ neg_01_heatmap_pearson.png")


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICA 2 — HEATMAP DE P-VALORES
# ══════════════════════════════════════════════════════════════════════════════
print("6. Generando Heatmap de P-Valores...")

# Anotación: valor numérico + etiqueta de significancia
annot_pval = pval_matrix.copy().astype(object)
for r in pval_matrix.index:
    for c in pval_matrix.columns:
        p = pval_matrix.loc[r, c]
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "* " if p < 0.05 else "ns"
        annot_pval.loc[r, c] = f"{p:.3f}\n{stars}"

# Verde oscuro = significativo (p bajo), rojo = no significativo (p alto)
cmap_pval = sns.diverging_palette(10, 145, s=80, l=55, n=256, as_cmap=True)

fig2, ax = plt.subplots(figsize=(len(cols_delitos) * 1.5 + 2, len(cols_negocios) * 0.65 + 3))

sns.heatmap(
    pval_matrix.astype(float),
    annot=annot_pval, fmt="", cmap=cmap_pval,
    vmin=0, vmax=0.1, center=0.05,
    linewidths=0.4, linecolor='#e0e0e0',
    cbar_kws={"shrink": 0.7, "label": "p-valor"},
    ax=ax, annot_kws={"size": 8}
)

ax.set_title(
    "P-Valores de Correlación: Establecimientos vs Delitos por AGEB\n"
    "Verde oscuro = estadísticamente significativo (p<0.05)  |  ns = no significativo",
    fontsize=14, fontweight='bold', pad=18
)
ax.set_xlabel("Tipo de Delito", fontsize=11, labelpad=10)
ax.set_ylabel("Categoría de Establecimiento (DENUE)", fontsize=11, labelpad=10)
ax.tick_params(axis='x', rotation=40, labelsize=9)
ax.tick_params(axis='y', rotation=0, labelsize=9)

plt.tight_layout()
plt.savefig('neg_02_heatmap_pvalores.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✔ neg_02_heatmap_pvalores.png")


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICA 3 — SCATTER PLOTS DE LOS PARES CON MAYOR CORRELACIÓN
# ══════════════════════════════════════════════════════════════════════════════
print("7. Generando cuadrícula de scatter plots (pares más fuertes)...")

# Encontrar los 8 pares de (negocio, delito) con mayor |r| de Pearson
pairs = []
for neg in cols_negocios:
    for dlt in cols_delitos:
        r = corr_pearson.loc[neg, dlt]
        p = pval_matrix.loc[neg, dlt]
        pairs.append({'negocio': neg, 'delito': dlt, 'r': r, 'p': p, 'abs_r': abs(r)})

df_pairs = pd.DataFrame(pairs).sort_values('abs_r', ascending=False)
top_pairs = df_pairs.head(8)

sns.set_theme(style="whitegrid", font_scale=0.9)
fig3, axes = plt.subplots(2, 4, figsize=(22, 11))
axes = axes.flatten()

COLOR_POS = '#2980b9'
COLOR_NEG = '#c0392b'
COLOR_LINE = '#e74c3c'

for idx, (_, row) in enumerate(top_pairs.iterrows()):
    ax = axes[idx]
    neg, dlt = row['negocio'], row['delito']
    r_p, p_p = row['r'], row['p']
    r_s, p_s = stats.spearmanr(df_final[neg].fillna(0), df_final[dlt].fillna(0))

    color_puntos = COLOR_POS if r_p >= 0 else COLOR_NEG
    sig_label = "✓ Significativo" if p_p < 0.05 else "✗ No significativo"

    sns.regplot(
        x=df_final[neg].fillna(0),
        y=df_final[dlt].fillna(0),
        scatter_kws={'alpha': 0.35, 's': 22, 'color': color_puntos},
        line_kws={'color': COLOR_LINE, 'linewidth': 1.8},
        ci=95, ax=ax
    )

    ax.set_title(
        f"{neg}\nvs  {dlt}\n"
        f"Pearson r={r_p:.3f}  Spearman ρ={r_s:.3f}\n"
        f"p={p_p:.4f}  →  {sig_label}",
        fontsize=8.5, fontweight='bold', pad=7, linespacing=1.4
    )
    ax.set_xlabel(f"N° {neg}", fontsize=8)
    ax.set_ylabel(f"N° {dlt}", fontsize=8)
    ax.tick_params(labelsize=7)

# Ocultar ejes sobrantes
for idx in range(len(top_pairs), len(axes)):
    axes[idx].set_visible(False)

fig3.suptitle(
    "Top 8 Pares con Mayor Correlación: Densidad de Negocios vs Delitos por AGEB\n"
    "(Ordenados por valor absoluto de Pearson r — azul = correlación positiva, rojo = negativa)",
    fontsize=13, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.savefig('neg_03_scatter_top_pares.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✔ neg_03_scatter_top_pares.png")


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICA 4 — RANKING DE CORRELACIONES (BARRAS HORIZONTALES)
# ══════════════════════════════════════════════════════════════════════════════
print("8. Generando ranking de correlaciones vs Total Delitos...")

# Tomamos la columna "Total Delitos" y ordenamos todas las categorías
ranking = corr_pearson[['Total Delitos']].copy()
ranking.columns = ['Pearson r']
ranking['Spearman ρ'] = corr_spearman[['Total Delitos']].values
ranking['p-valor Pearson'] = pval_matrix[['Total Delitos']].values
ranking = ranking.sort_values('Pearson r', ascending=True)

fig4, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(18, max(6, len(cols_negocios) * 0.7 + 2)),
                                    sharey=True)

def bar_colors_alpha(series, pvals, alpha_ns=0.35):
    colors = []
    for v, p in zip(series, pvals):
        base = '#2980b9' if v >= 0 else '#c0392b'
        colors.append(base if p < 0.05 else base + '60')
    return colors

colors_p = bar_colors_alpha(ranking['Pearson r'], ranking['p-valor Pearson'])
colors_s = bar_colors_alpha(ranking['Spearman ρ'], ranking['p-valor Pearson'])

# — Pearson —
bars_p = ax_l.barh(ranking.index, ranking['Pearson r'],
                   color=colors_p, edgecolor='white', height=0.6)
ax_l.axvline(0, color='black', linewidth=0.9)
ax_l.axvspan(-0.1, 0.1, alpha=0.07, color='gray', label='Efecto despreciable')
ax_l.set_xlim(-0.9, 0.9)
ax_l.set_xlabel("Pearson r", fontsize=11)
ax_l.set_title("Correlación Pearson\nvs Total Delitos", fontsize=13, fontweight='bold')
for bar, p in zip(bars_p, ranking['p-valor Pearson']):
    sig = "*" if p < 0.05 else ""
    offset = 0.02 if bar.get_width() >= 0 else -0.02
    ha = 'left' if bar.get_width() >= 0 else 'right'
    ax_l.text(bar.get_width() + offset,
              bar.get_y() + bar.get_height() / 2,
              f"{bar.get_width():.3f}{sig}", va='center', ha=ha, fontsize=8.5)

# — Spearman —
bars_s = ax_r.barh(ranking.index, ranking['Spearman ρ'],
                   color=colors_s, edgecolor='white', height=0.6)
ax_r.axvline(0, color='black', linewidth=0.9)
ax_r.axvspan(-0.1, 0.1, alpha=0.07, color='gray')
ax_r.set_xlim(-0.9, 0.9)
ax_r.set_xlabel("Spearman ρ", fontsize=11)
ax_r.set_title("Correlación Spearman\nvs Total Delitos (robusto a outliers)", fontsize=13, fontweight='bold')
for bar in bars_s:
    offset = 0.02 if bar.get_width() >= 0 else -0.02
    ha = 'left' if bar.get_width() >= 0 else 'right'
    ax_r.text(bar.get_width() + offset,
              bar.get_y() + bar.get_height() / 2,
              f"{bar.get_width():.3f}", va='center', ha=ha, fontsize=8.5)

# Leyenda
patch_sig  = mpatches.Patch(color='#2980b9', label='Positivo significativo (p<0.05)')
patch_neg  = mpatches.Patch(color='#c0392b', label='Negativo significativo (p<0.05)')
patch_ns   = mpatches.Patch(color='#2980b960', label='No significativo (p≥0.05)')
patch_zone = mpatches.Patch(color='gray', alpha=0.2, label='Zona sin efecto (|r|<0.10)')

fig4.legend(handles=[patch_sig, patch_neg, patch_ns, patch_zone],
            loc='lower center', ncol=4, fontsize=9,
            bbox_to_anchor=(0.5, -0.06), frameon=True)

fig4.suptitle(
    "Ranking de Correlación: Tipos de Establecimiento vs Total de Delitos por AGEB\n"
    "Categorías de negocios ordenadas por fuerza de correlación  |  * = p<0.05",
    fontsize=13, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.savefig('neg_04_ranking_correlaciones.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✔ neg_04_ranking_correlaciones.png")


# ─────────────────────────────────────────────────────────────────────────────
# RESUMEN NUMÉRICO EN CONSOLA
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 65)
print("  RESUMEN: Correlaciones vs Total Delitos (Pearson, ordenado)")
print("═" * 65)
resumen = ranking.copy()
resumen['Significativo'] = resumen['p-valor Pearson'].apply(lambda p: '✓' if p < 0.05 else '✗')
print(resumen[['Pearson r', 'Spearman ρ', 'p-valor Pearson', 'Significativo']].to_string())
print("═" * 65)

print("\n✅ ANÁLISIS COMPLETADO. Se generaron 4 archivos:")
print("   neg_01_heatmap_pearson.png      → Correlaciones por tipo de negocio y delito")
print("   neg_02_heatmap_pvalores.png     → Significancia estadística de cada celda")
print("   neg_03_scatter_top_pares.png    → Dispersión de los 8 pares más correlacionados")
print("   neg_04_ranking_correlaciones.png → Ranking de categorías vs Total de Delitos")