"""
SCRIPT 9 — Correlación: Nivel Educativo vs Incidencia Delictiva (ENVIPE GDL)
=============================================================================
Analiza la relación entre el nivel de escolaridad promedio por UPM
(Unidad Primaria de Muestreo) y los distintos tipos de delitos reportados,
usando 4 evidencias visuales:

  1. Heatmap de correlación Spearman (distribución educativa × tipos de delito)
  2. Heatmap de p-valores (qué correlaciones son estadísticamente significativas)
  3. Cuadrícula de scatter plots de los pares con mayor correlación
  4. Ranking de correlaciones más fuertes (positivas y negativas)

Fuentes:
  - gdl_delitos.csv  → ENVIPE 2023 — módulo de delitos (con factor de expansión FAC_DEL)
  - gdl_sdem.csv     → ENVIPE 2023 — módulo sociodemográfico (nivel educativo por persona)
"""

import pandas as pd
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
print("1. Cargando bases de datos (ENVIPE: delitos + sociodemográfico)...")
df_delitos = pd.read_csv('gdl_delitos.csv')
df_sdem    = pd.read_csv('gdl_sdem.csv')

print(f"   → Delitos: {len(df_delitos):,} registros  |  Sociodemográfico: {len(df_sdem):,} registros")

# ─────────────────────────────────────────────────────────────────────────────
# 2. CONSTRUIR VARIABLES EDUCATIVAS POR UPM
# ─────────────────────────────────────────────────────────────────────────────
print("2. Construyendo indicadores educativos por UPM...")

# — Proxy numérico: años de escolaridad equivalentes por nivel —
edu_mapping = {
    'Sin escolaridad': 0,
    'Preescolar':      3,
    'Primaria':        6,
    'Secundaria':      9,
    'Carrera técnica con secundaria': 9,
    'Normal básica': 12,
    'Preparatoria o bachillerato': 12,
    'Carrera técnica con preparatoria': 12,
    'Licenciatura':   16,
    'Posgrado (Maestría/Doctorado)': 18,
}

df_sdem = df_sdem[~df_sdem['NIVEL_EDU'].isin(['NE', 'No especificado'])].copy()   # excluir "No especificado"
df_sdem['escolaridad_anios'] = df_sdem['NIVEL_EDU'].map(edu_mapping)

# — Indicadores agregados por UPM —
upm_edu = df_sdem.groupby('UPM').agg(
    edu_media       = ('escolaridad_anios', 'mean'),     # promedio de años
    pct_sin_edu     = ('escolaridad_anios',              # % sin escolaridad
                       lambda x: (x == 0).mean() * 100),
    pct_superior    = ('escolaridad_anios',              # % con educación superior
                       lambda x: (x >= 16).mean() * 100),
    n_personas      = ('escolaridad_anios', 'count'),
).reset_index()

print(f"   → {len(upm_edu)} UPMs con datos educativos")

# ─────────────────────────────────────────────────────────────────────────────
# 3. CONSTRUIR VARIABLES DELICTIVAS POR UPM
# ─────────────────────────────────────────────────────────────────────────────
print("3. Construyendo indicadores delictivos por UPM...")

# Volumen total de delitos (expandido con FAC_DEL)
upm_total = (df_delitos.groupby('UPM')['FAC_DEL']
             .sum()
             .rename('Volumen_Total_Delitos')
             .reset_index())

# Volumen por tipo de delito (expandido)
upm_tipo = (df_delitos.groupby(['UPM', 'TIPO_DELITO_STR'])['FAC_DEL']
            .sum()
            .unstack(fill_value=0)
            .reset_index())

# Unir total + por tipo
upm_crime = pd.merge(upm_total, upm_tipo, on='UPM', how='outer').fillna(0)
print(f"   → {len(upm_crime)} UPMs con datos delictivos")

# ─────────────────────────────────────────────────────────────────────────────
# 4. CRUCE DE TABLAS
# ─────────────────────────────────────────────────────────────────────────────
print("4. Cruzando tablas por UPM...")
df_corr = pd.merge(upm_edu, upm_crime, on='UPM', how='inner')
print(f"   → {len(df_corr)} UPMs en común para el análisis")

# Variables educativas (X) y delictivas (Y)
cols_edu    = ['edu_media', 'pct_sin_edu', 'pct_superior']
labels_edu  = {
    'edu_media':      'Años de escolaridad\n(media UPM)',
    'pct_sin_edu':    '% Sin escolaridad\nen la UPM',
    'pct_superior':   '% Con educación\nsuperior en UPM',
}

tipos_delito = [c for c in upm_crime.columns if c not in ['UPM', 'Volumen_Total_Delitos']]
cols_delito  = ['Volumen_Total_Delitos'] + tipos_delito

# ─────────────────────────────────────────────────────────────────────────────
# 5. CALCULAR MATRICES DE CORRELACIÓN (SPEARMAN — no paramétrica)
# ─────────────────────────────────────────────────────────────────────────────
print("5. Calculando correlaciones Spearman y p-valores...")

def cross_spearman(df, rows, cols):
    """Matriz cruzada de Spearman entre dos grupos de variables."""
    r_mat = pd.DataFrame(index=rows, columns=cols, dtype=float)
    p_mat = pd.DataFrame(index=rows, columns=cols, dtype=float)
    for r in rows:
        for c in cols:
            x = df[r].fillna(0)
            y = df[c].fillna(0)
            rho, p = stats.spearmanr(x, y)
            r_mat.loc[r, c] = rho
            p_mat.loc[r, c] = p
    return r_mat, p_mat

corr_spearman, pval_mat = cross_spearman(df_corr, cols_edu, cols_delito)

# Renombrar filas con etiquetas legibles
corr_spearman.index = [labels_edu[k] for k in cols_edu]
pval_mat.index      = [labels_edu[k] for k in cols_edu]

print(f"   → Matriz {len(cols_edu)} indicadores edu × {len(cols_delito)} tipos de delito")


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICA 1 — HEATMAP DE CORRELACIÓN SPEARMAN
# ══════════════════════════════════════════════════════════════════════════════
print("6. Generando Heatmap de Correlación Spearman...")

def make_annot(corr_df, pval_df):
    annot = corr_df.copy().astype(object)
    for r in corr_df.index:
        for c in corr_df.columns:
            v = float(corr_df.loc[r, c])
            p = float(pval_df.loc[r, c])
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            annot.loc[r, c] = f"{v:.2f}{stars}"
    return annot

annot_sp = make_annot(corr_spearman, pval_mat)

sns.set_theme(style="white", font_scale=1.05)
cmap_div = sns.color_palette("vlag", as_cmap=True)

n_cols = len(cols_delito)
fig1, ax = plt.subplots(figsize=(n_cols * 1.35 + 2, 5))

sns.heatmap(
    corr_spearman.astype(float),
    annot=annot_sp, fmt="", cmap=cmap_div,
    vmin=-1, vmax=1, center=0,
    linewidths=0.4, linecolor='#e0e0e0',
    cbar_kws={"shrink": 0.8, "label": "Spearman ρ"},
    ax=ax, annot_kws={"size": 9}
)

ax.set_title(
    "Correlación Spearman: Nivel Educativo vs Tipos de Delito por UPM (ENVIPE 2023)\n"
    "* p<0.05  |  ** p<0.01  |  *** p<0.001",
    fontsize=13, fontweight='bold', pad=16
)
ax.set_xlabel("Tipo de Delito", fontsize=11, labelpad=10)
ax.set_ylabel("Indicador Educativo", fontsize=11, labelpad=10)
ax.tick_params(axis='x', rotation=40, labelsize=8.5)
ax.tick_params(axis='y', rotation=0, labelsize=9)

plt.tight_layout()
plt.savefig('edu_01_heatmap_spearman.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✔ edu_01_heatmap_spearman.png")


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICA 2 — HEATMAP DE P-VALORES
# ══════════════════════════════════════════════════════════════════════════════
print("7. Generando Heatmap de P-Valores...")

annot_pval = pval_mat.copy().astype(object)
for r in pval_mat.index:
    for c in pval_mat.columns:
        p = float(pval_mat.loc[r, c])
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "* " if p < 0.05 else "ns"
        annot_pval.loc[r, c] = f"{p:.3f}\n{stars}"

cmap_pval = sns.diverging_palette(10, 145, s=80, l=55, n=256, as_cmap=True)

fig2, ax = plt.subplots(figsize=(n_cols * 1.35 + 2, 5))

sns.heatmap(
    pval_mat.astype(float),
    annot=annot_pval, fmt="", cmap=cmap_pval,
    vmin=0, vmax=0.1, center=0.05,
    linewidths=0.4, linecolor='#e0e0e0',
    cbar_kws={"shrink": 0.8, "label": "p-valor"},
    ax=ax, annot_kws={"size": 8.5}
)

ax.set_title(
    "P-Valores de Correlación: Nivel Educativo vs Delitos por UPM\n"
    "Verde oscuro = estadísticamente significativo (p<0.05)  |  ns = no significativo",
    fontsize=13, fontweight='bold', pad=16
)
ax.set_xlabel("Tipo de Delito", fontsize=11, labelpad=10)
ax.set_ylabel("Indicador Educativo", fontsize=11, labelpad=10)
ax.tick_params(axis='x', rotation=40, labelsize=8.5)
ax.tick_params(axis='y', rotation=0, labelsize=9)

plt.tight_layout()
plt.savefig('edu_02_heatmap_pvalores.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✔ edu_02_heatmap_pvalores.png")


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICA 3 — SCATTER PLOTS DE LOS 8 PARES CON MAYOR CORRELACIÓN
# ══════════════════════════════════════════════════════════════════════════════
print("8. Generando cuadrícula de scatter plots (pares más fuertes)...")

# Reindexar con etiquetas originales para recuperar las columnas del df
corr_raw, pval_raw = cross_spearman(df_corr, cols_edu, cols_delito)

pairs = []
for edu in cols_edu:
    for dlt in cols_delito:
        rho = float(corr_raw.loc[edu, dlt])
        p   = float(pval_raw.loc[edu, dlt])
        pairs.append({'edu': edu, 'delito': dlt, 'rho': rho, 'p': p, 'abs_rho': abs(rho)})

df_pairs = pd.DataFrame(pairs).sort_values('abs_rho', ascending=False)
top_pairs = df_pairs.head(8)

sns.set_theme(style="whitegrid", font_scale=0.88)
fig3, axes = plt.subplots(2, 4, figsize=(22, 11))
axes = axes.flatten()

COLOR_POS  = '#2980b9'
COLOR_NEG  = '#c0392b'
COLOR_LINE = '#e74c3c'

for idx, (_, row) in enumerate(top_pairs.iterrows()):
    ax    = axes[idx]
    edu   = row['edu']
    dlt   = row['delito']
    rho   = row['rho']
    p     = row['p']

    x_data = df_corr[edu].fillna(0)
    y_data = df_corr[dlt].fillna(0)

    # Añadir también Pearson para comparar
    r_p, p_p = stats.pearsonr(x_data, y_data)

    color   = COLOR_POS if rho >= 0 else COLOR_NEG
    sig_lbl = "✓ Significativo" if p < 0.05 else "✗ No significativo"

    sns.regplot(
        x=x_data, y=y_data,
        scatter_kws={'alpha': 0.45, 's': 30, 'color': color},
        line_kws={'color': COLOR_LINE, 'linewidth': 1.8},
        ci=95, ax=ax
    )

    ax.set_title(
        f"{labels_edu[edu]}\nvs  {dlt}\n"
        f"Spearman ρ={rho:.3f}  Pearson r={r_p:.3f}\n"
        f"p={p:.4f}  →  {sig_lbl}",
        fontsize=8.5, fontweight='bold', pad=7, linespacing=1.4
    )
    ax.set_xlabel(labels_edu[edu].replace('\n', ' '), fontsize=8)
    ax.set_ylabel(f"Volumen estimado\n{dlt}", fontsize=7.5)
    ax.tick_params(labelsize=7)

for idx in range(len(top_pairs), len(axes)):
    axes[idx].set_visible(False)

fig3.suptitle(
    "Top 8 Pares con Mayor Correlación: Indicadores Educativos vs Tipos de Delito (ENVIPE 2023)\n"
    "(Ordenados por valor absoluto de Spearman ρ — azul = positivo, rojo = negativo)",
    fontsize=13, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.savefig('edu_03_scatter_top_pares.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✔ edu_03_scatter_top_pares.png")


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICA 4 — RANKING DE CORRELACIONES vs VOLUMEN TOTAL (BARRAS)
# ══════════════════════════════════════════════════════════════════════════════
print("9. Generando ranking de correlaciones vs Volumen Total de Delitos...")

# Usamos los 3 indicadores educativos como filas, tipos de delito como columnas
# Tomamos la columna de Volumen_Total_Delitos y todos los tipos para el ranking
ranking_data = []
for edu in cols_edu:
    for dlt in cols_delito:
        rho = float(corr_raw.loc[edu, dlt])
        p   = float(pval_raw.loc[edu, dlt])
        ranking_data.append({
            'Indicador': labels_edu[edu].replace('\n', ' '),
            'Delito':    dlt,
            'Spearman ρ': rho,
            'p-valor':   p,
            'Sig':       p < 0.05,
        })

df_rank = pd.DataFrame(ranking_data)

# Filtrar solo "Volumen_Total_Delitos" para el ranking por indicador educativo
rank_total = (df_rank[df_rank['Delito'] == 'Volumen_Total_Delitos']
              .set_index('Indicador')
              .sort_values('Spearman ρ', ascending=True))

# Panel izquierdo: ranking vs total delitos
# Panel derecho: heatmap resumen de todos los pares (compacto)
fig4, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(18, 8))

# — Barras: indicadores educativos vs Total Delitos —
bar_colors = ['#2980b9' if v >= 0 else '#c0392b' for v in rank_total['Spearman ρ']]
bar_alpha   = [1.0 if sig else 0.4 for sig in rank_total['Sig']]

bars = ax_l.barh(rank_total.index, rank_total['Spearman ρ'],
                 color=[c + ('' if a == 1.0 else '60') for c, a in zip(bar_colors, bar_alpha)],
                 edgecolor='white', height=0.5)

ax_l.axvline(0, color='black', linewidth=0.9)
ax_l.axvspan(-0.15, 0.15, alpha=0.07, color='gray')
ax_l.set_xlim(-1.0, 1.0)
ax_l.set_xlabel("Spearman ρ", fontsize=11)
ax_l.set_title("Indicadores Educativos\nvs Volumen Total de Delitos", fontsize=12, fontweight='bold')

for bar, (_, row) in zip(bars, rank_total.iterrows()):
    sig = " *" if row['Sig'] else ""
    offset = 0.03 if bar.get_width() >= 0 else -0.03
    ha = 'left' if bar.get_width() >= 0 else 'right'
    ax_l.text(bar.get_width() + offset,
              bar.get_y() + bar.get_height() / 2,
              f"ρ={bar.get_width():.3f}  p={row['p-valor']:.3f}{sig}",
              va='center', ha=ha, fontsize=9)

# — Heatmap resumen: todos los indicadores × todos los delitos —
pivot = df_rank.pivot(index='Indicador', columns='Delito', values='Spearman ρ')
annot_r = df_rank.pivot(index='Indicador', columns='Delito', values='Spearman ρ').copy().astype(object)
piv_p   = df_rank.pivot(index='Indicador', columns='Delito', values='p-valor')

for r in pivot.index:
    for c in pivot.columns:
        v = pivot.loc[r, c]
        p = piv_p.loc[r, c]
        s = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        annot_r.loc[r, c] = f"{v:.2f}{s}"

sns.heatmap(
    pivot.astype(float),
    annot=annot_r, fmt="", cmap=cmap_div,
    vmin=-1, vmax=1, center=0,
    linewidths=0.3, linecolor='#e0e0e0',
    cbar_kws={"shrink": 0.6, "label": "Spearman ρ"},
    ax=ax_r, annot_kws={"size": 7.5}
)
ax_r.set_title("Mapa de Calor Completo\nTodos los Indicadores × Tipos de Delito", fontsize=12, fontweight='bold')
ax_r.set_xlabel("Tipo de Delito", fontsize=10, labelpad=8)
ax_r.set_ylabel("")
ax_r.tick_params(axis='x', rotation=40, labelsize=8)
ax_r.tick_params(axis='y', rotation=0, labelsize=8.5)

# Leyenda general
patch_pos  = mpatches.Patch(color='#2980b9', label='Correlación positiva (p<0.05)')
patch_neg  = mpatches.Patch(color='#c0392b', label='Correlación negativa (p<0.05)')
patch_ns   = mpatches.Patch(color='#2980b960', label='No significativo (p≥0.05)')
patch_zone = mpatches.Patch(color='gray', alpha=0.2, label='Zona sin efecto (|ρ|<0.15)')

fig4.legend(handles=[patch_pos, patch_neg, patch_ns, patch_zone],
            loc='lower center', ncol=4, fontsize=9,
            bbox_to_anchor=(0.5, -0.06), frameon=True)

fig4.suptitle(
    "Ranking y Resumen: Nivel Educativo vs Incidencia Delictiva por UPM (ENVIPE 2023)\n"
    "* p<0.05  |  ** p<0.01  |  *** p<0.001  |  Zona gris = efecto despreciable",
    fontsize=13, fontweight='bold', y=1.02
)
plt.tight_layout()
plt.savefig('edu_04_ranking_heatmap.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✔ edu_04_ranking_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# RESUMEN EN CONSOLA
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("  RESUMEN: Spearman ρ vs Volumen Total de Delitos")
print("═" * 70)
for _, row in rank_total.sort_values('Spearman ρ').iterrows():
    sig = "✓ Sig." if row['Sig'] else "✗ No sig."
    print(f"  {row.name:<40} ρ={row['Spearman ρ']:+.3f}  p={row['p-valor']:.4f}  {sig}")
print("═" * 70)

print("\n✅ ANÁLISIS COMPLETADO. Se generaron 4 archivos:")
print("   edu_01_heatmap_spearman.png  → Correlaciones por indicador educativo y delito")
print("   edu_02_heatmap_pvalores.png  → Significancia estadística de cada celda")
print("   edu_03_scatter_top_pares.png → Dispersión de los 8 pares más correlacionados")
print("   edu_04_ranking_heatmap.png   → Ranking vs total + mapa completo")