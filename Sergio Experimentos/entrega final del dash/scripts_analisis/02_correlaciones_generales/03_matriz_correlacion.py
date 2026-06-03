"""
SCRIPT 3 — Matriz de Correlación: Lugares × Tipos de Delito
=============================================================
Fuentes:
  - TMod_Vic.csv     → modalidad por delito (BPCOD + AREAM_OCU) NACIONAL
  - TPer_Vic1.csv    → delitos sufridos por persona (AP4_2_xx) Guadalajara
  - gdl_denue.csv    → establecimientos clasificados Guadalajara

Salidas:
  - matriz_lugar_delito.png      → heatmap lugar × tipo (% por fila)
  - matriz_pearson_tipos.png     → correlación Pearson entre tipos
  - matriz_denue_delito.png      → índice DENUE × tipo de delito
  - reporte_correlacion.csv      → pares Pearson ordenados
  - reporte_correlacion_pct.csv  → tabla completa porcentual
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Paleta ────────────────────────────────────────────────────────────────────
FONDO  = '#0d0d1a'
PANEL  = '#131330'
TEXTO  = 'white'
BORDE  = '#2a2a5a'

# ── Diccionarios ──────────────────────────────────────────────────────────────
LUGAR_DELITO = {
    1:  'Calle / vía pública',
    2:  'Vivienda propia',
    3:  'Vivienda ajena',
    4:  'Negocio / empresa',
    5:  'Banco',
    6:  'Transporte público',
    7:  'Vehículo particular',
    8:  'Mercado / tianguis',
    9:  'Escuela',
    10: 'Centro comercial',
    12: 'Bar / cantina / antro',
    13: 'Estacionamiento',
    14: 'Carretera',
    15: 'Parque / área verde',
    16: 'Cajero automático',
    17: 'Gasolinera',
    18: 'Terminal transporte',
    21: 'Restaurante',
    24: 'Hotel / motel',
    36: 'Tienda conveniencia',
    39: 'Farmacia',
    40: 'Joyería',
    41: 'Casa de empeño',
    43: 'Otro comercio',
    31: 'Internet / digital',
    32: 'Por teléfono',
    29: 'Otro lugar',
}

TIPO_DELITO = {
    1:  'Robo vehículo',
    2:  'Robo accesorios',
    3:  'Robo vivienda',
    4:  'Robo transporte',
    5:  'Robo calle',
    6:  'Robo negocio',
    7:  'Robo banco/cajero',
    8:  'Fraude bancario',
    9:  'Extorsión',
    10: 'Amenazas',
    11: 'Lesiones',
    12: 'Secuestro',
    13: 'Delito sexual',
    14: 'Homicidio familiar',
    15: 'Otro delito',
}

DELITO_SUFRIDO = {
    'AP4_2_01': 'Robo vehículo',
    'AP4_2_02': 'Robo accesorios',
    'AP4_2_03': 'Robo vivienda',
    'AP4_2_04': 'Robo transporte',
    'AP4_2_05': 'Robo en calle',
    'AP4_2_06': 'Secuestro',
    'AP4_2_07': 'Fraude bancario',
    'AP4_2_08': 'Extorsión',
    'AP4_2_09': 'Delito sexual',
    'AP4_2_10': 'Lesiones',
    'AP4_2_11': 'Amenazas',
    'AP4_2_12': 'Homicidio familiar',
    'AP4_2_13': 'Otro delito',
}

# Mapeo conceptual DENUE → tipos de delito relacionados
DENUE_A_TIPO = {
    'Bar / Cantina':          ['Robo negocio', 'Lesiones', 'Amenazas', 'Delito sexual'],
    'Antro / Discoteca':      ['Robo negocio', 'Lesiones', 'Delito sexual', 'Amenazas'],
    'Restaurante':            ['Robo negocio', 'Extorsión', 'Amenazas'],
    'Farmacia':               ['Robo negocio', 'Extorsión'],
    'Banco / Financiero':     ['Robo banco/cajero', 'Fraude bancario', 'Extorsión'],
    'Cajero ATM':             ['Robo banco/cajero', 'Robo calle'],
    'Joyería / Relojería':    ['Robo negocio', 'Extorsión'],
    'Gasolinera':             ['Robo negocio', 'Extorsión'],
    'Hotel / Motel':          ['Robo negocio', 'Delito sexual', 'Lesiones'],
    'Tienda conveniencia':    ['Robo negocio', 'Robo calle', 'Extorsión'],
    'Casa de empeño':         ['Robo negocio', 'Fraude bancario'],
    'Licorería':              ['Robo negocio', 'Lesiones'],
    'Supermercado':           ['Robo negocio', 'Robo accesorios'],
    'Estacionamiento':        ['Robo vehículo', 'Robo accesorios'],
    'Casino / Apuestas':      ['Fraude bancario', 'Extorsión', 'Robo negocio'],
    'Telefonía / Electrónica':['Robo negocio', 'Fraude bancario'],
    'Escuela':                ['Robo calle', 'Amenazas', 'Lesiones'],
    'Hospital / Clínica':     ['Robo negocio', 'Extorsión'],
    'Policía / Seguridad':    ['Amenazas', 'Lesiones'],
}

# ══════════════════════════════════════════════════════════════════════════════
# 1. CARGAR DATOS — TMod_Vic NACIONAL para la matriz lugar × tipo
# ══════════════════════════════════════════════════════════════════════════════
print("📂 Cargando TMod_Vic.csv (datos nacionales)...")
tmod = pd.read_csv('TMod_Vic.csv', encoding='latin1')
tmod['LUGAR_STR'] = tmod['AREAM_OCU'].map(LUGAR_DELITO)
tmod['TIPO_STR']  = tmod['BPCOD'].map(TIPO_DELITO)

print(f"  Total registros TMod_Vic: {len(tmod):,}")
print(f"  Con lugar válido:         {tmod['LUGAR_STR'].notna().sum():,}")
print(f"  Con tipo válido:          {tmod['TIPO_STR'].notna().sum():,}")

df_cross = tmod.dropna(subset=['LUGAR_STR', 'TIPO_STR']).copy()
print(f"  Filas para la matriz:     {len(df_cross):,}")

print("\n📂 Cargando gdl_victimas.csv y gdl_denue.csv...")
vic   = pd.read_csv('gdl_victimas.csv')
denue = pd.read_csv('gdl_denue.csv')

# ══════════════════════════════════════════════════════════════════════════════
# 2. TABLA PIVOTE: Lugar × Tipo de Delito
# ══════════════════════════════════════════════════════════════════════════════
print("\n📊 Construyendo tabla pivote Lugar × Tipo de Delito (datos nacionales)...")

pivot_raw = pd.crosstab(df_cross['LUGAR_STR'], df_cross['TIPO_STR'])

# Lugares con al menos 20 registros
pivot_filt = pivot_raw[pivot_raw.sum(axis=1) >= 20].copy()

# Normalizar por fila → porcentaje
pivot_pct = pivot_filt.div(pivot_filt.sum(axis=1), axis=0) * 100

pivot_raw.to_csv('reporte_correlacion_conteos.csv')
pivot_pct.to_csv('reporte_correlacion_pct.csv')
print(f"  Tabla: {pivot_pct.shape[0]} lugares × {pivot_pct.shape[1]} tipos de delito")

# Correlación de Pearson entre tipos (columnas)
corr_matrix = pivot_raw.corr(method='pearson')

# ══════════════════════════════════════════════════════════════════════════════
# 3. FIGURA 1 — Heatmap Principal: Lugar × Tipo de Delito
# ══════════════════════════════════════════════════════════════════════════════
print("\n🎨 Generando Figura 1: Heatmap Lugar × Tipo de Delito...")

n_rows = len(pivot_pct)
fig_h  = max(10, n_rows * 0.6 + 3)

fig1, ax1 = plt.subplots(figsize=(20, fig_h), facecolor=FONDO)
ax1.set_facecolor(PANEL)

# Ordenar: filas por total decreciente, columnas por total decreciente
row_order = pivot_filt.sum(axis=1).sort_values(ascending=False).index
col_order  = pivot_filt.sum(axis=0).sort_values(ascending=False).index
pivot_plot = pivot_pct.loc[row_order, col_order]

sns.heatmap(
    pivot_plot,
    ax=ax1,
    cmap=sns.color_palette("rocket", as_cmap=True),
    linewidths=0.3,
    linecolor=FONDO,
    annot=True,
    fmt='.1f',
    annot_kws={'size': 7.5, 'color': 'white', 'weight': 'bold'},
    cbar_kws={'label': '% del total de ese lugar', 'shrink': 0.8},
    vmin=0,
)

ax1.set_title(
    '🔥 Matriz de Correlación — Lugar del Delito × Tipo de Delito\n'
    '(% de cada tipo de delito dentro de ese lugar — ENVIPE Nacional)',
    color=TEXTO, fontsize=14, fontweight='bold', pad=16
)
ax1.set_xlabel('Tipo de Delito', color=TEXTO, fontsize=11)
ax1.set_ylabel('Lugar donde ocurrió', color=TEXTO, fontsize=11)
ax1.tick_params(colors=TEXTO, labelsize=8.5)
plt.xticks(rotation=38, ha='right', color=TEXTO)
plt.yticks(rotation=0, color=TEXTO)

cbar1 = ax1.collections[0].colorbar
cbar1.ax.yaxis.set_tick_params(color=TEXTO, labelcolor=TEXTO)
cbar1.set_label('% del total de ese lugar', color=TEXTO, fontsize=9)
for sp in ax1.spines.values():
    sp.set_edgecolor(BORDE)

plt.tight_layout(pad=1.8)
fig1.savefig('matriz_lugar_delito.png', dpi=150, bbox_inches='tight', facecolor=FONDO)
plt.close(fig1)
print("  ✅ Guardado: matriz_lugar_delito.png")

# ══════════════════════════════════════════════════════════════════════════════
# 4. FIGURA 2 — Correlación Pearson entre Tipos de Delito
# ══════════════════════════════════════════════════════════════════════════════
print("\n🎨 Generando Figura 2: Correlación Pearson entre tipos de delito...")

fig2, ax2 = plt.subplots(figsize=(13, 11), facecolor=FONDO)
ax2.set_facecolor(PANEL)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap_div = sns.diverging_palette(250, 10, as_cmap=True)

sns.heatmap(
    corr_matrix,
    ax=ax2,
    cmap=cmap_div,
    mask=mask,
    center=0,
    vmin=-1, vmax=1,
    linewidths=0.5,
    linecolor=FONDO,
    annot=True,
    fmt='.2f',
    annot_kws={'size': 8.5, 'weight': 'bold'},
    square=True,
    cbar_kws={'label': 'r de Pearson', 'shrink': 0.7},
)

ax2.set_title(
    '📐 Correlación de Pearson entre Tipos de Delito\n'
    '(según frecuencia por lugar de ocurrencia — ENVIPE Nacional)',
    color=TEXTO, fontsize=13, fontweight='bold', pad=14
)
ax2.tick_params(colors=TEXTO, labelsize=8.5)
plt.xticks(rotation=40, ha='right', color=TEXTO)
plt.yticks(rotation=0, color=TEXTO)

cbar2 = ax2.collections[0].colorbar
cbar2.ax.yaxis.set_tick_params(color=TEXTO, labelcolor=TEXTO)
cbar2.set_label('r de Pearson', color=TEXTO, fontsize=9)
for sp in ax2.spines.values():
    sp.set_edgecolor(BORDE)

plt.tight_layout(pad=1.5)
fig2.savefig('matriz_pearson_tipos.png', dpi=150, bbox_inches='tight', facecolor=FONDO)
plt.close(fig2)
print("  ✅ Guardado: matriz_pearson_tipos.png")

# ══════════════════════════════════════════════════════════════════════════════
# 5. FIGURA 3 — Delitos sufridos GDL (AP4_2_xx) × tipo: tabla de frecuencias
# ══════════════════════════════════════════════════════════════════════════════
print("\n🎨 Generando Figura 3: Delitos sufridos en Guadalajara...")

cols_del = [c for c in vic.columns if c.startswith('AP4_2_') and c in DELITO_SUFRIDO]
delito_freq = {DELITO_SUFRIDO[c]: int((vic[c] == 1).sum()) for c in cols_del}
serie_gdl = pd.Series(delito_freq).sort_values(ascending=False)

fig3, ax3 = plt.subplots(figsize=(14, 7), facecolor=FONDO)
ax3.set_facecolor(PANEL)

colors = plt.cm.plasma(np.linspace(0.15, 0.9, len(serie_gdl)))
bars = ax3.bar(range(len(serie_gdl)), serie_gdl.values, color=colors, edgecolor='none', width=0.7)

for i, (v, bar) in enumerate(zip(serie_gdl.values, bars)):
    ax3.text(i, v + max(serie_gdl) * 0.01, f'{v:,}',
             ha='center', va='bottom', color=TEXTO, fontsize=8.5, fontweight='bold')

ax3.set_xticks(range(len(serie_gdl)))
ax3.set_xticklabels(serie_gdl.index, rotation=38, ha='right', color=TEXTO, fontsize=9)
ax3.set_ylabel('Número de víctimas', color=TEXTO, fontsize=10)
ax3.tick_params(colors=TEXTO)
ax3.set_facecolor(PANEL)
ax3.set_title(
    '🏙 Frecuencia de Delitos Sufridos — Guadalajara (ENVIPE TPer_Vic1)',
    color=TEXTO, fontsize=13, fontweight='bold', pad=12
)
for sp in ax3.spines.values():
    sp.set_edgecolor(BORDE)

plt.tight_layout(pad=1.5)
fig3.savefig('grafico_delitos_gdl.png', dpi=150, bbox_inches='tight', facecolor=FONDO)
plt.close(fig3)
print("  ✅ Guardado: grafico_delitos_gdl.png")

# ══════════════════════════════════════════════════════════════════════════════
# 6. FIGURA 4 — Heatmap DENUE × Tipo de Delito (índice de exposición)
# ══════════════════════════════════════════════════════════════════════════════
print("\n🎨 Generando Figura 4: Índice DENUE × Tipo de Delito...")

denue_counts = (
    denue[denue['categoria'] != 'Otro']['categoria']
    .value_counts()
    .to_dict()
)

tipos_all = sorted(TIPO_DELITO.values())
cats_denue = [c for c in sorted(DENUE_A_TIPO.keys()) if c in denue_counts]

mat_data = []
for cat in cats_denue:
    tipos_rel = DENUE_A_TIPO[cat]
    n_estab   = denue_counts.get(cat, 0)
    fila = {tipo: (n_estab if tipo in tipos_rel else 0) for tipo in tipos_all}
    mat_data.append(fila)

df_denue_mat  = pd.DataFrame(mat_data, index=cats_denue, columns=tipos_all)
df_denue_norm = df_denue_mat.div(
    df_denue_mat.max(axis=1).replace(0, 1), axis=0
) * 100

# Ordenar filas por total de exposición
row_idx = df_denue_mat.sum(axis=1).sort_values(ascending=False).index
df_denue_norm = df_denue_norm.loc[row_idx]
df_denue_mat_ord = df_denue_mat.loc[row_idx]

n_cats = len(cats_denue)
fig4, ax4 = plt.subplots(
    figsize=(20, max(10, n_cats * 0.6 + 3)),
    facecolor=FONDO
)
ax4.set_facecolor(PANEL)

sns.heatmap(
    df_denue_norm,
    ax=ax4,
    cmap='YlOrRd',
    linewidths=0.3,
    linecolor=FONDO,
    annot=df_denue_mat_ord,
    fmt='.0f',
    annot_kws={'size': 7.5, 'color': '#1a1a2e', 'weight': 'bold'},
    cbar_kws={'label': 'Índice de exposición (0–100)', 'shrink': 0.7},
    vmin=0, vmax=100,
)

ax4.set_title(
    '🏢 Índice de Correlación — Categoría DENUE × Tipo de Delito\n'
    '(valor anotado = nº establecimientos en Guadalajara; color = índice de exposición)',
    color=TEXTO, fontsize=12, fontweight='bold', pad=14
)
ax4.set_xlabel('Tipo de Delito', color=TEXTO, fontsize=11)
ax4.set_ylabel('Categoría de Establecimiento (DENUE)', color=TEXTO, fontsize=11)
ax4.tick_params(colors=TEXTO, labelsize=8.5)
plt.xticks(rotation=38, ha='right', color=TEXTO)
plt.yticks(rotation=0, color=TEXTO)

cbar4 = ax4.collections[0].colorbar
cbar4.ax.yaxis.set_tick_params(color=TEXTO, labelcolor=TEXTO)
cbar4.set_label('Índice de exposición (0–100)', color=TEXTO, fontsize=9)
for sp in ax4.spines.values():
    sp.set_edgecolor(BORDE)

plt.tight_layout(pad=1.8)
fig4.savefig('matriz_denue_delito.png', dpi=150, bbox_inches='tight', facecolor=FONDO)
plt.close(fig4)
print("  ✅ Guardado: matriz_denue_delito.png")

# ══════════════════════════════════════════════════════════════════════════════
# 7. REPORTE TEXTO
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*68)
print("📋 TOP: TIPO DE DELITO MÁS FRECUENTE POR LUGAR")
print("═"*68)
for lugar in pivot_plot.index:
    top_tipo = pivot_plot.loc[lugar].idxmax()
    top_pct  = pivot_plot.loc[lugar].max()
    total    = int(pivot_filt.loc[lugar].sum())
    print(f"  {lugar:<30} → {top_tipo:<22} ({top_pct:.1f}%,  n={total:,})")

# Pares Pearson
corr_pairs = []
for i in corr_matrix.index:
    for j in corr_matrix.columns:
        if i < j:
            r = corr_matrix.loc[i, j]
            corr_pairs.append({'Tipo A': i, 'Tipo B': j, 'r Pearson': round(r, 3)})

df_pairs = pd.DataFrame(corr_pairs).sort_values('r Pearson', ascending=False)
df_pairs.to_csv('reporte_correlacion.csv', index=False)

print("\n" + "═"*68)
print("📋 TOP CORRELACIONES PEARSON (tipos de delito, por lugar — r > 0.6)")
print("═"*68)
top_pos = df_pairs[df_pairs['r Pearson'] > 0.6]
top_neg = df_pairs[df_pairs['r Pearson'] < -0.3]

print("\n  Correlaciones POSITIVAS (r > 0.60):")
if not top_pos.empty:
    for _, row in top_pos.head(12).iterrows():
        print(f"    {row['Tipo A']:<22} ↔ {row['Tipo B']:<22}  r = {row['r Pearson']:+.3f}")
else:
    print("    (ninguna supera 0.60)")

print("\n  Correlaciones NEGATIVAS (r < -0.30):")
if not top_neg.empty:
    for _, row in top_neg.head(10).iterrows():
        print(f"    {row['Tipo A']:<22} ↔ {row['Tipo B']:<22}  r = {row['r Pearson']:+.3f}")
else:
    print("    (ninguna supera -0.30)")

print("\n✅ Archivos CSV exportados:")
print("   📄 reporte_correlacion_conteos.csv")
print("   📄 reporte_correlacion_pct.csv")
print("   📄 reporte_correlacion.csv")

print("\n🎉 Script completado. Imágenes generadas:")
print("   📊 matriz_lugar_delito.png    — Heatmap lugar × tipo (% por fila)")
print("   📐 matriz_pearson_tipos.png   — Pearson entre tipos de delito")
print("   📊 grafico_delitos_gdl.png    — Frecuencia delitos en Guadalajara")
print("   🏢 matriz_denue_delito.png    — Índice DENUE × tipo de delito")
