"""
SCRIPT 2 — Análisis de correlación espacial
=============================================
Inputs:  gdl_denue.csv, gdl_delitos.csv, gdl_victimas.csv, gdl_percepcion.csv
         (generados por 01_preparar_datos.py)

Análisis:
  A) ¿Qué tipos de establecimiento concentran más delitos cerca?
  B) ¿Dónde ocurren los delitos según AREAM_OCU?
  C) ¿Qué colonias se perciben más inseguras?
  D) Perfil de víctimas (sexo, edad) por tipo de delito
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ── Cargar datos ──────────────────────────────────────────────────────────────
denue   = pd.read_csv('gdl_denue.csv')
delitos = pd.read_csv('gdl_delitos.csv')
vic     = pd.read_csv('gdl_victimas.csv')
perc    = pd.read_csv('gdl_percepcion.csv')

# ── Diccionarios ──────────────────────────────────────────────────────────────
LUGAR_DELITO = {
    1: 'Calle / vía pública', 2: 'Vivienda propia', 3: 'Vivienda ajena',
    4: 'Negocio / empresa', 5: 'Banco', 6: 'Transporte público',
    7: 'Vehículo particular', 8: 'Mercado / tianguis', 9: 'Escuela',
    10: 'Centro comercial', 12: 'Bar / cantina / antro',
    13: 'Estacionamiento', 14: 'Carretera', 15: 'Parque / área verde',
    16: 'Cajero automático', 17: 'Gasolinera', 18: 'Terminal transporte',
    21: 'Restaurante', 24: 'Hotel / motel', 36: 'Tienda conveniencia',
    39: 'Farmacia', 40: 'Joyería', 41: 'Casa de empeño', 43: 'Otro comercio',
}

TIPO_DELITO = {
    1: 'Robo vehículo', 2: 'Robo accesorios', 3: 'Robo vivienda',
    4: 'Robo transporte', 5: 'Robo calle', 6: 'Robo negocio',
    7: 'Robo banco/cajero', 8: 'Fraude bancario', 9: 'Extorsión',
    10: 'Amenazas', 11: 'Lesiones', 12: 'Secuestro',
    13: 'Delito sexual', 14: 'Homicidio familiar', 15: 'Otro',
}

DELITO_SUFRIDO = {
    'AP4_2_01': 'Robo vehículo',    'AP4_2_02': 'Robo accesorios',
    'AP4_2_03': 'Robo vivienda',    'AP4_2_04': 'Robo transporte',
    'AP4_2_05': 'Robo en calle',    'AP4_2_06': 'Secuestro',
    'AP4_2_07': 'Fraude bancario',  'AP4_2_08': 'Extorsión',
    'AP4_2_09': 'Delito sexual',    'AP4_2_10': 'Lesiones',
    'AP4_2_11': 'Amenazas',         'AP4_2_12': 'Homicidio familiar',
    'AP4_2_13': 'Otro',
}

delitos['LUGAR_STR']       = delitos['AREAM_OCU'].map(LUGAR_DELITO)
delitos['TIPO_DELITO_STR'] = delitos['BPCOD'].map(TIPO_DELITO)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 1: Donde ocurren los delitos + tipos más frecuentes
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 14), facecolor='#0d0d1a')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

COLOR_BAR  = '#E05A5A'
COLOR_BAR2 = '#5AA0E0'
COLOR_BAR3 = '#50C878'
FONDO      = '#0d0d1a'
TEXTO      = 'white'

def estilo_ax(ax, title):
    ax.set_facecolor('#131330')
    ax.set_title(title, color=TEXTO, fontsize=11, fontweight='bold', pad=8)
    ax.tick_params(colors=TEXTO, labelsize=8)
    ax.xaxis.label.set_color(TEXTO)
    ax.yaxis.label.set_color(TEXTO)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a2a5a')

# ── Panel A: Lugares donde ocurren delitos ────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
lugar_cnt = delitos['LUGAR_STR'].value_counts().dropna().head(15)
bars = ax1.barh(range(len(lugar_cnt)), lugar_cnt.values,
                color=COLOR_BAR, edgecolor='none', height=0.7)
ax1.set_yticks(range(len(lugar_cnt)))
ax1.set_yticklabels(lugar_cnt.index, color=TEXTO, fontsize=8)
for i, v in enumerate(lugar_cnt.values):
    ax1.text(v + 10, i, str(v), color=TEXTO, va='center', fontsize=7.5)
estilo_ax(ax1, '⚠ Lugares donde ocurren los delitos (Jalisco ENVIPE)')
ax1.set_xlabel('Número de delitos', color=TEXTO)

# ── Panel B: Tipos de delito ───────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
tipo_cnt = delitos['TIPO_DELITO_STR'].value_counts().dropna().head(10)
colors_t = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(tipo_cnt)))
ax2.barh(range(len(tipo_cnt)), tipo_cnt.values,
         color=colors_t, edgecolor='none', height=0.7)
ax2.set_yticks(range(len(tipo_cnt)))
ax2.set_yticklabels(tipo_cnt.index, color=TEXTO, fontsize=8)
estilo_ax(ax2, '🔴 Tipos de delito')

# ── Panel C: Delitos sufridos en Guadalajara (ENVIPE víctimas) ────────────────
ax3 = fig.add_subplot(gs[1, 0])
cols_del = [c for c in vic.columns if c.startswith('AP4_2_')]
delito_freq = {DELITO_SUFRIDO[c]: (vic[c] == 1).sum()
               for c in cols_del if c in DELITO_SUFRIDO}
delito_s = pd.Series(delito_freq).sort_values(ascending=True)
ax3.barh(range(len(delito_s)), delito_s.values,
         color=COLOR_BAR2, edgecolor='none', height=0.7)
ax3.set_yticks(range(len(delito_s)))
ax3.set_yticklabels(delito_s.index, color=TEXTO, fontsize=8)
estilo_ax(ax3, '🏙 Delitos sufridos en Guadalajara')

# ── Panel D: Percepción de seguridad ─────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
seg_dia   = perc['AP1_1'].map({1: 'Sí', 2: 'No', 3: 'NS'}).value_counts()
seg_noche = perc['AP1_2'].map({1: 'Sí', 2: 'No', 3: 'NS'}).value_counts()
x = np.arange(3)
etiq = ['Sí', 'No', 'NS/NC']
vals_dia   = [seg_dia.get(e, 0) for e in ['Sí', 'No', 'NS']]
vals_noche = [seg_noche.get(e, 0) for e in ['Sí', 'No', 'NS']]
ax4.bar(x - 0.2, vals_dia,   0.35, label='De día',   color='#4CAF50', alpha=0.85)
ax4.bar(x + 0.2, vals_noche, 0.35, label='De noche', color='#E05A5A', alpha=0.85)
ax4.set_xticks(x)
ax4.set_xticklabels(etiq, color=TEXTO)
ax4.legend(fontsize=9, labelcolor=TEXTO, facecolor='#131330', edgecolor='#2a2a5a')
estilo_ax(ax4, '🏘 ¿Colonia segura? (Guadalajara)')

# ── Panel E: Perfil de víctima (edad) ────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
vic_solo = vic[vic[[c for c in vic.columns if c.startswith('AP4_2_')]].eq(1).any(axis=1)]
bins = [15, 25, 35, 45, 55, 65, 100]
labels = ['15-24', '25-34', '35-44', '45-54', '55-64', '65+']
vic_solo = vic_solo.copy()
vic_solo['RANGO_EDAD'] = pd.cut(vic_solo['EDAD'], bins=bins, labels=labels, right=False)
edad_sex = vic_solo.groupby(['RANGO_EDAD', 'SEXO']).size().unstack(fill_value=0)
if 1 in edad_sex.columns:
    ax5.bar(range(len(labels)), edad_sex[1].values, 0.4,
            color='#5AA0E0', label='Hombre', alpha=0.85)
if 2 in edad_sex.columns:
    ax5.bar([i + 0.4 for i in range(len(labels))], edad_sex[2].values, 0.4,
            color='#E080A0', label='Mujer', alpha=0.85)
ax5.set_xticks([i + 0.2 for i in range(len(labels))])
ax5.set_xticklabels(labels, color=TEXTO, fontsize=8)
ax5.legend(fontsize=9, labelcolor=TEXTO, facecolor='#131330', edgecolor='#2a2a5a')
estilo_ax(ax5, '👤 Perfil de víctimas por edad y sexo')

fig.suptitle("Análisis de Incidencia Delictiva — Guadalajara / Jalisco (ENVIPE + DENUE)",
             color='white', fontsize=15, fontweight='bold', y=0.98)

fig.savefig('analisis_correlacion.png', dpi=150, bbox_inches='tight', facecolor=FONDO)
plt.close(fig)
print("✅ Guardado: analisis_correlacion.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 2: Establecimientos DENUE más relevantes por categoría
# ══════════════════════════════════════════════════════════════════════════════
fig2, ax = plt.subplots(figsize=(14, 9), facecolor=FONDO)
ax.set_facecolor('#131330')

cat_cnt = denue[denue['categoria'] != 'Otro']['categoria'].value_counts().sort_values()
colors_d = plt.cm.plasma(np.linspace(0.15, 0.9, len(cat_cnt)))
bars = ax.barh(range(len(cat_cnt)), cat_cnt.values,
               color=colors_d, edgecolor='none', height=0.75)
ax.set_yticks(range(len(cat_cnt)))
ax.set_yticklabels(cat_cnt.index, color='white', fontsize=10)
for i, v in enumerate(cat_cnt.values):
    ax.text(v + 5, i, f'{v:,}', color='white', va='center', fontsize=9)

ax.set_title('Establecimientos por categoría de riesgo — Guadalajara (DENUE INEGI)',
             color='white', fontsize=13, fontweight='bold', pad=10)
ax.tick_params(colors='white')
ax.set_xlabel('Número de establecimientos', color='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#2a2a5a')

fig2.savefig('denue_categorias.png', dpi=150, bbox_inches='tight', facecolor=FONDO)
plt.close(fig2)
print("✅ Guardado: denue_categorias.png")

# ══════════════════════════════════════════════════════════════════════════════
# TABLA RESUMEN: Correlación lugar del delito ↔ categoría DENUE
# ══════════════════════════════════════════════════════════════════════════════
print("\n📊 CORRELACIÓN: Lugar del delito (ENVIPE) ↔ Categorías DENUE")
print("─" * 65)
correlacion = {
    'Calle / vía pública':    ['Bar / Cantina', 'Cajero ATM', 'Gasolinera', 'Tienda conveniencia'],
    'Negocio / empresa':      ['Bar / Cantina', 'Joyería / Relojería', 'Farmacia', 'Tienda conveniencia'],
    'Banco':                  ['Banco / Financiero', 'Cajero ATM'],
    'Bar / cantina / antro':  ['Bar / Cantina', 'Antro / Discoteca', 'Licorería'],
    'Tienda conveniencia':    ['Tienda conveniencia'],
    'Cajero automático':      ['Cajero ATM', 'Banco / Financiero'],
    'Centro comercial':       ['Supermercado', 'Joyería / Relojería', 'Estacionamiento'],
    'Hotel / motel':          ['Hotel / Motel'],
    'Casa de empeño':         ['Casa de empeño'],
    'Farmacia':               ['Farmacia'],
    'Joyería':                ['Joyería / Relojería'],
    'Estacionamiento':        ['Estacionamiento'],
}
for lugar, cats in correlacion.items():
    total_estab = sum(len(denue[denue['categoria'] == c]) for c in cats)
    print(f"  {lugar:<30} → {', '.join(cats)} ({total_estab:,} estab. en GDL)")

print("\n✅ Script completado.")