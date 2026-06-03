"""
SCRIPT 10 — Perfil de la Víctima: Individuo vs Delitos (ENVIPE GDL 2023)
========================================================================
Analiza qué perfiles sociodemográficos (Edad y Sexo) son más vulnerables 
a ciertos tipos de delitos, probando la "Teoría del Objetivo Expuesto".

Genera 4 evidencias visuales:
  1. Pirámide de Victimización (Volumen total por Edad y Sexo)
  2. Heatmap de Riesgo Específico (Tipo de Delito vs Perfil Demográfico)
  3. Gráfico de Barras: Top 5 Delitos por Sexo
  4. Distribución de tipos de delito a lo largo de la vida (Violencia vs Fraude/Patrimonio)

Fuentes:
  - gdl_delitos.csv  → Módulo de delitos (incidencias)
  - gdl_victimas.csv → Módulo de víctimas (perfil sociodemográfico)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA E INTEGRACIÓN DE DATOS
# ─────────────────────────────────────────────────────────────────────────────
print("1. Cargando bases de datos (ENVIPE: Delitos + Víctimas)...")
df_delitos  = pd.read_csv('gdl_delitos.csv')
df_victimas = pd.read_csv('gdl_victimas.csv')

# Unimos delitos con la información de la víctima usando el ID_PER (ID de Persona)
# df_delitos tiene una fila por delito, df_victimas tiene una fila por persona.
df_full = pd.merge(
    df_delitos,
    df_victimas[['ID_PER', 'EDAD', 'SEXO_STR']],
    on='ID_PER',
    how='inner',
    suffixes=('', '_vic')
)

# Limpiamos nombres de variables que pudieron duplicarse y usamos las del módulo de víctimas
df_full['SEXO_STR'] = df_full['SEXO_STR_vic'].combine_first(df_full['SEXO_STR'])
df_full['EDAD'] = df_full['EDAD_vic'].combine_first(df_full['EDAD'])

print(f"   → {len(df_full):,} delitos vinculados exitosamente a un perfil de víctima.")

# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPARACIÓN DE PERFILES (GRUPOS DE EDAD Y CLASIFICACIÓN DE DELITOS)
# ─────────────────────────────────────────────────────────────────────────────
print("2. Procesando perfiles demográficos y categorías delictivas...")

# Crear rangos de edad
bins   = [17, 29, 44, 59, 100]
labels = ['1. Joven (18-29)', '2. Adulto (30-44)', '3. Maduro (45-59)', '4. Mayor (60+)']
df_full['Rango_Edad'] = pd.cut(df_full['EDAD'], bins=bins, labels=labels)

# Clasificar los delitos en dos grandes grupos (Violencia / Exposición Física vs Patrimonio / Fraude)
delitos_violencia_calle = [
    'Robo en calle (a transeúnte)', 'Robo en transporte público', 'Robo total de vehículo',
    'Robo de accesorios de vehículo', 'Lesiones', 'Amenazas/intimidación', 'Delito sexual', 'Secuestro'
]
delitos_patrimonio_casa = [
    'Extorsión', 'Fraude bancario', 'Robo en vivienda', 'Robo en banco/cajero',
    'Robo en negocio', 'Homicidio de familiar', 'Otro delito'
]

def clasificar_delito(d):
    if d in delitos_violencia_calle: return 'Callejero / Exposición Física'
    if d in delitos_patrimonio_casa: return 'Patrimonial / Fraude'
    return 'Otro'

df_full['Categoria_Riesgo'] = df_full['TIPO_DELITO_STR'].apply(clasificar_delito)

# Factor de expansión (estimación real poblacional)
df_full['Volumen_Estimado'] = df_full['FAC_DEL'].fillna(1)


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICA 1 — PIRÁMIDE DE VICTIMIZACIÓN (EDAD Y SEXO)
# ══════════════════════════════════════════════════════════════════════════════
print("3. Generando Pirámide de Victimización...")

# Agrupamos volumen total por Sexo y Edad
piramide = df_full.groupby(['Rango_Edad', 'SEXO_STR'])['Volumen_Estimado'].sum().reset_index()

fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# Separar hombres y mujeres para que crezcan en direcciones opuestas
hombres = piramide[piramide['SEXO_STR'] == 'Hombre'].copy()
mujeres = piramide[piramide['SEXO_STR'] == 'Mujer'].copy()

# Hacemos negativos los valores de hombres para el efecto pirámide
hombres['Volumen_Estimado'] = -hombres['Volumen_Estimado']

bar_h = ax1.barh(hombres['Rango_Edad'], hombres['Volumen_Estimado'], color='#2980b9', label='Hombres', height=0.6)
bar_m = ax1.barh(mujeres['Rango_Edad'], mujeres['Volumen_Estimado'], color='#8e44ad', label='Mujeres', height=0.6)

# Añadir etiquetas de datos
for bar in bar_h:
    ax1.text(bar.get_width() - 5000, bar.get_y() + bar.get_height()/2, 
             f"{abs(int(bar.get_width())):,}", va='center', ha='right', color='white', fontweight='bold')
for bar in bar_m:
    ax1.text(bar.get_width() + 5000, bar.get_y() + bar.get_height()/2, 
             f"{int(bar.get_width()):,}", va='center', ha='left', color='white', fontweight='bold')

ax1.axvline(0, color='black', linewidth=1)
ax1.set_title("Pirámide de Victimización: ¿Quién sufre más delitos en total?", fontsize=14, fontweight='bold', pad=20)
ax1.set_xlabel("Volumen Total Estimado de Delitos", fontsize=11)
ax1.set_ylabel("Grupo de Edad", fontsize=11)

# Ajustar ticks del eje X para que sean positivos en ambos lados
ticks = ax1.get_xticks()
ax1.set_xticklabels([f"{abs(int(t)):,}" for t in ticks])

ax1.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
plt.tight_layout()
plt.savefig('ind_01_piramide_victimizacion.png', dpi=150, facecolor='white')
plt.close()
print("   ✔ ind_01_piramide_victimizacion.png")


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICA 2 — HEATMAP DE RIESGO ESPECÍFICO (TIPO DE DELITO VS PERFIL)
# ══════════════════════════════════════════════════════════════════════════════
print("4. Generando Heatmap de Riesgo Específico...")

# Crear columna combinada de Perfil
df_full['Perfil'] = df_full['SEXO_STR'] + " - " + df_full['Rango_Edad'].astype(str).str.split('. ').str[1]

# Pivot table: Delito (filas) vs Perfil (columnas) -> Suma de Volumen
heatmap_data = df_full.groupby(['TIPO_DELITO_STR', 'Perfil'])['Volumen_Estimado'].sum().unstack(fill_value=0)

# Normalizar por columna (qué porcentaje del volumen de ese perfil representa ese delito)
heatmap_pct = heatmap_data.div(heatmap_data.sum(axis=0), axis=1) * 100

# Ordenar delitos por volumen total
top_delitos = heatmap_data.sum(axis=1).sort_values(ascending=False).index
heatmap_pct = heatmap_pct.loc[top_delitos]

fig2, ax2 = plt.subplots(figsize=(12, 8))
cmap_risk = sns.light_palette("#c0392b", as_cmap=True)

sns.heatmap(
    heatmap_pct, 
    annot=True, fmt=".1f", cmap=cmap_risk,
    linewidths=0.5, linecolor='white',
    cbar_kws={"shrink": 0.8, "label": "% del total de delitos sufridos por ese perfil"},
    ax=ax2, annot_kws={"size": 9, "fontweight": "bold"}
)

# Añadir símbolo '%' a las celdas
for t in ax2.texts: t.set_text(t.get_text() + "%")

ax2.set_title(
    "Mapa de Calor de Riesgo Específico (Composición delictiva por perfil)\n"
    "Lectura vertical: De todos los delitos sufridos por [Perfil], el X% fue [Tipo de Delito]",
    fontsize=13, fontweight='bold', pad=15
)
ax2.set_xlabel("Perfil de la Víctima (Sexo y Edad)", fontsize=11, labelpad=10)
ax2.set_ylabel("Tipo de Delito", fontsize=11, labelpad=10)
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('ind_02_heatmap_riesgo.png', dpi=150, facecolor='white')
plt.close()
print("   ✔ ind_02_heatmap_riesgo.png")


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICA 3 — TOP 5 DELITOS POR SEXO (BARRAS COMPARATIVAS)
# ══════════════════════════════════════════════════════════════════════════════
print("5. Generando comparativa Top 5 Delitos por Sexo...")

# Volumen por delito y sexo
delitos_sexo = df_full.groupby(['SEXO_STR', 'TIPO_DELITO_STR'])['Volumen_Estimado'].sum().reset_index()

hombres_top5 = delitos_sexo[delitos_sexo['SEXO_STR'] == 'Hombre'].nlargest(5, 'Volumen_Estimado')
mujeres_top5 = delitos_sexo[delitos_sexo['SEXO_STR'] == 'Mujer'].nlargest(5, 'Volumen_Estimado')

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 6))

# Panel Hombres
sns.barplot(data=hombres_top5, y='TIPO_DELITO_STR', x='Volumen_Estimado', color='#2980b9', ax=ax3a)
ax3a.set_title("Top 5 Delitos contra Hombres", fontsize=14, fontweight='bold', color='#2980b9')
ax3a.set_xlabel("Volumen Estimado", fontsize=11)
ax3a.set_ylabel("")
for p in ax3a.patches:
    ax3a.annotate(f"{int(p.get_width()):,}", (p.get_width() - 5000, p.get_y() + p.get_height() / 2),
                  ha='right', va='center', color='white', fontweight='bold', fontsize=11)

# Panel Mujeres
sns.barplot(data=mujeres_top5, y='TIPO_DELITO_STR', x='Volumen_Estimado', color='#8e44ad', ax=ax3b)
ax3b.set_title("Top 5 Delitos contra Mujeres", fontsize=14, fontweight='bold', color='#8e44ad')
ax3b.set_xlabel("Volumen Estimado", fontsize=11)
ax3b.set_ylabel("")
for p in ax3b.patches:
    ax3b.annotate(f"{int(p.get_width()):,}", (p.get_width() - 5000, p.get_y() + p.get_height() / 2),
                  ha='right', va='center', color='white', fontweight='bold', fontsize=11)

fig3.suptitle("Diferencias de Vulnerabilidad según Género (ENVIPE 2023)", fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('ind_03_top_delitos_sexo.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✔ ind_03_top_delitos_sexo.png")


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICA 4 — EVOLUCIÓN DEL TIPO DE RIESGO A LO LARGO DE LA VIDA
# ══════════════════════════════════════════════════════════════════════════════
print("6. Generando Evolución de Riesgos a lo largo de la vida...")

# Agrupamos por Edad y Categoría de Riesgo (Violencia/Exposición vs Patrimonio/Fraude)
# Calculamos proporciones al 100% por grupo de edad
evolucion = df_full.groupby(['Rango_Edad', 'Categoria_Riesgo'])['Volumen_Estimado'].sum().unstack(fill_value=0)
evolucion_pct = evolucion.div(evolucion.sum(axis=1), axis=0) * 100

fig4, ax4 = plt.subplots(figsize=(10, 6))

colors = {'Callejero / Exposición Física': '#e74c3c', 'Patrimonial / Fraude': '#34495e', 'Otro': '#bdc3c7'}
evolucion_pct[['Callejero / Exposición Física', 'Patrimonial / Fraude']].plot(
    kind='bar', stacked=True, color=[colors[c] for c in ['Callejero / Exposición Física', 'Patrimonial / Fraude']], ax=ax4
)

ax4.axhline(50, color='white', linestyle='--', alpha=0.7)

# Etiquetas internas
for c in ax4.containers:
    ax4.bar_label(c, fmt='%.1f%%', label_type='center', color='white', fontweight='bold')

ax4.set_title(
    "Transición de Vulnerabilidad a lo largo de la Vida\n"
    "(Jóvenes expuestos en la calle vs Adultos Mayores vulnerables en su patrimonio)",
    fontsize=14, fontweight='bold', pad=15
)
ax4.set_xlabel("Grupo de Edad", fontsize=12)
ax4.set_ylabel("% del Total de Delitos Sufridos", fontsize=12)
ax4.tick_params(axis='x', rotation=0)

ax4.legend(title="Naturaleza del Delito", loc='upper right', bbox_to_anchor=(1.35, 1))

plt.tight_layout()
plt.savefig('ind_04_evolucion_vida.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✔ ind_04_evolucion_vida.png")
