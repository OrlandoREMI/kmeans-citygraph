import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ox.settings.use_cache = True

# ── 1. CARGAR DATOS ──────────────────────────────────────────────
print("Cargando datos...")
df = pd.read_csv('iieg_2023.csv')
df = df.dropna(subset=['x', 'y'])

# ── 2. CLASIFICAR POR NIVEL DE GRAVEDAD ──────────────────────────
# Criterio: impacto en la vida/integridad física primero,
# luego impacto económico, luego frecuencia
gravedad = {
    # NIVEL 4 — Crítico: atentan contra la vida
    'homicidio doloso':        4,
    'feminicidio':             4,
    'violacion':               4,
    'abuso sexual infantil':   4,

    # NIVEL 3 — Alto: violencia física grave
    'lesiones dolosas':        3,
    'violencia familiar':      3,
    'robo a bancos':           3,
    'robo a carga pesada':     3,

    # NIVEL 2 — Medio: impacto económico directo
    'robo a persona':          2,
    'robo a negocio':          2,
    'robo a casa habitacion':  2,
    'robo a cuentahabientes':  2,

    # NIVEL 1 — Bajo: daño patrimonial menor
    'robo a vehiculos particulares': 1,
    'robo de motocicleta':           1,
    'robo de autopartes':            1,
    'robo a int de vehiculos':       1,
}

etiquetas = {4: 'Crítico', 3: 'Alto', 2: 'Medio', 1: 'Bajo'}
colores_nivel = {4: '#FF0000', 3: '#FF8C00', 2: '#FFD700', 1: '#00BFFF'}

df['nivel']    = df['delito'].map(gravedad)
df['etiqueta'] = df['nivel'].map(etiquetas)

# ── 3. PUNTAJE PONDERADO POR COLONIA ────────────────────────────
# Cada incidente vale su nivel de gravedad (no solo contar)
# Colonia con muchos homicidios pesa más que colonia con muchos robos de autopartes
colonias = df.groupby(['colonia', 'x', 'y']).agg(
    total=('delito', 'count'),
    puntaje=('nivel', 'sum'),          # suma de gravedad
    criticos=('nivel', lambda x: (x == 4).sum()),
    altos=('nivel', lambda x: (x == 3).sum()),
    medios=('nivel', lambda x: (x == 2).sum()),
    bajos=('nivel', lambda x: (x == 1).sum()),
).reset_index()

# Puntaje promedio por incidente (qué tan grave es la zona en promedio)
colonias['gravedad_promedio'] = colonias['puntaje'] / colonias['total']
colonias['prioridad'] = colonias['puntaje'].rank(ascending=False).astype(int)
colonias = colonias.sort_values('puntaje', ascending=False)

print("=== TOP 15 COLONIAS POR PUNTAJE DE GRAVEDAD ===")
print(colonias[['colonia','total','puntaje','criticos','gravedad_promedio','prioridad']].head(15).to_string())

# ── 4. MAPA VIAL ─────────────────────────────────────────────────
print("Cargando mapa...")
G = ox.graph_from_place("Guadalajara, Jalisco, Mexico", network_type='drive')

# ── 5. FIGURA: 3 PANELES ─────────────────────────────────────────
fig = plt.figure(figsize=(28, 18), facecolor='#0d0d1a')
fig.suptitle("Prioridad de patrullaje por gravedad de delitos — Guadalajara 2023",
             fontsize=20, fontweight='bold', color='white', y=0.98)

# ─── Panel 1: Mapa con puntos coloreados por nivel ───────────────
ax1 = fig.add_subplot(1, 3, (1, 2))
ax1.set_facecolor('#0d0d1a')

ox.plot_graph(G, ax=ax1, show=False, close=False,
              bgcolor='none', edge_color='#2a2a4a',
              edge_linewidth=0.25, node_size=0)

# Dibujar de menos a más grave para que los críticos queden encima
for nivel in [1, 2, 3, 4]:
    subset = df[df['nivel'] == nivel]
    color  = colores_nivel[nivel]
    label  = etiquetas[nivel]
    size   = {1: 2, 2: 3, 3: 5, 4: 8}[nivel]
    alpha  = {1: 0.25, 2: 0.35, 3: 0.55, 4: 0.85}[nivel]
    ax1.scatter(subset['x'], subset['y'],
                c=color, s=size, alpha=alpha,
                linewidths=0, zorder=3 + nivel,
                label=f'Nivel {nivel} — {label} ({len(subset):,})')

# Marcar top 5 colonias más críticas con etiqueta
top5 = colonias.head(5)
for _, row in top5.iterrows():
    ax1.annotate(
        f"#{int(row['prioridad'])} {row['colonia'].title()}",
        xy=(row['x'], row['y']),
        fontsize=7.5, color='white', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#8B0000',
                  alpha=0.85, edgecolor='#FF0000', linewidth=0.8),
        zorder=9
    )

handles = [
    mpatches.Patch(color=colores_nivel[n],
                   label=f"Nivel {n} — {etiquetas[n]}  ({len(df[df['nivel']==n]):,} casos)")
    for n in [4, 3, 2, 1]
]
legend = ax1.legend(handles=handles, loc='lower left', fontsize=10,
                    facecolor='#0d0d1a', edgecolor='#444466',
                    labelcolor='white', title='Nivel de gravedad',
                    title_fontsize=11)
legend.get_title().set_color('white')
ax1.set_title("Distribución geográfica por gravedad", color='white', fontsize=13, pad=10)
ax1.set_axis_off()

# ─── Panel 2 (arriba der): Top 15 colonias por puntaje ───────────
ax2 = fig.add_subplot(2, 3, 3)
ax2.set_facecolor('#0d0d1a')

top15 = colonias.head(15)

# Color de barra según gravedad promedio de la colonia
norm = plt.Normalize(vmin=1, vmax=4)
cmap = plt.cm.get_cmap('RdYlBu_r')
colores_barras = [cmap(norm(v)) for v in top15['gravedad_promedio']]

bars = ax2.barh(range(len(top15)), top15['puntaje'],
                color=colores_barras, edgecolor='none', height=0.7)

for i, (bar, row) in enumerate(zip(bars, top15.itertuples())):
    ax2.text(row.puntaje + 5, i,
             f"{row.puntaje}  ({row.criticos}★)",
             va='center', color='white', fontsize=8)

ax2.set_yticks(range(len(top15)))
ax2.set_yticklabels([c.title() for c in top15['colonia']],
                    color='white', fontsize=8)
ax2.set_xlabel('Puntaje de gravedad acumulado', color='white', fontsize=9)
ax2.set_title("Top 15 colonias — puntaje ponderado\n(★ = delitos críticos)",
              color='white', fontsize=10, pad=8)
ax2.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_edgecolor('#444466')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_facecolor('#0d0d1a')

# ─── Panel 3 (abajo der): Distribución de niveles ────────────────
ax3 = fig.add_subplot(2, 3, 6)
ax3.set_facecolor('#0d0d1a')

conteo_nivel = df['nivel'].value_counts().sort_index(ascending=False)
etiq_nivel   = [f"Nivel {n}\n{etiquetas[n]}" for n in conteo_nivel.index]
colores_pie  = [colores_nivel[n] for n in conteo_nivel.index]

wedges, texts, autotexts = ax3.pie(
    conteo_nivel.values,
    labels=etiq_nivel,
    colors=colores_pie,
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.75,
    textprops={'color': 'white', 'fontsize': 9}
)
for at in autotexts:
    at.set_fontsize(8)
    at.set_color('white')
    at.set_fontweight('bold')

ax3.set_title("Proporción por nivel de gravedad",
              color='white', fontsize=10, pad=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("gravedad_delitos.png", dpi=150,
            bbox_inches='tight', facecolor='#0d0d1a')
plt.close(fig)
print("¡Listo! Revisa 'gravedad_delitos.png'")