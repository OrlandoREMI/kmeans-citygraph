import matplotlib
matplotlib.use('Agg')
import osmnx as ox
import matplotlib.pyplot as plt

ox.settings.use_cache = True

print("Cargando Zapopan...")
G = ox.graph_from_place("Zapopan, Jalisco, Mexico", network_type='drive')

print("Generando imagen...")

# Crear la figura MANUALMENTE antes de plot_graph
fig, ax = plt.subplots(figsize=(20, 20), facecolor='white')

ox.plot_graph(
    G,
    ax=ax,              # ← pasar el ax ya creado
    show=False,
    close=False,
    bgcolor='white',
    edge_color='black',
    edge_linewidth=0.5,
    node_size=0
)

fig.savefig("mapa_zapopan_final.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("¡Listo!")