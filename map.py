import osmnx as ox
import networkx as nx
import folium
import webbrowser

# 1. Descargar grafo
print("Descargando mapa de Guadalajara...")
G = ox.graph_from_place("Guadalajara, Jalisco", network_type='drive')

# 2. Rangos
LAT_MIN, LAT_MAX = 20.60, 20.75
LON_MIN, LON_MAX = -103.50, -103.25

def pedir_ubicacion(tipo):
    while True:
        try:
            print(f"\nIngresa {tipo}")
            lat = float(input(f"Latitud ({LAT_MIN} a {LAT_MAX}): "))
            lon = float(input(f"Longitud ({LON_MIN} a {LON_MAX}): "))

            if LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX:
                return lat, lon
            else:
                print("Fuera de Guadalajara, intenta de nuevo.")

        except ValueError:
            print("Entrada invalida.")

# 3. Inputs
ubicacion_actual = pedir_ubicacion("tu ubicación actual")
destino = pedir_ubicacion("tu destino")

# 4. Nodos
orig_node = ox.distance.nearest_nodes(G, X=ubicacion_actual[1], Y=ubicacion_actual[0])
dest_node = ox.distance.nearest_nodes(G, X=destino[1], Y=destino[0])

# 5. Ruta
try:
    ruta = nx.shortest_path(G, orig_node, dest_node, weight='length')
    distancia = nx.shortest_path_length(G, orig_node, dest_node, weight='length')

    print(f"\nDistancia: {distancia:.2f} m ({distancia/1000:.2f} km)")

    # 6. Coordenadas
    coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in ruta]

    # 🔥 7. Mapa PRO (sin tiles por defecto)
    mapa = folium.Map(
        location=ubicacion_actual,
        zoom_start=14,
        tiles=None  # 👈 evita el error 403
    )

    # 🔹 Capas de mapa (puedes cambiar entre ellas)
    folium.TileLayer(
        tiles="CartoDB positron",
        name="Claro (recomendado)"
    ).add_to(mapa)

    folium.TileLayer(
        tiles="CartoDB dark_matter",
        name="Oscuro"
    ).add_to(mapa)

    folium.TileLayer(
        tiles="https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.png",
        attr="Map tiles by Stamen Design",
        name="Terreno"
    ).add_to(mapa)

    # 🔹 Ruta
    folium.PolyLine(coords, color="blue", weight=5).add_to(mapa)

    # 🔹 Marcadores
    folium.Marker(
        ubicacion_actual,
        tooltip="Inicio",
        icon=folium.Icon(color="green")
    ).add_to(mapa)

    folium.Marker(
        destino,
        tooltip="Destino",
        icon=folium.Icon(color="red")
    ).add_to(mapa)

    # 🔥 Control de capas (botón para cambiar mapa)
    folium.LayerControl().add_to(mapa)

    # 8. Guardar archivo
    archivo = "ruta.html"
    mapa.save(archivo)

    print("Abriendo mapa en tu navegador...")
    webbrowser.open(archivo)

except nx.NetworkXNoPath:
    print("No hay ruta disponible.")