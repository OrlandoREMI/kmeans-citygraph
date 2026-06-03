"""
red_vial_tiempos.py
====================
Módulo para calcular tiempos de traslado entre dos puntos cualesquiera
de la red vial de un municipio, usando OpenStreetMap + osmnx.

ENTRADA: Coordenadas (lat, lon) de origen y destino
SALIDA:  Red vial con costos en segundos por segmento + ruta óptima
"""

import os
import pickle
import numpy as np
import osmnx as ox
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

ox.settings.use_cache = True

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DE PENALIZACIÓN Y VELOCIDADES POR DEFECTO
# ─────────────────────────────────────────────────────────────────────────────

COSTO_SEMAFORO_SEG = 40          # Penalización promedio por semáforo (segundos)

# Velocidades de referencia (km/h) según tipo de vía, cuando OSM no tiene dato
VELOCIDADES_DEFAULT = {
    'motorway':       90,
    'motorway_link':  60,
    'trunk':          70,
    'trunk_link':     50,
    'primary':        50,
    'primary_link':   40,
    'secondary':      40,
    'secondary_link': 30,
    'tertiary':       30,
    'tertiary_link':  25,
    'unclassified':   25,
    'residential':    20,
    'living_street':  10,
    'service':        15,
    'road':           25,
}
VELOCIDAD_FALLBACK = 25          # Si el tipo de calle no está en el dict anterior


# ─────────────────────────────────────────────────────────────────────────────
# 1. DESCARGAR RED VIAL
# ─────────────────────────────────────────────────────────────────────────────

def descargar_red(municipio: str) -> nx.MultiDiGraph:
    """
    ENTRADA: Nombre del municipio (ej. "Guadalajara, Jalisco, Mexico")
    SALIDA:  MultiDiGraph de OSM con la red vial de manejo (drive)
    """
    print(f"[1/4] Descargando red vial de '{municipio}'...")
    G = ox.graph_from_place(municipio, network_type='drive')
    print(f"      Nodos: {len(G.nodes):,}  |  Aristas: {len(G.edges):,}")
    return G


# ─────────────────────────────────────────────────────────────────────────────
# 2. AGREGAR TIEMPOS DE TRASLADO
# ─────────────────────────────────────────────────────────────────────────────

def agregar_tiempos_de_traslado(red: nx.MultiDiGraph,
                                 municipio: str | None = None,
                                 penalizar_semaforos: bool = True
                                 ) -> nx.MultiDiGraph:
    """
    Asigna `travel_time_real` (segundos) a cada arista de la red.

    Lógica:
    1. Si el segmento tiene velocidad registrada en OSM → se usa directamente.
    2. Si no tiene velocidad → se imputa según el tipo de calle (highway tag).
    3. Si hay semáforo al nodo destino → se suma COSTO_SEMAFORO_SEG.

    ENTRADA: red vial (MultiDiGraph), municipio para obtener semáforos (opcional)
    SALIDA:  misma red con atributo 'travel_time_real' en cada arista (segundos)
    """
    print("[2/4] Calculando tiempos de traslado por segmento...")

    # Paso A: velocidades de OSM (llena los que tienen 'maxspeed')
    red = ox.add_edge_speeds(red, fallback=VELOCIDAD_FALLBACK)
    red = ox.add_edge_travel_times(red)          # travel_time = length / speed_kph * 3.6

    # Paso B: imputar velocidades en vías sin dato usando highway tag
    for u, v, k, data in red.edges(data=True, keys=True):
        if data.get('speed_kph', 0) == VELOCIDAD_FALLBACK:
            highway = data.get('highway', '')
            if isinstance(highway, list):
                highway = highway[0]
            vel_imputada = VELOCIDADES_DEFAULT.get(str(highway), VELOCIDAD_FALLBACK)
            distancia    = data.get('length', 1)
            data['speed_kph']    = vel_imputada
            data['travel_time']  = (distancia / vel_imputada) * 3.6   # segundos

    # Paso C: penalización por semáforos
    nodos_semaforo = set()
    if penalizar_semaforos and municipio:
        try:
            print("      Descargando semáforos de OSM...")
            semaforos = ox.features_from_place(municipio,
                                               tags={'highway': 'traffic_signals'})
            semaforos = semaforos[semaforos.geometry.type == 'Point']
            nodos_semaforo = set(
                ox.distance.nearest_nodes(
                    red,
                    X=semaforos.geometry.x,
                    Y=semaforos.geometry.y
                )
            )
            print(f"      Semáforos encontrados: {len(nodos_semaforo):,}")
        except Exception as e:
            print(f"      (Aviso: no se pudieron cargar semáforos — {e})")

    # Paso D: escribir travel_time_real en cada arista
    for u, v, k, data in red.edges(data=True, keys=True):
        tiempo_base = data.get('travel_time', 1)
        penalizacion = COSTO_SEMAFORO_SEG if v in nodos_semaforo else 0
        data['travel_time_real'] = tiempo_base + penalizacion

    total_aristas = len(red.edges)
    print(f"      Tiempos asignados a {total_aristas:,} aristas.")
    return red


# ─────────────────────────────────────────────────────────────────────────────
# 3. GUARDAR RED
# ─────────────────────────────────────────────────────────────────────────────

def guardar_red(red: nx.MultiDiGraph, municipio: str, carpeta: str = "redes") -> str:
    """
    Persiste la red en disco (formato pickle) para no re-descargarla.

    ENTRADA: red, nombre del municipio, carpeta destino
    SALIDA:  ruta del archivo guardado
    """
    os.makedirs(carpeta, exist_ok=True)
    nombre_archivo = municipio.replace(",", "").replace(" ", "_").lower() + ".pkl"
    ruta = os.path.join(carpeta, nombre_archivo)
    with open(ruta, 'wb') as f:
        pickle.dump(red, f)
    print(f"[3/4] Red guardada en '{ruta}'")
    return ruta


# ─────────────────────────────────────────────────────────────────────────────
# 4. CARGAR RED
# ─────────────────────────────────────────────────────────────────────────────

def cargar_red(municipio: str, carpeta: str = "redes") -> nx.MultiDiGraph | None:
    """
    Recupera la red guardada para un municipio.

    ENTRADA: nombre del municipio, carpeta donde se guardó
    SALIDA:  MultiDiGraph con la red, o None si no existe en disco
    """
    nombre_archivo = municipio.replace(",", "").replace(" ", "_").lower() + ".pkl"
    ruta = os.path.join(carpeta, nombre_archivo)
    if os.path.exists(ruta):
        print(f"[INFO] Cargando red desde '{ruta}'...")
        with open(ruta, 'rb') as f:
            return pickle.load(f)
    print(f"[INFO] No existe red guardada para '{municipio}'. Se descargará.")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 5. OBTENER O CONSTRUIR LA RED (helper)
# ─────────────────────────────────────────────────────────────────────────────

def obtener_red(municipio: str,
                forzar_descarga: bool = False,
                penalizar_semaforos: bool = True
                ) -> nx.MultiDiGraph:
    """
    Orquesta descarga/carga + cálculo de tiempos.
    Si ya existe en disco y no se fuerza, la carga directamente.
    """
    red = None if forzar_descarga else cargar_red(municipio)

    if red is None:
        red = descargar_red(municipio)
        red = agregar_tiempos_de_traslado(red, municipio, penalizar_semaforos)
        guardar_red(red, municipio)

    return red


# ─────────────────────────────────────────────────────────────────────────────
# 6. CALCULAR TIEMPO DE TRASLADO ENTRE DOS PUNTOS
# ─────────────────────────────────────────────────────────────────────────────

def calcular_traslado(red: nx.MultiDiGraph,
                       origen_lat: float, origen_lon: float,
                       destino_lat: float, destino_lon: float
                       ) -> dict:
    """
    Calcula la ruta óptima (menor tiempo) entre dos coordenadas.

    ENTRADA:
        red         — red vial con 'travel_time_real' en aristas
        origen_lat  — latitud del punto de partida
        origen_lon  — longitud del punto de partida
        destino_lat — latitud del destino
        destino_lon — longitud del destino

    SALIDA:  dict con:
        'nodo_origen'    — ID del nodo OSM más cercano al origen
        'nodo_destino'   — ID del nodo OSM más cercano al destino
        'ruta_nodos'     — lista de nodos en la ruta
        'tiempo_seg'     — tiempo total estimado en segundos
        'tiempo_min'     — tiempo total estimado en minutos
        'distancia_m'    — distancia total en metros
        'num_segmentos'  — cantidad de aristas en la ruta
    """
    # Nodos más cercanos a las coordenadas
    # nearest_nodes devuelve array; [0] extrae el escalar
    nodo_orig = ox.distance.nearest_nodes(red, X=origen_lon,  Y=origen_lat)
    nodo_dest = ox.distance.nearest_nodes(red, X=destino_lon, Y=destino_lat)
    # Si devuelve array/lista, tomar el primer elemento
    if hasattr(nodo_orig, '__iter__'):
        nodo_orig = int(list(nodo_orig)[0])
    if hasattr(nodo_dest, '__iter__'):
        nodo_dest = int(list(nodo_dest)[0])

    # Ruta de menor costo (Dijkstra sobre travel_time_real)
    ruta = nx.shortest_path(red, nodo_orig, nodo_dest, weight='travel_time_real')

    # Acumular tiempo y distancia
    tiempo_total   = 0.0
    distancia_total = 0.0
    for u, v in zip(ruta[:-1], ruta[1:]):
        # En multigrafos puede haber varias aristas entre u y v; tomamos la más rápida
        aristas   = red[u][v]
        mejor     = min(aristas.values(), key=lambda d: d.get('travel_time_real', float('inf')))
        tiempo_total    += mejor.get('travel_time_real', 0)
        distancia_total += mejor.get('length', 0)

    return {
        'nodo_origen':   nodo_orig,
        'nodo_destino':  nodo_dest,
        'ruta_nodos':    ruta,
        'tiempo_seg':    round(tiempo_total, 1),
        'tiempo_min':    round(tiempo_total / 60, 2),
        'distancia_m':   round(distancia_total, 1),
        'num_segmentos': len(ruta) - 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. VISUALIZAR RUTA SOBRE EL MAPA
# ─────────────────────────────────────────────────────────────────────────────

def visualizar_ruta(red: nx.MultiDiGraph,
                    resultado: dict,
                    origen_lat: float, origen_lon: float,
                    destino_lat: float, destino_lon: float,
                    titulo: str = "Ruta óptima",
                    archivo_salida: str = "ruta_optima.png"
                    ) -> None:
    """
    Dibuja la red vial con la ruta calculada resaltada.
    También muestra el mapa de calor de velocidades (como referencia).

    ENTRADA:
        red            — red vial con tiempos
        resultado      — dict devuelto por calcular_traslado()
        origen_*       — coordenadas del punto de partida
        destino_*      — coordenadas del destino
        titulo         — título del mapa
        archivo_salida — nombre del PNG a guardar
    """
    print(f"[4/4] Generando visualización...")

    fig, ax = plt.subplots(figsize=(22, 20), facecolor='#0d0d1a')
    ax.set_facecolor('#0d0d1a')

    # ── Calor de velocidades en fondo ──────────────────────────────────────
    cmap = plt.colormaps['RdYlGn']
    norm = plt.Normalize(vmin=10, vmax=70)
    lines_bg, colors_bg, widths_bg = [], [], []

    for u, v, data in red.edges(data=True):
        coords = (
            list(zip(*data['geometry'].xy))
            if 'geometry' in data
            else [(red.nodes[u]['x'], red.nodes[u]['y']),
                  (red.nodes[v]['x'], red.nodes[v]['y'])]
        )
        lines_bg.append(coords)
        vel = data.get('speed_kph', VELOCIDAD_FALLBACK)
        colors_bg.append(cmap(norm(vel)))
        widths_bg.append(0.6 if vel > 40 else 0.25)

    lc_bg = LineCollection(lines_bg, colors=colors_bg,
                           linewidths=widths_bg, alpha=0.35, zorder=1)
    ax.add_collection(lc_bg)

    # ── Ruta resaltada ─────────────────────────────────────────────────────
    ruta      = resultado['ruta_nodos']
    lines_rt  = []
    for u, v in zip(ruta[:-1], ruta[1:]):
        aristas = red[u][v]
        mejor   = min(aristas.values(), key=lambda d: d.get('travel_time_real', float('inf')))
        coords  = (
            list(zip(*mejor['geometry'].xy))
            if 'geometry' in mejor
            else [(red.nodes[u]['x'], red.nodes[u]['y']),
                  (red.nodes[v]['x'], red.nodes[v]['y'])]
        )
        lines_rt.append(coords)

    lc_rt = LineCollection(lines_rt, colors='#00CFFF',
                           linewidths=2.5, alpha=0.95, zorder=3)
    ax.add_collection(lc_rt)

    # ── Marcadores de origen y destino ─────────────────────────────────────
    ax.scatter([origen_lon],  [origen_lat],
               c='#00FF88', s=180, zorder=5, edgecolors='white', linewidths=1.5,
               label=f'Origen  ({origen_lat:.4f}, {origen_lon:.4f})')
    ax.scatter([destino_lon], [destino_lat],
               c='#FF4466', s=180, zorder=5, edgecolors='white', linewidths=1.5,
               label=f'Destino ({destino_lat:.4f}, {destino_lon:.4f})')

    ax.autoscale()

    # ── Información del traslado ────────────────────────────────────────────
    info = (f"⏱  Tiempo estimado: {resultado['tiempo_min']:.1f} min  "
            f"({resultado['tiempo_seg']:.0f} s)\n"
            f"📏  Distancia: {resultado['distancia_m']/1000:.2f} km  "
            f"|  Segmentos: {resultado['num_segmentos']}")

    ax.text(0.02, 0.97, info,
            transform=ax.transAxes, fontsize=12, color='white',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a3e',
                      edgecolor='#00CFFF', alpha=0.85))

    # ── Leyenda y estética ─────────────────────────────────────────────────
    legend = ax.legend(fontsize=10, loc='lower left',
                       framealpha=0.8, facecolor='#0d0d1a',
                       edgecolor='#555577', labelcolor='white')

    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label('Velocidad promedio (km/h)', color='white', fontsize=12)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    ax.set_title(titulo, color='white', fontsize=18, fontweight='bold', pad=16)
    ax.axis('off')

    fig.savefig(archivo_salida, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    plt.close(fig)
    print(f"      Mapa guardado como '{archivo_salida}'")


# ─────────────────────────────────────────────────────────────────────────────
# 8. PIPELINE COMPLETO (punto de entrada principal)
# ─────────────────────────────────────────────────────────────────────────────

def calcular_y_visualizar(municipio: str,
                           origen_lat: float, origen_lon: float,
                           destino_lat: float, destino_lon: float,
                           forzar_descarga: bool = False,
                           penalizar_semaforos: bool = True,
                           archivo_salida: str = "ruta_optima.png"
                           ) -> dict:

    # 1. Obtener red (desde disco o descargando)
    red = obtener_red(municipio, forzar_descarga, penalizar_semaforos)

    # 2. Calcular traslado
    print("[3/4] Calculando ruta óptima...")
    resultado = calcular_traslado(red, origen_lat, origen_lon,
                                       destino_lat, destino_lon)

    print(f"\n{'─'*50}")
    print(f"  Tiempo estimado : {resultado['tiempo_min']:.1f} min")
    print(f"  Distancia       : {resultado['distancia_m']/1000:.2f} km")
    print(f"  Segmentos       : {resultado['num_segmentos']}")
    print(f"{'─'*50}\n")

    # 3. Visualizar
    titulo = (f"Ruta óptima — {municipio.split(',')[0]}\n"
              f"({resultado['tiempo_min']:.1f} min | "
              f"{resultado['distancia_m']/1000:.2f} km)")

    visualizar_ruta(red, resultado,
                    origen_lat, origen_lon,
                    destino_lat, destino_lon,
                    titulo=titulo,
                    archivo_salida=archivo_salida)

    return resultado


# ─────────────────────────────────────────────────────────────────────────────
# DEMO: ejecutar directamente con coordenadas de ejemplo en Guadalajara
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# COORDENADAS APROXIMADAS TERRITORIO: GUADALAJARA, JALISCO
# ─────────────────────────────────────────────────────────────────────────────
# Centro geográfico aproximado:
#   Latitud:  20.6597
#   Longitud: -103.3496
#
# Bounding Box (Límites de la caja delimitadora del municipio):
#   Norte (Latitud máxima):   20.7490
#   Sur   (Latitud mínima):   20.5970
#   Este  (Longitud máxima): -103.2530
#   Oeste (Longitud mínima): -103.3980
# ─────────────────────────────────────────────────────────────────────────────
""""
if __name__ == "__main__":
    MUNICIPIO = "Guadalajara, Jalisco, Mexico"

    # Ejemplo: Plaza Tapatía → Minerva
    resultado = calcular_y_visualizar(
        municipio        = MUNICIPIO,
        origen_lat       = 20.6771,    # Plaza Tapatía (aprox)
        origen_lon       = -103.3470,
        destino_lat      = 20.6734,    # Glorieta Minerva (aprox)
        destino_lon      = -103.3876,
        penalizar_semaforos = True,
        archivo_salida   = "ruta_optima_gdl.png"
    )
"""
if __name__ == "__main__":
    # 1. Definimos la nueva ciudad/municipio
    MUNICIPIO = "Morelia, Michoacan, Mexico"

    # 2. Usamos coordenadas que estén estrictamente DENTRO de Morelia
    resultado = calcular_y_visualizar(
        municipio        = MUNICIPIO,
        origen_lat       = 19.7124,    # Catedral de Morelia
        origen_lon       = -101.1923,
        destino_lat      = 19.7027,    # Fuente de las Tarascas
        destino_lon      = -101.1805,
        penalizar_semaforos = True,
        # 3. Cambiamos el nombre del archivo para no sobreescribir el de GDL
        archivo_salida   = "ruta_optima_morelia1.png" 
    )

# ─────────────────────────────────────────────────────────────────────────────
# DICCIONARIO DE PUNTOS DE PRUEBA
# ─────────────────────────────────────────────────────────────────────────────
PUNTOS_PRUEBA = {
    "Catedral de GDL":      (20.6771, -103.3470),
    "Hospicio Cabañas":     (20.6770, -103.3375),
    "Teatro Degollado":     (20.6773, -103.3443),
    "Templo Expiatorio":    (20.6738, -103.3556),
    "Glorieta Minerva":     (20.6743, -103.3875),
    "Arcos de GDL":         (20.6738, -103.3813),
    "Expo Guadalajara":     (20.6542, -103.3908),
    "Basílica de Zapopan":  (20.7214, -103.3920),
    "Estadio Akron":        (20.6817, -103.4626)
}

# Ejemplo de uso para probar varias rutas rápidamente:
# origen_nombre, origen_coords = "Catedral de GDL", PUNTOS_PRUEBA["Catedral de GDL"]
# destino_nombre, destino_coords = "Expo Guadalajara", PUNTOS_PRUEBA["Expo Guadalajara"]
#
# calcular_y_visualizar(
#     municipio="Guadalajara, Jalisco, Mexico", # Ojo: cambiar si usas puntos de Zapopan
#     origen_lat=origen_coords[0], origen_lon=origen_coords[1],
#     destino_lat=destino_coords[0], destino_lon=destino_coords[1],
#     archivo_salida=f"ruta_{origen_nombre.replace(' ', '_')}_a_{destino_nombre.replace(' ', '_')}.png"
# )