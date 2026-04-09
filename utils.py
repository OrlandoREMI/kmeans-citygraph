"""
utils.py — Infraestructura de red vial para modelo de despliegue policial
"""

# ===========================================================================
# FASE 1 — Construcción del grafo
# Responsable: ___________
# ===========================================================================
# ENTRADA:  Nombre de un municipio
# SALIDA:   Red vial del municipio donde cada segmento tiene un costo en segundos


def descargar_red(municipio):
    # TODO: Descargar red vial de OpenStreetMap para el municipio
    raise NotImplementedError


def agregar_tiempos_de_traslado(red):
    # TODO: Asignar tiempo en segundos a cada segmento de la red.
    #       Si el segmento no tiene velocidad registrada, imputar
    #       según el tipo de calle (avenida, residencial, etc.)
    raise NotImplementedError


def guardar_red(red, municipio):
    # TODO: Persistir la red en disco para no re-descargarla en cada ejecución
    raise NotImplementedError


def cargar_red(municipio):
    # TODO: Recuperar la red guardada para un municipio
    raise NotImplementedError


# ===========================================================================
# FASE 2 — Proyección y snapping
# Responsable: ___________
# ===========================================================================
# ENTRADA:  Tabla de incidentes con coordenadas, red vial del municipio
# SALIDA:   Cada incidente asignado a su nodo más cercano en la red


def convertir_coordenadas(incidentes):
    # TODO: Convertir lat/lon a sistema métrico plano (metros)
    raise NotImplementedError


def asignar_nodos(incidentes, red):
    # TODO: Para cada incidente, encontrar el nodo de la red más cercano
    raise NotImplementedError


# ===========================================================================
# FASE 3 — Matriz de tiempos
# Responsable: ___________
# ===========================================================================
# ENTRADA:  Nodos de incidentes, nodos candidatos de despliegue, red vial
# SALIDA:   Matriz donde cada celda es el tiempo en segundos entre un
#           candidato de despliegue y un incidente


def calcular_matriz_de_tiempos(nodos_incidentes, nodos_candidatos, red):
    # TODO: Calcular tiempo de ruta más corta para cada par
    #       (candidato, incidente) en la red vial
    raise NotImplementedError
