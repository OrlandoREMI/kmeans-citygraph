import osmnx as ox
import geopandas as gpd

print("🌐 Descargando centroides de colonias desde OpenStreetMap...")

tags = {
    "place": ["neighbourhood", "suburb"]
}

colonias = ox.features_from_place(
    "Guadalajara, Jalisco, Mexico",
    tags=tags
)

print(f"📦 Se encontraron {len(colonias)} elementos. Filtrando puntos...")

# 1. Ahora filtramos para quedarnos con los PUNTOS
colonias = colonias[colonias.geometry.type == "Point"]

# 2. Rescatar el nombre
if 'name' in colonias.columns:
    colonias['nombre'] = colonias['name']
else:
    colonias['nombre'] = 'Desconocida'

# 3. Limpiar
colonias = colonias[['nombre', 'geometry']].dropna()
colonias = colonias[colonias['nombre'] != ''] 

# 4. Guardar
colonias.to_file("colonias.geojson", driver="GeoJSON")

print(f"✅ ¡Éxito! Se guardaron {len(colonias)} colonias (como puntos) en 'colonias.geojson'")