"""
SCRIPT 6 — Mapa Interactivo de Delitos por AGEB
=================================================
Genera un mapa HTML interactivo (Leaflet) para visualizar
los delitos en las AGEB de Guadalajara.
"""
import pandas as pd
import geopandas as gpd
import json
import webbrowser
import os

# ── 1. Cargar AGEBs de Guadalajara ───────────────────────────────────
print("Cargando AGEBs...")
agebs = gpd.read_file("guadalajara_AGEB/2025_14039_A07052026_1549.shp")
agebs["CVEGEO"] = agebs["CVEGEO"].astype(str).str.strip()

# Filtrar por municipio 039 (Guadalajara) asegurando el CRS correcto
agebs_gdl = agebs[agebs["CVEGEO"].str[2:5] == "039"].copy()
if agebs_gdl.crs is None:
    agebs_gdl = agebs_gdl.set_crs(epsg=4326)
else:
    agebs_gdl = agebs_gdl.to_crs(epsg=4326)

# ── 2. Cargar incidencias ────────────────────────────────────────────
print("Cargando base de delitos...")
inc = pd.read_csv("iieg_2023.csv")
inc = inc.dropna(subset=["x", "y", "delito"])

categorias_delito = {
    'Homicidios y Feminicidios': ['homicidio doloso', 'feminicidio'],
    'Lesiones Dolosas': ['lesiones dolosas'],
    'Robo a Persona': ['robo a persona', 'robo a cuentahabientes'],
    'Robo a Negocio': ['robo a negocio', 'robo a bancos']
}

def clasificar_delito(delito_str):
    d = str(delito_str).strip().lower()
    for cat, lista in categorias_delito.items():
        if d in lista:
            return cat
    return None

inc["delito"] = inc["delito"].apply(clasificar_delito)
inc = inc.dropna(subset=["delito"])

# Ver delitos disponibles
delitos = sorted(inc["delito"].unique())
print("\nCategorías de Alto Impacto procesadas:")
for d in delitos:
    print(f"  - {d}")
print("\n")

# ── 3. Contar delitos por AGEB para cada tipo ────────────────────────
print("Cruzando espacialmente (sjoin)...")
gdf_inc = gpd.GeoDataFrame(
    inc,
    geometry=gpd.points_from_xy(inc["x"], inc["y"]),
    crs="EPSG:4326"
).to_crs(agebs_gdl.crs)

# Spatial join: ver qué delitos caen dentro de qué polígono AGEB
join = gpd.sjoin(gdf_inc, agebs_gdl[["CVEGEO", "geometry"]],
                 how="left", predicate="within")

# Tabla pivote: una fila por AGEB, una columna por delito
tabla = join.groupby(["CVEGEO", "delito"]).size().unstack(fill_value=0).reset_index()

# Unir tabla de frecuencias con la geometría original
agebs_final = agebs_gdl.merge(tabla, on="CVEGEO", how="left").fillna(0)

# ── 4. Preparar variables para JavaScript ────────────────────────────
print("Preparando datos para la web...")
# Convertir el mapa de polígonos a GeoJSON en string
geojson_str = agebs_final.to_json()

lista_delitos = [c for c in tabla.columns if c != "CVEGEO"]

# Construir datos de conteo como diccionario rápido para JavaScript
datos_js = {}
for delito in lista_delitos:
    datos_js[delito] = agebs_final.set_index("CVEGEO")[delito].to_dict()

datos_json = json.dumps(datos_js)

# HTML del mapa
print("Generando archivo HTML...")
html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Incidencia delictiva por AGEB — Guadalajara 2023</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; font-family: Arial, sans-serif; background: #1a1a2e; color: white; }}
        #map {{ height: 100vh; width: 100%; }}
        #panel {{
            position: absolute; top: 15px; right: 15px; z-index: 1000;
            background: rgba(255,255,255,0.95); color: #222;
            padding: 15px 18px; border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            min-width: 220px;
        }}
        #panel h3 {{ margin: 0 0 10px 0; font-size: 14px; color: #333; }}
        #selector {{
            width: 100%; padding: 6px; border-radius: 6px;
            border: 1px solid #ccc; font-size: 13px;
        }}
        #info {{
            margin-top: 12px; font-size: 12px; color: #555;
            border-top: 1px solid #eee; padding-top: 8px;
        }}
        #legend {{
            margin-top: 12px; font-size: 11px;
        }}
        .legend-bar {{
            height: 12px; width: 100%;
            background: linear-gradient(to right, #ffffb2, #fd8d3c, #bd0026);
            border-radius: 3px; margin: 4px 0;
        }}
        .legend-labels {{
            display: flex; justify-content: space-between; font-size: 10px; color: #666;
        }}
    </style>
</head>
<body>
<div id="map"></div>
<div id="panel">
    <h3>Delitos de Alto Impacto por AGEB<br>Guadalajara 2023</h3>
    <select id="selector" onchange="actualizarMapa()">
        {"".join(f'<option value="{d}">{d}</option>' for d in lista_delitos)}
    </select>
    <div id="info">Selecciona un delito para visualizar</div>
    <div id="legend">
        <div class="legend-bar"></div>
        <div class="legend-labels">
            <span>0</span><span id="max-label">máx</span>
        </div>
    </div>
</div>

<script>
const geojson = {geojson_str};
const datos   = {datos_json};

const map = L.map("map").setView([20.67, -103.35], 12);

L.tileLayer("https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png", {{
    attribution: "CartoDB"
}}).addTo(map);

let layer = null;

function getColor(val, max) {{
    if (max === 0 || val === 0) return "#f0f0f0";
    const t = val / max;
    if (t < 0.2) return "#ffffb2";
    if (t < 0.4) return "#fecc5c";
    if (t < 0.6) return "#fd8d3c";
    if (t < 0.8) return "#f03b20";
    return "#bd0026";
}}

function actualizarMapa() {{
    const delito = document.getElementById("selector").value;
    const vals   = datos[delito];
    const max    = Math.max(...Object.values(vals));

    document.getElementById("max-label").textContent = max + " casos";
    document.getElementById("info").textContent =
        "Total: " + Object.values(vals).reduce((a,b)=>a+b,0) + " incidencias";

    if (layer) map.removeLayer(layer);

    layer = L.geoJSON(geojson, {{
        style: function(feature) {{
            const cvegeo = feature.properties.CVEGEO;
            const val    = vals[cvegeo] || 0;
            return {{
                fillColor:   getColor(val, max),
                fillOpacity: 0.75,
                color:       "#999",
                weight:      0.5
            }};
        }},
        onEachFeature: function(feature, lyr) {{
            const cvegeo = feature.properties.CVEGEO;
            const val    = vals[cvegeo] || 0;
            lyr.bindTooltip(
                "<b>AGEB:</b> " + cvegeo + "<br>" +
                "<b>" + delito + ":</b> " +
                val + " casos",
                {{ sticky: true }}
            );
        }}
    }}).addTo(map);
}}

// Inicializar con el primer delito
actualizarMapa();
</script>
</body>
</html>
"""

# ── 5. Guardar ───────────────────────────────────────────────────────
salida = "mapa_delitos_ageb.html"
ruta_absoluta = os.path.abspath(salida)

with open(salida, "w", encoding="utf-8") as f:
    f.write(html)

print(f"✅ ¡Completado! El mapa web interactivo se ha guardado en:")
print(f"   {ruta_absoluta}")
print("\nPuedes abrirlo dando doble clic al archivo en tu gestor de archivos o usando un navegador web.")
