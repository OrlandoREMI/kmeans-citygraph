"""
SCRIPT 7 — Dashboard Interactivo Multi-Capa por AGEB
=====================================================
Capas disponibles:
  A) Delitos de Alto Impacto (4 categorías IIEG 2023)
  B) Negocios DENUE (por tipo de establecimiento)
  C) Nivel educativo promedio (RESAGEBURB / Censo 2020)
  D) Población y demografía (Censo 2020)
  E) Vivienda e infraestructura (Censo 2020)
"""
import pandas as pd
import geopandas as gpd
import json
import os

# ── 1. CARGAR AGEBS ──────────────────────────────────────────────────
print("Cargando AGEBs...")
agebs = gpd.read_file("guadalajara_AGEB/2025_14039_A07052026_1549.shp")
agebs["CVEGEO"] = agebs["CVEGEO"].astype(str).str.strip()
agebs_gdl = agebs[agebs["CVEGEO"].str[2:5] == "039"].copy()
agebs_gdl = agebs_gdl.to_crs(epsg=4326) if agebs_gdl.crs else agebs_gdl.set_crs(epsg=4326)
print(f"  → {len(agebs_gdl)} AGEBs de Guadalajara")

# ── 2. DELITOS ALTO IMPACTO ──────────────────────────────────────────
print("Procesando delitos IIEG...")
categorias_delito = {
    'Homicidios y Feminicidios': ['homicidio doloso', 'feminicidio'],
    'Lesiones Dolosas':          ['lesiones dolosas'],
    'Robo a Persona':            ['robo a persona', 'robo a cuentahabientes'],
    'Robo a Negocio':            ['robo a negocio', 'robo a bancos'],
}

inc = pd.read_csv("iieg_2023.csv").dropna(subset=["x", "y", "delito"])
inc["delito"] = inc["delito"].str.strip().str.lower()

def clasificar_delito(d):
    for cat, lista in categorias_delito.items():
        if d in lista:
            return cat
    return None

inc["categoria"] = inc["delito"].apply(clasificar_delito)
inc_alto = inc.dropna(subset=["categoria"])

gdf_inc = gpd.GeoDataFrame(
    inc_alto,
    geometry=gpd.points_from_xy(inc_alto["x"], inc_alto["y"]),
    crs="EPSG:4326"
).to_crs(agebs_gdl.crs)

join_del = gpd.sjoin(gdf_inc, agebs_gdl[["CVEGEO", "geometry"]], how="left", predicate="within")
tabla_del = join_del.groupby(["CVEGEO", "categoria"]).size().unstack(fill_value=0).reset_index()
print(f"  → {len(inc_alto)} registros de delitos alto impacto cruzados")

# ── 3. NEGOCIOS DENUE ────────────────────────────────────────────────
print("Procesando negocios DENUE...")
denue = pd.read_csv("gdl_denue.csv").dropna(subset=["longitud", "latitud", "categoria"])
denue = denue[denue["categoria"] != "Otro"]

gdf_denue = gpd.GeoDataFrame(
    denue,
    geometry=gpd.points_from_xy(denue.longitud, denue.latitud),
    crs="EPSG:4326"
).to_crs(agebs_gdl.crs)

join_dn = gpd.sjoin(gdf_denue, agebs_gdl[["CVEGEO", "geometry"]], how="left", predicate="within")
tabla_dn = join_dn.groupby(["CVEGEO", "categoria"]).size().unstack(fill_value=0).reset_index()
# Prefijo para diferenciar capas
tabla_dn.columns = ["CVEGEO"] + [f"NEG: {c}" for c in tabla_dn.columns if c != "CVEGEO"]
print(f"  → {len(denue)} negocios DENUE cruzados")

# ── 4. DATOS CENSO (RESAGEBURB) ──────────────────────────────────────
print("Procesando datos del Censo (RESAGEBURB)...")
resag = pd.read_csv("RESAGEBURB_14CSV20.csv", encoding="latin1")

# Filtrar solo Guadalajara (MUN=039, AGEB a nivel AGEB no manzana: MZA='000')
resag_gdl = resag[(resag["MUN"] == 39) & (resag["MZA"] == 0)].copy()

# Crear CVEGEO para el merge: ENTIDAD(2) + MUN(3) + LOC(4) + AGEB(4)
resag_gdl["CVEGEO"] = (
    resag_gdl["ENTIDAD"].astype(str).str.zfill(2) +
    resag_gdl["MUN"].astype(str).str.zfill(3) +
    resag_gdl["LOC"].astype(str).str.zfill(4) +
    resag_gdl["AGEB"].astype(str).str.strip()
)

# Seleccionar indicadores relevantes
cols_census = {
    "POBTOT":    "CEN: Población Total",
    "GRAPROES":  "CEN: Escolaridad Promedio (años)",
    "PEA":       "CEN: Población Económ. Activa",
    "POCUPADA":  "CEN: Población Ocupada",
    "PDESOCUP":  "CEN: Población Desocupada",
    "PSINDER":   "CEN: Sin Derechohabiencia (salud)",
    "P15YM_AN":  "CEN: Analfabetismo +15 años",
    "P18YM_PB":  "CEN: Con Educación Básica +18 años",
    "VPH_INTER": "CEN: Viviendas con Internet",
    "VPH_AUTOM": "CEN: Viviendas con Automóvil",
    "VPH_PC":    "CEN: Viviendas con Computadora",
    "VPH_C_SERV":"CEN: Viviendas con Servicios Completos",
    "PCON_DISC": "CEN: Personas con Discapacidad",
    "P60YMAS":   "CEN: Adultos Mayores (+60 años)",
}

cols_sel = ["CVEGEO"] + [c for c in cols_census.keys() if c in resag_gdl.columns]
tabla_cen = resag_gdl[cols_sel].copy()
tabla_cen = tabla_cen.rename(columns=cols_census)

# Limpiar valores no numéricos del INEGI (*, N/D, etc.)
for col in tabla_cen.columns:
    if col != "CVEGEO":
        tabla_cen[col] = pd.to_numeric(tabla_cen[col], errors="coerce").fillna(0)

print(f"  → {len(tabla_cen)} AGEBs del Censo procesadas")

# ── 5. MERGE FINAL ───────────────────────────────────────────────────
print("Uniendo todas las capas...")
agebs_final = agebs_gdl.copy()

# Delitos
if "CVEGEO" in tabla_del.columns:
    agebs_final = agebs_final.merge(tabla_del, on="CVEGEO", how="left")

# DENUE
if "CVEGEO" in tabla_dn.columns:
    agebs_final = agebs_final.merge(tabla_dn, on="CVEGEO", how="left")

# Censo
agebs_final = agebs_final.merge(tabla_cen, on="CVEGEO", how="left")
agebs_final = agebs_final.fillna(0)

# Obtener listas de capas por grupo para el menú
capas_delitos  = [c for c in agebs_final.columns if c in categorias_delito.keys()]
capas_negocios = [c for c in agebs_final.columns if c.startswith("NEG: ")]
capas_censo    = [c for c in agebs_final.columns if c.startswith("CEN: ")]

print(f"  Delitos:  {capas_delitos}")
print(f"  Negocios: {capas_negocios}")
print(f"  Censo:    {capas_censo}")

# ── 6. GENERAR GEOJSON Y DATOS JS ────────────────────────────────────
print("Preparando GeoJSON y datos JS...")
geojson_str = agebs_final.to_json()

todas_capas = capas_delitos + capas_negocios + capas_censo
datos_js = {}
for capa in todas_capas:
    datos_js[capa] = agebs_final.set_index("CVEGEO")[capa].to_dict()

datos_json = json.dumps(datos_js)

# Construir opciones del selector agrupadas (<optgroup>)
def build_options(nombre_grupo, lista):
    opts = f'<optgroup label="{nombre_grupo}">\n'
    for c in lista:
        opts += f'  <option value="{c}">{c.replace("NEG: ", "").replace("CEN: ", "")}</option>\n'
    opts += '</optgroup>\n'
    return opts

opciones_html = (
    build_options("🔴 Delitos de Alto Impacto", capas_delitos) +
    build_options("🏪 Negocios (DENUE)", capas_negocios) +
    build_options("📊 Datos Sociodemográficos (Censo 2020)", capas_censo)
)

# Paletas de color según el grupo
color_paletas = {}
for c in capas_delitos:
    color_paletas[c] = "rojo"
for c in capas_negocios:
    color_paletas[c] = "azul"
for c in capas_censo:
    color_paletas[c] = "verde"

paletas_json = json.dumps(color_paletas)

# ── 7. GENERAR HTML ──────────────────────────────────────────────────
print("Generando HTML del dashboard...")
html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dashboard Delictivo — Guadalajara 2023</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', sans-serif;
      background: #0f1117;
      color: #e8eaf0;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }}

    /* ── HEADER ── */
    #header {{
      background: linear-gradient(135deg, #1a1d2e 0%, #12151f 100%);
      border-bottom: 1px solid #2a2d3d;
      padding: 12px 24px;
      display: flex;
      align-items: center;
      gap: 16px;
      flex-shrink: 0;
      z-index: 2000;
    }}
    #header .dot {{
      width: 10px; height: 10px; border-radius: 50%; background: #e74c3c;
      box-shadow: 0 0 8px #e74c3c;
    }}
    #header h1 {{
      font-size: 16px; font-weight: 700; letter-spacing: 0.5px;
      background: linear-gradient(90deg, #e74c3c, #f39c12);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    #header .subtitle {{
      font-size: 12px; color: #6c7293; margin-left: auto;
    }}

    /* ── LAYOUT ── */
    #main {{
      display: flex;
      flex: 1;
      overflow: hidden;
    }}

    /* ── PANEL LATERAL ── */
    #sidebar {{
      width: 300px;
      background: #1a1d2e;
      border-right: 1px solid #2a2d3d;
      display: flex;
      flex-direction: column;
      overflow-y: auto;
      flex-shrink: 0;
    }}
    .sidebar-section {{
      padding: 16px;
      border-bottom: 1px solid #2a2d3d;
    }}
    .sidebar-section h3 {{
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: #6c7293;
      margin-bottom: 10px;
    }}

    /* ── SELECTOR ── */
    #selector {{
      width: 100%;
      background: #0f1117;
      color: #e8eaf0;
      border: 1px solid #2a2d3d;
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 13px;
      font-family: 'Inter', sans-serif;
      cursor: pointer;
      outline: none;
      transition: border-color 0.2s;
    }}
    #selector:hover {{ border-color: #4a4d6d; }}
    #selector optgroup {{ color: #6c7293; font-weight: 700; }}
    #selector option {{ color: #e8eaf0; background: #1a1d2e; }}

    /* ── STATS CARDS ── */
    .stat-card {{
      background: #0f1117;
      border: 1px solid #2a2d3d;
      border-radius: 8px;
      padding: 12px;
      margin-bottom: 8px;
      transition: border-color 0.2s;
    }}
    .stat-card:hover {{ border-color: #4a4d6d; }}
    .stat-label {{ font-size: 11px; color: #6c7293; margin-bottom: 4px; }}
    .stat-value {{ font-size: 22px; font-weight: 700; color: #e8eaf0; }}
    .stat-sub   {{ font-size: 11px; color: #6c7293; margin-top: 2px; }}

    /* ── LEYENDA ── */
    #legend-bar {{
      height: 10px; border-radius: 4px; margin: 8px 0 4px;
    }}
    .legend-labels {{
      display: flex; justify-content: space-between;
      font-size: 10px; color: #6c7293;
    }}

    /* ── TIPO BADGE ── */
    #tipo-badge {{
      display: inline-block;
      font-size: 10px;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 20px;
      margin-bottom: 10px;
    }}

    /* ── MAPA ── */
    #map {{
      flex: 1;
      z-index: 1;
    }}

    /* ── TOOLTIP ── */
    .leaflet-tooltip {{
      background: #1a1d2e !important;
      border: 1px solid #2a2d3d !important;
      color: #e8eaf0 !important;
      border-radius: 8px !important;
      padding: 8px 12px !important;
      font-family: 'Inter', sans-serif !important;
      font-size: 12px !important;
      box-shadow: 0 4px 15px rgba(0,0,0,0.5) !important;
    }}
  </style>
</head>
<body>

<div id="header">
  <div class="dot"></div>
  <h1>Dashboard Delictivo — Guadalajara 2023</h1>
  <span class="subtitle">IIEG · DENUE · Censo 2020 · AGEB</span>
</div>

<div id="main">
  <!-- SIDEBAR -->
  <div id="sidebar">

    <div class="sidebar-section">
      <h3>Capa a visualizar</h3>
      <span id="tipo-badge">Cargando...</span>
      <select id="selector" onchange="actualizarMapa()">
        {opciones_html}
      </select>
    </div>

    <div class="sidebar-section">
      <h3>Estadísticas de la capa</h3>
      <div class="stat-card">
        <div class="stat-label">Total en Guadalajara</div>
        <div class="stat-value" id="stat-total">—</div>
        <div class="stat-sub" id="stat-sub">Selecciona una capa</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">AGEB con mayor concentración</div>
        <div class="stat-value" id="stat-max-val">—</div>
        <div class="stat-sub" id="stat-max-ageb">—</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">AGEBs con presencia</div>
        <div class="stat-value" id="stat-agebs">—</div>
        <div class="stat-sub">de {len(agebs_gdl)} AGEBs totales</div>
      </div>
    </div>

    <div class="sidebar-section">
      <h3>Escala de color</h3>
      <div id="legend-bar"></div>
      <div class="legend-labels">
        <span>0</span>
        <span id="legend-mid">—</span>
        <span id="legend-max">máx</span>
      </div>
    </div>

    <div class="sidebar-section">
      <h3>Instrucciones</h3>
      <p style="font-size:11px; color:#6c7293; line-height:1.6;">
        Selecciona una capa del menú superior.<br>
        Pasa el cursor sobre cada AGEB para ver el valor exacto.<br>
        Las zonas en blanco tienen valor 0 o sin dato.
      </p>
    </div>

  </div><!-- /sidebar -->

  <!-- MAPA -->
  <div id="map"></div>
</div>

<script>
const geojson  = {geojson_str};
const datos    = {datos_json};
const paletas  = {paletas_json};

// Inicializar mapa
const map = L.map("map", {{ zoomControl: true }}).setView([20.668, -103.347], 12);

L.tileLayer("https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png", {{
  attribution: "© OpenStreetMap contributors, © CARTO",
  maxZoom: 18
}}).addTo(map);

let layer = null;

// Paletas de color
function getColor(val, max, tipo) {{
  if (max === 0 || val === 0) return "#1e2130";
  const t = Math.min(val / max, 1);

  if (tipo === "rojo") {{
    // Amarillo → Naranja → Rojo oscuro
    if (t < 0.2) return "#3d1a1a";
    if (t < 0.4) return "#7a1e1e";
    if (t < 0.6) return "#c0392b";
    if (t < 0.8) return "#e74c3c";
    return "#ff1744";
  }} else if (tipo === "azul") {{
    // Azul oscuro → Azul brillante
    if (t < 0.2) return "#0d1b3e";
    if (t < 0.4) return "#1a3a7a";
    if (t < 0.6) return "#1565c0";
    if (t < 0.8) return "#1e88e5";
    return "#42a5f5";
  }} else {{
    // Verde oscuro → Verde brillante
    if (t < 0.2) return "#0d2d1a";
    if (t < 0.4) return "#1b5e20";
    if (t < 0.6) return "#2e7d32";
    if (t < 0.8) return "#43a047";
    return "#66bb6a";
  }}
}}

function getLegendGradient(tipo) {{
  if (tipo === "rojo")  return "linear-gradient(to right, #1e2130, #3d1a1a, #c0392b, #ff1744)";
  if (tipo === "azul")  return "linear-gradient(to right, #1e2130, #0d1b3e, #1565c0, #42a5f5)";
  return "linear-gradient(to right, #1e2130, #0d2d1a, #2e7d32, #66bb6a)";
}}

function getBadgeStyle(tipo) {{
  if (tipo === "rojo")  return "background:#3d1a1a; color:#ff1744; border:1px solid #ff1744;";
  if (tipo === "azul")  return "background:#0d1b3e; color:#42a5f5; border:1px solid #42a5f5;";
  return "background:#0d2d1a; color:#66bb6a; border:1px solid #66bb6a;";
}}

function getBadgeText(capa) {{
  if (paletas[capa] === "rojo")  return "🔴 Delito";
  if (paletas[capa] === "azul")  return "🏪 Negocio";
  return "📊 Sociodemográfico";
}}

function actualizarMapa() {{
  const capa  = document.getElementById("selector").value;
  const vals  = datos[capa];
  const tipo  = paletas[capa] || "verde";
  const allVals = Object.values(vals);
  const max   = Math.max(...allVals);
  const total = allVals.reduce((a, b) => a + b, 0);
  const withPresence = allVals.filter(v => v > 0).length;

  // Actualizar badge
  const badge = document.getElementById("tipo-badge");
  badge.textContent = getBadgeText(capa);
  badge.style.cssText = getBadgeStyle(tipo);

  // Actualizar stats
  document.getElementById("stat-total").textContent = total.toLocaleString("es-MX");
  document.getElementById("stat-sub").textContent   = capa.replace("NEG: ","").replace("CEN: ","");
  document.getElementById("stat-agebs").textContent = withPresence.toLocaleString("es-MX");

  // Encontrar AGEB con mayor valor
  const maxEntry = Object.entries(vals).reduce((a, b) => b[1] > a[1] ? b : a, ["—", 0]);
  document.getElementById("stat-max-val").textContent = maxEntry[1].toLocaleString("es-MX");
  document.getElementById("stat-max-ageb").textContent = "AGEB " + maxEntry[0];

  // Leyenda
  document.getElementById("legend-bar").style.background = getLegendGradient(tipo);
  document.getElementById("legend-mid").textContent = Math.round(max / 2).toLocaleString("es-MX");
  document.getElementById("legend-max").textContent  = max.toLocaleString("es-MX");

  // Remover capa anterior
  if (layer) map.removeLayer(layer);

  layer = L.geoJSON(geojson, {{
    style: function(feature) {{
      const cvegeo = feature.properties.CVEGEO;
      const val    = vals[cvegeo] || 0;
      return {{
        fillColor:   getColor(val, max, tipo),
        fillOpacity: val > 0 ? 0.82 : 0.25,
        color:       "#000",
        weight:      0.4
      }};
    }},
    onEachFeature: function(feature, lyr) {{
      const cvegeo = feature.properties.CVEGEO;
      const val    = vals[cvegeo] || 0;
      const label  = capa.replace("NEG: ","").replace("CEN: ","");
      lyr.bindTooltip(
        `<b style="font-size:13px;">AGEB ${{cvegeo}}</b><br>` +
        `<span style="color:#a0a3b1;">${{label}}:</span> ` +
        `<b style="color:#fff;">${{val.toLocaleString("es-MX")}}</b>`,
        {{ sticky: true, opacity: 1 }}
      );
    }}
  }}).addTo(map);
}}

// Inicializar con primera opción
actualizarMapa();
</script>
</body>
</html>"""

# ── 8. GUARDAR ────────────────────────────────────────────────────────
salida = "mapa_dashboard.html"
ruta_abs = os.path.abspath(salida)
with open(salida, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\n✅ Dashboard guardado en:\n   {ruta_abs}")
print(f"\nCapas disponibles ({len(todas_capas)}):")
print(f"  Delitos:   {len(capas_delitos)}")
print(f"  Negocios:  {len(capas_negocios)}")
print(f"  Censo:     {len(capas_censo)}")
print("\nAbre el archivo .html en tu navegador para verlo.")
