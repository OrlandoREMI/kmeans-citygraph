"""
SCRIPT 09 — Dashboard Maestro Final
===================================
Este script genera el Dashboard Maestro interactivo definitivo ('09_dashboard_completo.html'),
integrando toda la información del proyecto:
  1. Capa Simple (Choropleth por AGEB de Delitos, DENUE, Censo y Entorno Urbano)
  2. Análisis Bivariado (Correlación Pearson, p-valor, Scatter Plot y tendencias)
  3. Mapa de Calor (Heatmaps fluidos de puntos exactos del IIEG y DENUE)
  4. Análisis Temporal (Estacionalidad mensual, días de la semana y catálogo completo de delitos)
  5. Análisis de Colonias (Ranking dinámico de las 20 colonias con más delitos, con búsqueda y geolocalización)
  6. Encuesta ENVIPE (Percepción de seguridad día/noche, problemas principales y perfil sociodemográfico de víctimas)

Utiliza Chart.js para visualizaciones fluidas y Leaflet para mapas dinámicos.
"""

import pandas as pd
import geopandas as gpd
import json
import os
import numpy as np

print("🚀 Iniciando procesamiento para el Dashboard Maestro Final...")

# ── 1. CARGAR AGEBS ──────────────────────────────────────────────────
print("Cargando polígonos de AGEBs...")
agebs = gpd.read_file("guadalajara_AGEB/2025_14039_A07052026_1549.shp")
agebs["CVEGEO"] = agebs["CVEGEO"].astype(str).str.strip()
agebs_gdl = agebs[agebs["CVEGEO"].str[2:5] == "039"].copy()
agebs_gdl = agebs_gdl.to_crs(epsg=4326) if agebs_gdl.crs else agebs_gdl.set_crs(epsg=4326)
print(f"  → {len(agebs_gdl)} AGEBs de Guadalajara cargados.")

# ── 2. DELITOS IIEG (COOP & AGEB & TEMPORAL) ──────────────────────────
print("Procesando delitos IIEG (Completo)...")
categorias_delito = {
    'Homicidios y Feminicidios': ['homicidio doloso', 'feminicidio'],
    'Lesiones Dolosas':          ['lesiones dolosas'],
    'Robo a Persona':            ['robo a persona', 'robo a cuentahabientes'],
    'Robo a Negocio':            ['robo a negocio', 'robo a bancos'],
}

inc = pd.read_csv("iieg_2023.csv").dropna(subset=["x", "y", "delito"])
inc["delito"] = inc["delito"].str.strip().str.lower()

# Clasificación para correlaciones
def clasificar_delito(d):
    for cat, lista in categorias_delito.items():
        if d in lista: return cat
    return None

inc["categoria_alto"] = inc["delito"].apply(clasificar_delito)

# Cargar a GeoDataFrame para cruce espacial
gdf_inc = gpd.GeoDataFrame(
    inc,
    geometry=gpd.points_from_xy(inc["x"], inc["y"]),
    crs="EPSG:4326"
).to_crs(agebs_gdl.crs)

# Cruce con AGEBs
join_del = gpd.sjoin(gdf_inc, agebs_gdl[["CVEGEO", "geometry"]], how="inner", predicate="within")

# Tabla de delitos por AGEB (para capas del coroplético)
tabla_del = join_del.dropna(subset=["categoria_alto"]).groupby(["CVEGEO", "categoria_alto"]).size().unstack(fill_value=0).reset_index()

# ── Guardar puntos exactos para Heatmaps ──
heatmap_data = {}
for cat in join_del["categoria_alto"].dropna().unique():
    cat_pts = join_del[join_del["categoria_alto"] == cat]
    heatmap_data[cat] = cat_pts[["y", "x"]].values.tolist()

# Delito general (todos los puntos de delitos combinados)
heatmap_data["TODOS_DELITOS"] = join_del[["y", "x"]].values.tolist()

# ── Análisis Temporal de Delitos ──
join_del["fecha_dt"] = pd.to_datetime(join_del["fecha"], errors="coerce")
join_del = join_del.dropna(subset=["fecha_dt"])

join_del["mes_num"] = join_del["fecha_dt"].dt.month
join_del["dia_sem_num"] = join_del["fecha_dt"].dt.dayofweek # 0=Lunes, 6=Domingo

# Nombres en español
meses_nombres = {1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun", 7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"}
dias_nombres = {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves", 4: "Viernes", 5: "Sábado", 6: "Domingo"}

# Agregaciones temporales
temporal_meses = join_del["mes_num"].value_counts().sort_index().rename(index=meses_nombres).to_dict()
temporal_dias = join_del["dia_sem_num"].value_counts().sort_index().rename(index=dias_nombres).to_dict()
ranking_delitos_todos = join_del["delito"].str.title().value_counts().to_dict()

temporal_json = {
    "labels_meses": list(temporal_meses.keys()),
    "valores_meses": list(temporal_meses.values()),
    "labels_dias": list(temporal_dias.keys()),
    "valores_dias": list(temporal_dias.values()),
    "labels_delitos": list(ranking_delitos_todos.keys())[:15],
    "valores_delitos": list(ranking_delitos_todos.values())[:15]
}

# ── Análisis de Colonias ──
# Agrupar por colonia, calcular coordenadas medias (centroides) para vuelos de cámara interactivos
colonia_stats = join_del.groupby("colonia").agg(
    total=("delito", "count"),
    lat=("y", "mean"),
    lng=("x", "mean")
).reset_index()

# Ordenar por peligrosidad (total de delitos)
top_colonias = colonia_stats.sort_values(by="total", ascending=False).head(30)
colonias_json = top_colonias.to_dict(orient="records")

# ── 3. NEGOCIOS DENUE (PUNTOS Y AGEB) ────────────────────────────────
print("Procesando negocios DENUE...")
denue = pd.read_csv("gdl_denue.csv").dropna(subset=["longitud", "latitud", "categoria"])
denue = denue[denue["categoria"] != "Otro"].copy()

gdf_denue = gpd.GeoDataFrame(
    denue,
    geometry=gpd.points_from_xy(denue.longitud, denue.latitud),
    crs="EPSG:4326"
).to_crs(agebs_gdl.crs)

join_dn = gpd.sjoin(gdf_denue, agebs_gdl[["CVEGEO", "geometry"]], how="inner", predicate="within")
tabla_dn = join_dn.groupby(["CVEGEO", "categoria"]).size().unstack(fill_value=0).reset_index()
tabla_dn.columns = ["CVEGEO"] + [f"NEG: {c}" for c in tabla_dn.columns if c != "CVEGEO"]

# Guardar puntos para Heatmaps del DENUE
for cat in join_dn["categoria"].unique():
    cat_pts = join_dn[join_dn["categoria"] == cat]
    heatmap_data[f"NEG: {cat}"] = cat_pts[["latitud", "longitud"]].values.tolist()

# ── 4. DATOS CENSO (RESAGEBURB) ──────────────────────────────────────
print("Procesando datos del Censo 2020...")
resag = pd.read_csv("RESAGEBURB_14CSV20.csv", encoding="latin1", low_memory=False)
resag_gdl = resag[(resag["MUN"] == 39) & (resag["MZA"] == 0)].copy()
resag_gdl["CVEGEO"] = (
    resag_gdl["ENTIDAD"].astype(str).str.zfill(2) +
    resag_gdl["MUN"].astype(str).str.zfill(3) +
    resag_gdl["LOC"].astype(str).str.zfill(4) +
    resag_gdl["AGEB"].astype(str).str.strip()
)
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
for col in tabla_cen.columns:
    if col != "CVEGEO":
        tabla_cen[col] = pd.to_numeric(tabla_cen[col], errors="coerce").fillna(0)

# ── 5. ENTORNO URBANO (FRENTES) ──────────────────────────────────────
print("Procesando Entorno Urbano...")
frentes = pd.read_csv("inv_frentes.csv", dtype={"CVEGEO": str, "CVE_MUN": str})
frentes_gdl = frentes[frentes["CVE_MUN"].str.zfill(3) == "039"].copy()
if not frentes_gdl.empty:
    frentes_gdl["CVEGEO_AGEB"] = frentes_gdl["CVEGEO"].str[:13]
    frentes_gdl["alumbrado"] = (frentes_gdl["ALUMPUB_D"] == "Dispone").astype(int)
    frentes_gdl["banqueta"] = (frentes_gdl["BANQUETA_D"] == "Dispone").astype(int)
    frentes_gdl["arboles"] = (frentes_gdl["ARBOLES_D"] == "Dispone").astype(int)
    t_frentes = frentes_gdl.groupby("CVEGEO_AGEB")[["alumbrado", "banqueta", "arboles"]].mean() * 100
    t_frentes = t_frentes.reset_index().rename(columns={
        "CVEGEO_AGEB": "CVEGEO",
        "alumbrado": "URB: % Frentes con Alumbrado",
        "banqueta":  "URB: % Frentes con Banqueta",
        "arboles":   "URB: % Frentes con Árboles"
    })
else:
    t_frentes = pd.DataFrame(columns=["CVEGEO"])

# ── 6. ENCUESTAS ENVIPE Y SDEM (PROCESAMIENTO COMPLETO) ────────────────
print("Procesando encuestas de percepción y víctimas (ENVIPE & SDEM)...")

# gdl_percepcion
df_per = pd.read_csv("gdl_percepcion.csv")
perc_dia = df_per["SEGURA_DIA"].value_counts().to_dict()
perc_noche = df_per["SEGURA_NOCHE"].value_counts().to_dict()
perc_problemas = df_per["PROBLEMA_PRINC"].value_counts().head(10).to_dict()

# gdl_victimas
df_vic = pd.read_csv("gdl_victimas.csv")
# Separar los delitos por comas y sumarlos todos
delitos_sufridos_lista = df_vic["DELITOS_SUFRIDOS"].dropna().str.split(", ").explode()
ranking_victimas_del = delitos_sufridos_lista.value_counts().head(10).to_dict()

# Agrupar edades
def agrupar_edad(e):
    if e < 18: return "Menor de 18"
    elif e <= 29: return "18-29 años"
    elif e <= 44: return "30-44 años"
    elif e <= 59: return "45-59 años"
    else: return "60 años o más"

df_vic["edad_grupo"] = df_vic["EDAD"].apply(agrupar_edad)
edades_dist = df_vic["edad_grupo"].value_counts().to_dict()

# gdl_sdem
df_sdem = pd.read_csv("gdl_sdem.csv")
nivel_edu_map = {
    0: 'Sin escolaridad', 1: 'Preescolar', 2: 'Primaria',
    3: 'Secundaria', 4: 'Carrera técnica con secundaria',
    5: 'Normal básica', 6: 'Preparatoria o bachillerato',
    7: 'Carrera técnica con preparatoria', 8: 'Licenciatura',
    9: 'Posgrado (Maestría/Doctorado)', 99: 'No especificado'
}
df_sdem["NIVEL_EDU"] = df_sdem["NIV"].map(nivel_edu_map).fillna("No especificado")
educacion_dist = df_sdem["NIVEL_EDU"].value_counts().to_dict()

# Armar JSON de encuestas
envipe_json = {
    "perc_dia_labels": list(perc_dia.keys()),
    "perc_dia_values": list(perc_dia.values()),
    "perc_noche_labels": list(perc_noche.keys()),
    "perc_noche_values": list(perc_noche.values()),
    "prob_labels": [k[:25] for k in perc_problemas.keys()], # Cortar nombres largos
    "prob_values": list(perc_problemas.values()),
    "vic_del_labels": [k[:25] for k in ranking_victimas_del.keys()],
    "vic_del_values": list(ranking_victimas_del.values()),
    "edad_labels": ["Menor de 18", "18-29 años", "30-44 años", "45-59 años", "60 años o más"],
    "edad_values": [edades_dist.get("Menor de 18", 0), edades_dist.get("18-29 años", 0), edades_dist.get("30-44 años", 0), edades_dist.get("45-59 años", 0), edades_dist.get("60 años o más", 0)],
    "edu_labels": list(educacion_dist.keys()),
    "edu_values": list(educacion_dist.values())
}

# ── 7. MERGE FINAL DE CAPAS CHOROPLETH ─────────────────────────────────
print("Uniendo todas las capas coropléticas...")
agebs_final = agebs_gdl.copy()
if "CVEGEO" in tabla_del.columns: agebs_final = agebs_final.merge(tabla_del, on="CVEGEO", how="left")
if "CVEGEO" in tabla_dn.columns:  agebs_final = agebs_final.merge(tabla_dn, on="CVEGEO", how="left")
agebs_final = agebs_final.merge(tabla_cen, on="CVEGEO", how="left")
agebs_final = agebs_final.merge(t_frentes, on="CVEGEO", how="left")
agebs_final = agebs_final.fillna(0)

capas_delitos  = list(categorias_delito.keys())
capas_negocios = [c for c in agebs_final.columns if c.startswith("NEG: ")]
capas_censo    = [c for c in agebs_final.columns if c.startswith("CEN: ")]
capas_urbano   = [c for c in agebs_final.columns if c.startswith("URB: ")]

print("Preparando GeoJSON...")
geojson_str = agebs_final.to_json()
todas_capas = capas_delitos + capas_negocios + capas_censo + capas_urbano
datos_js = {capa: agebs_final.set_index("CVEGEO")[capa].to_dict() for capa in todas_capas}

paletas_js = {}
for c in capas_delitos: paletas_js[c] = "rojo"
for c in capas_negocios: paletas_js[c] = "azul"
for c in capas_censo: paletas_js[c] = "verde"
for c in capas_urbano: paletas_js[c] = "naranja"

# Construcción de Selectores HTML
def build_options(lista):
    return "".join(f'<option value="{c}">{c.replace("NEG: ","").replace("CEN: ","").replace("URB: ","")}</option>' for c in lista)

def build_optgroup(label, lista):
    return f'<optgroup label="{label}">{build_options(lista)}</optgroup>'

opts_all = (
    build_optgroup("🔴 Delitos de Alto Impacto", capas_delitos) +
    build_optgroup("🏪 Negocios (DENUE)", capas_negocios) +
    build_optgroup("📊 Sociodemográfico (Censo)", capas_censo) +
    build_optgroup("🏙️ Entorno Urbano", capas_urbano)
)

opts_y_biv = (
    build_optgroup("🏪 Negocios (DENUE)", capas_negocios) +
    build_optgroup("📊 Sociodemográfico (Censo)", capas_censo) +
    build_optgroup("🏙️ Entorno Urbano", capas_urbano)
)

opts_heat = (
    '<option value="TODOS_DELITOS">🔥 Todos los Delitos IIEG (General)</option>' +
    build_optgroup("🔴 Delitos Específicos", capas_delitos) +
    build_optgroup("🏪 Negocios (DENUE)", capas_negocios)
)

# ── 8. GENERAR HTML COMPLETO DEL DASHBOARD ─────────────────────────────
print("Escribiendo código HTML final del Dashboard...")

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dashboard Urbano y Delictivo Maestro — Guadalajara</title>
  
  <!-- CSS Base Premium -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  
  <!-- Scripts Core -->
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://unpkg.com/leaflet.heat/dist/leaflet-heat.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Outfit', sans-serif;
      background: #08090c;
      color: #e2e8f0;
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }}
    
    /* ── HEADER ── */
    #header {{
      background: linear-gradient(135deg, #10121b 0%, #08090d 100%);
      border-bottom: 1px solid #1f2335;
      padding: 10px 24px;
      display: flex;
      align-items: center;
      gap: 16px;
      z-index: 2000;
      flex-shrink: 0;
    }}
    #header .dot {{
      width: 12px; height: 12px; border-radius: 50%; background: #ff4757;
      box-shadow: 0 0 10px #ff4757;
      animation: pulse 2s infinite;
    }}
    @keyframes pulse {{
      0% {{ transform: scale(0.9); opacity: 0.7; }}
      50% {{ transform: scale(1.1); opacity: 1; }}
      100% {{ transform: scale(0.9); opacity: 0.7; }}
    }}
    #header h1 {{
      font-size: 18px; font-weight: 700; letter-spacing: 0.5px;
      background: linear-gradient(90deg, #ff4757, #ff6b81, #ffa502);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    #header .sub {{
      font-size: 12px; color: #57606f; margin-left: auto;
      font-weight: 500;
    }}
    
    /* ── MAIN LAYOUT ── */
    #main {{
      display: flex;
      flex: 1;
      height: calc(100vh - 48px);
      overflow: hidden;
    }}
    
    /* ── SIDEBAR PREMIUM ── */
    #sidebar {{
      width: 380px;
      background: #0f111a;
      border-right: 1px solid #1f2335;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      flex-shrink: 0;
    }}
    
    /* ── TABS ── */
    .tabs {{
      display: flex;
      background: #0b0c13;
      border-bottom: 1px solid #1f2335;
      flex-shrink: 0;
      flex-wrap: wrap;
    }}
    .tab {{
      flex: 1;
      min-width: 90px;
      text-align: center;
      padding: 10px 0;
      font-size: 11px;
      font-weight: 600;
      color: #747d8c;
      cursor: pointer;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      border-bottom: 3px solid transparent;
    }}
    .tab:hover {{
      color: #ffffff;
      background: #151824;
    }}
    .tab.active {{
      color: #ffffff;
      border-bottom-color: #ff4757;
      background: #10121b;
    }}
    
    /* ── CONTENIDO TABS ── */
    .sidebar-scroll {{
      flex: 1;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
    }}
    .tab-content {{
      display: none;
      flex-direction: column;
      padding: 16px;
    }}
    .tab-content.active {{
      display: flex;
    }}
    
    /* ── SECCIONES ── */
    .sec {{
      margin-bottom: 16px;
      background: #141724;
      border: 1px solid #1f2335;
      border-radius: 12px;
      padding: 16px;
      box-shadow: 0 4px 6px rgba(0,0,0,0.15);
    }}
    .sec h3 {{
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      color: #a4b0be;
      margin-bottom: 12px;
      letter-spacing: 0.8px;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    
    /* ── ELEMENTOS DE FORMULARIO ── */
    select {{
      width: 100%;
      background: #0a0b10;
      color: #e2e8f0;
      border: 1px solid #2d324d;
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 13px;
      font-family: inherit;
      cursor: pointer;
      outline: none;
      transition: all 0.3s;
      margin-bottom: 8px;
    }}
    select:hover, select:focus {{
      border-color: #ff4757;
      box-shadow: 0 0 8px rgba(255, 71, 87, 0.2);
    }}
    
    .btn {{
      width: 100%;
      padding: 11px;
      background: linear-gradient(135deg, #ff4757 0%, #ff6b81 100%);
      color: #ffffff;
      border: none;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      font-family: inherit;
      transition: all 0.3s;
      box-shadow: 0 4px 12px rgba(255, 71, 87, 0.3);
    }}
    .btn:hover {{
      transform: translateY(-1px);
      box-shadow: 0 6px 16px rgba(255, 71, 87, 0.4);
    }}
    
    /* ── ESTADÍSTICAS & TARJETAS ── */
    .card {{
      background: #0a0b10;
      border: 1px solid #1f2335;
      border-radius: 8px;
      padding: 12px 14px;
      margin-bottom: 8px;
      transition: all 0.3s;
    }}
    .card:hover {{ border-color: #2d324d; }}
    .card .lbl {{ font-size: 11px; color: #747d8c; margin-bottom: 4px; font-weight: 500; }}
    .card .val {{ font-size: 22px; font-weight: 700; color: #ffffff; }}
    .card .sub {{ font-size: 11px; color: #ff6b81; margin-top: 2px; }}
    
    /* ── LEYENDAS ── */
    #legend-bar {{ height: 10px; border-radius: 5px; margin: 8px 0 6px; }}
    .legend-labels {{ display: flex; justify-content: space-between; font-size: 10px; color: #747d8c; font-weight: 600; }}
    
    #biv-legend {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 2px; width: 90px; height: 90px; margin: 8px auto; }}
    .biv-cell {{ border-radius: 2px; }}
    .biv-labels {{ display: flex; justify-content: space-between; font-size: 9px; color: #747d8c; }}
    
    /* ── SCATTER ── */
    #scatter-canvas {{ width: 100%; height: 160px; border-radius: 8px; background: #0a0b10; border: 1px solid #1f2335; display: block; }}
    
    /* ── R-BADGE ── */
    #r-badge {{ font-size: 26px; font-weight: 700; text-align:center; padding:8px; border-radius:8px; margin-bottom:4px; }}
    #r-interp {{ font-size: 11px; color: #a4b0be; text-align:center; line-height: 1.5; }}
    
    /* ── LISTA DE COLONIAS ── */
    .colonia-item {{
      background: #0a0b10;
      border: 1px solid #1f2335;
      border-radius: 8px;
      padding: 10px 14px;
      margin-bottom: 6px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .colonia-item:hover {{
      border-color: #ff4757;
      background: #141724;
      transform: translateX(4px);
    }}
    .colonia-item .nombre {{ font-size: 12px; font-weight: 600; color: #e2e8f0; }}
    .colonia-item .cifra {{
      font-size: 11px;
      background: #ff475722;
      color: #ff4757;
      border: 1px solid #ff475733;
      padding: 2px 8px;
      border-radius: 20px;
      font-weight: 700;
    }}
    
    /* ── AJUSTES SLIDERS ── */
    input[type=range] {{
      -webkit-appearance: none;
      width: 100%;
      background: #0a0b10;
      height: 6px;
      border-radius: 3px;
      outline: none;
      margin: 8px 0;
    }}
    input[type=range]::-webkit-slider-thumb {{
      -webkit-appearance: none;
      appearance: none;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: #ff4757;
      cursor: pointer;
      transition: 0.2s;
    }}
    input[type=range]::-webkit-slider-thumb:hover {{
      transform: scale(1.2);
    }}
    
    /* ── CONTENEDOR DE CHARTS ── */
    .chart-container {{
      position: relative;
      height: 160px;
      width: 100%;
      margin-top: 8px;
    }}
    .donut-row {{
      display: flex;
      gap: 8px;
      justify-content: space-between;
    }}
    .donut-item {{
      flex: 1;
      text-align: center;
    }}
    .donut-item span {{
      font-size: 10px;
      font-weight: 600;
      color: #747d8c;
      text-transform: uppercase;
    }}
    
    /* ── MAPA ── */
    #map {{
      flex: 1;
      z-index: 1;
      background: #08090c;
    }}
    
    /* ── LEAFLET TOOLTIP PREMIUM ── */
    .leaflet-tooltip {{
      background: #0f111a !important;
      border: 1px solid #1f2335 !important;
      color: #ffffff !important;
      border-radius: 8px !important;
      padding: 10px 14px !important;
      font-family: inherit !important;
      font-size: 12px !important;
      box-shadow: 0 8px 24px rgba(0,0,0,0.5) !important;
      opacity: 1 !important;
    }}
  </style>
</head>
<body>

<div id="header">
  <div class="dot"></div>
  <h1>Dashboard Urbano y Delictivo Maestro — Guadalajara</h1>
  <span class="sub">IIEG · DENUE · Censo 2020 · Frentes · ENVIPE</span>
</div>

<div id="main">
  <!-- SIDEBAR -->
  <div id="sidebar">
    <div class="tabs">
      <div class="tab active" onclick="switchTab('simple')">Simple</div>
      <div class="tab" onclick="switchTab('biv')">Bivariado</div>
      <div class="tab" onclick="switchTab('heat')">Calor</div>
      <div class="tab" onclick="switchTab('temp')">Temporal</div>
      <div class="tab" onclick="switchTab('cols')">Colonias</div>
      <div class="tab" onclick="switchTab('env')">ENVIPE</div>
    </div>
    
    <div class="sidebar-scroll">
      <!-- TAB: SIMPLE -->
      <div id="tab-simple" class="tab-content active">
        <div class="sec">
          <h3>🗺️ Capa Choropleth por AGEB</h3>
          <select id="sel-simple" onchange="updateSimple()">
            {opts_all}
          </select>
        </div>
        <div class="sec">
          <h3>📊 Estadísticas</h3>
          <div class="card"><div class="lbl">Total en Guadalajara</div><div class="val" id="s-tot">—</div></div>
          <div class="card"><div class="lbl">Máximo en un AGEB</div><div class="val" id="s-max">—</div><div class="sub" id="s-max-ag">—</div></div>
        </div>
        <div class="sec">
          <h3>🎨 Escala de Color</h3>
          <div id="legend-bar"></div>
          <div class="legend-labels"><span>0</span><span id="l-mid">—</span><span id="l-max">máx</span></div>
        </div>
      </div>
      
      <!-- TAB: BIVARIADO -->
      <div id="tab-biv" class="tab-content">
        <div class="sec">
          <h3>🔴 Variable X — Delito</h3>
          <select id="sel-bx">{build_options(capas_delitos)}</select>
          <h3 style="margin-top:8px">🔵 Variable Y — Factor</h3>
          <select id="sel-by">{opts_y_biv}</select>
          <button class="btn" onclick="updateBiv()">Analizar correlación ▶</button>
        </div>
        <div class="sec">
          <h3>📈 Coeficiente de Pearson (r)</h3>
          <div id="r-badge">—</div>
          <div id="r-interp">Selecciona variables</div>
        </div>
        <div class="sec">
          <h3>📉 Scatter Plot por AGEB</h3>
          <canvas id="scatter-canvas"></canvas>
          <div style="display:flex;justify-content:space-between;font-size:9px;color:#747d8c;margin-top:3px">
            <span id="x-lbl">X</span><span id="y-lbl">Y</span>
          </div>
        </div>
        <div class="sec">
          <h3>🎨 Leyenda Bivariada</h3>
          <div style="display:flex;gap:12px;align-items:flex-start">
            <div>
              <div id="biv-legend"></div>
              <div class="biv-labels"><span>Bajo X</span><span>Alto X</span></div>
              <div style="font-size:9px;color:#747d8c;text-align:right;margin-top:1px">↑ Alto Y</div>
            </div>
            <div style="font-size:10px;color:#a4b0be;line-height:1.7;margin-top:4px">
              <b style="color:#e74c3c">■</b> Alto delito + bajo factor<br>
              <b style="color:#2980b9">■</b> Bajo delito + alto factor<br>
              <b style="color:#8e44ad">■</b> Alto en ambos<br>
              <b style="color:#2d2d3a;border:1px solid #4a4d6d">■</b> Bajo en ambos
            </div>
          </div>
        </div>
        <div class="sec">
          <h3>🔍 Estadísticas de Validación</h3>
          <div class="card"><div class="lbl">AGEBs analizadas</div><div class="val" id="st-n">—</div></div>
          <div class="card"><div class="lbl">p-valor aproximado</div><div class="val" id="st-p">—</div><div class="sub">Si p &lt; 0.05 la correlación es significativa</div></div>
        </div>
      </div>
      
      <!-- TAB: HEATMAP -->
      <div id="tab-heat" class="tab-content">
        <div class="sec">
          <h3>🔥 Mapa de Calor Dinámico</h3>
          <p style="font-size:12px;color:#747d8c;margin-bottom:10px;">Visualiza la densidad real a partir de puntos de GPS exactos.</p>
          <select id="sel-heat" onchange="updateHeat()">
            {opts_heat}
          </select>
        </div>
        <div class="sec">
          <h3>⚙️ Ajustes Visuales del Calor</h3>
          <label style="font-size:11px;color:#747d8c;font-weight:500;">Radio de difuminado</label>
          <input type="range" id="heat-radius" min="10" max="60" value="25" oninput="updateHeatProps()">
          <label style="font-size:11px;color:#747d8c;font-weight:500;margin-top:8px;display:block;">Intensidad máxima</label>
          <input type="range" id="heat-max" min="0.1" max="3" step="0.1" value="0.8" oninput="updateHeatProps()">
        </div>
        <div class="sec">
          <h3>📋 Información</h3>
          <div class="card"><div class="lbl">Total de puntos graficados</div><div class="val" id="h-pts">—</div></div>
        </div>
      </div>
      
      <!-- TAB: TEMPORAL -->
      <div id="tab-temp" class="tab-content">
        <div class="sec">
          <h3>📅 Estacionalidad por Mes (IIEG)</h3>
          <div class="chart-container"><canvas id="chart-meses"></canvas></div>
        </div>
        <div class="sec">
          <h3>📆 Incidencia por Día de la Semana</h3>
          <div class="chart-container"><canvas id="chart-dias"></canvas></div>
        </div>
        <div class="sec">
          <h3>📋 Top 15 Delitos Registrados en GDL</h3>
          <div class="chart-container" style="height:220px;"><canvas id="chart-delitos-ranking"></canvas></div>
        </div>
      </div>
      
      <!-- TAB: COLONIAS -->
      <div id="tab-cols" class="tab-content">
        <div class="sec">
          <h3>🏘️ Colonias con Mayor Incidencia</h3>
          <p style="font-size:12px;color:#747d8c;margin-bottom:10px;">Haz clic en cualquier colonia para geolocalizarla automáticamente en el mapa.</p>
          <div id="colonias-lista" style="max-height: 520px; overflow-y: auto; padding-right: 2px;">
            <!-- Se llena dinámicamente -->
          </div>
        </div>
      </div>
      
      <!-- TAB: ENVIPE -->
      <div id="tab-env" class="tab-content">
        <div class="sec">
          <h3>🛡️ Percepción de Seguridad (GDL)</h3>
          <div class="donut-row">
            <div class="donut-item">
              <span>De Día</span>
              <div style="height: 100px;"><canvas id="chart-perc-dia"></canvas></div>
            </div>
            <div class="donut-item">
              <span>De Noche</span>
              <div style="height: 100px;"><canvas id="chart-perc-noche"></canvas></div>
            </div>
          </div>
        </div>
        <div class="sec">
          <h3>⚠️ Problemas más graves en la Colonia</h3>
          <div class="chart-container"><canvas id="chart-prob-principales"></canvas></div>
        </div>
        <div class="sec">
          <h3>🔴 Delitos Sufridos (Frecuencia ENVIPE)</h3>
          <div class="chart-container"><canvas id="chart-vic-delitos"></canvas></div>
        </div>
        <div class="sec">
          <h3>👥 Perfil de las Víctimas (Censo/SDEM)</h3>
          <label style="font-size: 11px; color:#747d8c; font-weight:600; display:block; margin-bottom: 4px;">Edades</label>
          <div style="height: 140px;"><canvas id="chart-vic-edades"></canvas></div>
          <label style="font-size: 11px; color:#747d8c; font-weight:600; display:block; margin: 12px 0 4px 0;">Nivel Educativo</label>
          <div style="height: 140px;"><canvas id="chart-edu"></canvas></div>
        </div>
      </div>
    </div>
  </div><!-- /sidebar -->

  <!-- MAPA -->
  <div id="map"></div>
</div>

<script>
const geojson   = {geojson_str};
const datos     = {json.dumps(datos_js)};
const paletas   = {json.dumps(paletas_js)};
const pt_data   = {json.dumps(heatmap_data)};
const temp_data = {json.dumps(temporal_json)};
const col_data  = {json.dumps(colonias_json)};
const env_data  = {json.dumps(envipe_json)};

let currentMode = 'simple';
let layer = null;
let heatLayer = null;
let colMarkers = [];

// Inicializar Mapa
const map = L.map("map", {{ zoomControl: false }}).setView([20.668, -103.347], 12);
L.control.zoom({{ position: 'bottomright' }}).addTo(map);

L.tileLayer("https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png", {{
  attribution: "© OpenStreetMap, © CARTO", maxZoom: 18
}}).addTo(map);

// ── UTILIDADES DE COLOR ──
function getColor(val, max, tipo) {{
  if(max===0||val===0) return "#151722";
  const t = Math.min(val/max, 1);
  if(tipo==="rojo"){{
    if(t<0.2)return"#2c1318";
    if(t<0.4)return"#501923";
    if(t<0.6)return"#801e2b";
    if(t<0.8)return"#bf273c";
    return"#ff4757";
  }}
  else if(tipo==="azul"){{
    if(t<0.2)return"#0c152a";
    if(t<0.4)return"#112349";
    if(t<0.6)return"#173770";
    if(t<0.8)return"#2154a6";
    return"#2f7bf6";
  }}
  else if(tipo==="naranja"){{
    if(t<0.2)return"#2b190c";
    if(t<0.4)return"#4d2911";
    if(t<0.6)return"#7f4017";
    if(t<0.8)return"#bf5d1d";
    return"#ffa502";
  }}
  else {{
    if(t<0.2)return"#0f2618";
    if(t<0.4)return"#164726";
    if(t<0.6)return"#1f6f36";
    if(t<0.8)return"#2fa44f";
    return"#2ed573";
  }}
}}

function getLegendGrad(tipo) {{
  if(tipo==="rojo") return "linear-gradient(to right, #151722, #801e2b, #ff4757)";
  if(tipo==="azul") return "linear-gradient(to right, #151722, #173770, #2f7bf6)";
  if(tipo==="naranja") return "linear-gradient(to right, #151722, #7f4017, #ffa502)";
  return "linear-gradient(to right, #151722, #1f6f36, #2ed573)";
}}

const BIVCOLS=[
  ["#2f7bf6","#b0a8cb","#8e44ad"],
  ["#a8c8e8","#b0b0b0","#c0392b"],
  ["#181a26","#a87050","#ff4757"]
];
function quantileBin(v,q33,q66){{ return v<=q33?2 : v<=q66?1 : 0; }}

// ── CONTROL DE PESTAÑAS (TABS) ──
function switchTab(mode) {{
  currentMode = mode;
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t=>t.classList.remove('active'));
  
  event.currentTarget.classList.add('active');
  document.getElementById('tab-'+mode).classList.add('active');
  
  // Resetear mapas
  if(heatLayer) map.removeLayer(heatLayer);
  if(layer) map.removeLayer(layer);
  clearColMarkers();
  
  if(mode==='simple') updateSimple();
  else if(mode==='biv') updateBiv();
  else if(mode==='heat') updateHeat();
  else if(mode==='temp') renderTemporalCharts();
  else if(mode==='cols') renderColonias();
  else if(mode==='env') renderEnvipeCharts();
}}

// ── TAB SIMPLE ──
function updateSimple() {{
  const capa = document.getElementById("sel-simple").value;
  const vals = datos[capa];
  const tipo = paletas[capa] || "verde";
  const allVals = Object.values(vals);
  const max = Math.max(...allVals);
  
  document.getElementById("s-tot").textContent = allVals.reduce((a,b)=>a+b,0).toLocaleString("es-MX");
  const maxE = Object.entries(vals).reduce((a,b)=>b[1]>a[1]?b:a,["",0]);
  document.getElementById("s-max").textContent = maxE[1].toLocaleString("es-MX");
  document.getElementById("s-max-ag").textContent = "AGEB "+maxE[0];
  
  document.getElementById("legend-bar").style.background = getLegendGrad(tipo);
  document.getElementById("l-mid").textContent = Math.round(max/2).toLocaleString("es-MX");
  document.getElementById("l-max").textContent = max.toLocaleString("es-MX");
  
  if(layer) map.removeLayer(layer);
  layer = L.geoJSON(geojson, {{
    style: f => {{
      const v = vals[f.properties.CVEGEO]||0;
      return {{fillColor: getColor(v,max,tipo), fillOpacity: v>0?0.82:0.25, color:"#0d0d1a", weight:0.5}};
    }},
    onEachFeature: (f,l) => {{
      const v = vals[f.properties.CVEGEO]||0;
      l.bindTooltip(`<b>AGEB ${{f.properties.CVEGEO}}</b><br>${{capa.replace("NEG: ","").replace("CEN: ","").replace("URB: ","")}}: <b>${{v.toLocaleString("es-MX")}}</b>`, {{sticky:true}});
    }}
  }}).addTo(map);
}}

// ── TAB BIVARIADO ──
function pearson(xs,ys){{
  const n=xs.length; if(n<3)return 0;
  const mx=xs.reduce((a,b)=>a+b,0)/n, my=ys.reduce((a,b)=>a+b,0)/n;
  let num=0,dx2=0,dy2=0;
  for(let i=0;i<n;i++){{ const dx=xs[i]-mx,dy=ys[i]-my; num+=dx*dy; dx2+=dx*dx; dy2+=dy*dy; }}
  return (dx2===0||dy2===0)?0:num/Math.sqrt(dx2*dy2);
}}

function drawScatter(xvals,yvals){{
  const canvas=document.getElementById("scatter-canvas");
  const ctx=canvas.getContext("2d");
  canvas.width=canvas.offsetWidth*window.devicePixelRatio||280;
  canvas.height=160*window.devicePixelRatio||160;
  ctx.scale(window.devicePixelRatio||1,window.devicePixelRatio||1);
  const W=canvas.offsetWidth||280, H=160, pad=25;
  ctx.fillStyle="#0a0b10"; ctx.fillRect(0,0,W,H);
  if(xvals.length===0) return;
  const xmin=Math.min(...xvals), xmax=Math.max(...xvals)||1;
  const ymin=Math.min(...yvals), ymax=Math.max(...yvals)||1;
  
  ctx.strokeStyle="#1f2335"; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad,H-pad); ctx.lineTo(W-pad,H-pad); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pad,pad); ctx.lineTo(pad,H-pad); ctx.stroke();
  
  for(let i=0;i<xvals.length;i++){{
    const px=pad+(xvals[i]-xmin)/(xmax-xmin+1e-10)*(W-2*pad);
    const py=(H-pad)-(yvals[i]-ymin)/(ymax-ymin+1e-10)*(H-2*pad);
    ctx.beginPath(); ctx.arc(px,py,3,0,Math.PI*2);
    ctx.fillStyle="rgba(255, 71, 87, 0.6)"; ctx.fill();
  }}
  // Línea de regresión
  const n=xvals.length;
  const mx=xvals.reduce((a,b)=>a+b,0)/n, my=yvals.reduce((a,b)=>a+b,0)/n;
  let num=0,den=0;
  for(let i=0;i<n;i++){{num+=(xvals[i]-mx)*(yvals[i]-my);den+=(xvals[i]-mx)**2;}}
  const slope=den===0?0:num/den;
  const intercept=my-slope*mx;
  const x1=xmin, y1=slope*x1+intercept;
  const x2=xmax, y2=slope*x2+intercept;
  const px1=pad+(x1-xmin)/(xmax-xmin+1e-10)*(W-2*pad);
  const py1=(H-pad)-(y1-ymin)/(ymax-ymin+1e-10)*(H-2*pad);
  const px2=pad+(x2-xmin)/(xmax-xmin+1e-10)*(W-2*pad);
  const py2=(H-pad)-(y2-ymin)/(ymax-ymin+1e-10)*(H-2*pad);
  ctx.beginPath(); ctx.moveTo(px1,py1); ctx.lineTo(px2,py2);
  ctx.strokeStyle="#2f7bf6"; ctx.lineWidth=2; ctx.stroke();
}}

function updateBiv() {{
  const cx=document.getElementById("sel-bx").value;
  const cy=document.getElementById("sel-by").value;
  const vx=datos[cx]||{{}}, vy=datos[cy]||{{}};
  const cvegeos=Object.keys(vx).filter(k=>k in vy);
  const xvals=cvegeos.map(k=>vx[k]||0), yvals=cvegeos.map(k=>vy[k]||0);
  
  const q33x=[...xvals].sort((a,b)=>a-b)[Math.floor(xvals.length*.333)];
  const q66x=[...xvals].sort((a,b)=>a-b)[Math.floor(xvals.length*.666)];
  const q33y=[...yvals].sort((a,b)=>a-b)[Math.floor(yvals.length*.333)];
  const q66y=[...yvals].sort((a,b)=>a-b)[Math.floor(yvals.length*.666)];
  
  const r=pearson(xvals,yvals);
  const absR=Math.abs(r);
  const tStat = r*Math.sqrt((xvals.length-2)/(1-r*r+1e-10));
  const pVal = 2*(1-Math.min(0.9999, Math.abs(tStat)/(Math.abs(tStat)+Math.sqrt(xvals.length-2))));
  
  const rb=document.getElementById("r-badge");
  rb.textContent=r.toFixed(3);
  let col="#747d8c", interp="Sin correlación aparente (r ≈ 0)";
  if(absR>=0.1 && absR<0.3){{col="#ffa502"; interp=r>0?"Correlación positiva débil":"Correlación negativa débil";}}
  else if(absR>=0.3 && absR<0.5){{col="#ff6b81"; interp=r>0?"Correlación positiva moderada":"Correlación negativa moderada";}}
  else if(absR>=0.5 && absR<0.7){{col="#ff4757"; interp=r>0?"Correlación positiva fuerte":"Correlación negativa fuerte";}}
  else if(absR>=0.7){{col="#ff2e44"; interp=r>0?"Correlación positiva muy fuerte":"Correlación negativa muy fuerte";}}
  
  rb.style.background=col+"18"; rb.style.color=col; rb.style.border="1px solid "+col;
  document.getElementById("r-interp").textContent=interp;
  
  document.getElementById("st-n").textContent=xvals.length.toLocaleString("es-MX");
  document.getElementById("st-p").textContent=pVal<0.001?"< 0.001":pVal.toFixed(4);
  
  document.getElementById("x-lbl").textContent="→ "+cx.replace("NEG: ","").replace("CEN: ","").replace("URB: ","");
  document.getElementById("y-lbl").textContent=cy.replace("NEG: ","").replace("CEN: ","").replace("URB: ","")+" ↑";
  drawScatter(xvals,yvals);
  
  if(layer) map.removeLayer(layer);
  layer=L.geoJSON(geojson,{{
    style:f=>{{
      const k=f.properties.CVEGEO;
      const row=quantileBin(vy[k]||0,q33y,q66y);
      const col=quantileBin(vx[k]||0,q33x,q66x)===0?2:quantileBin(vx[k]||0,q33x,q66x)===1?1:0;
      return {{fillColor:BIVCOLS[row][col], fillOpacity:0.85, color:"#08090c", weight:0.5}};
    }},
    onEachFeature:(f,l)=>{{
      const k=f.properties.CVEGEO;
      l.bindTooltip(`<b>AGEB ${{k}}</b><br>X (Delito): <b>${{(vx[k]||0).toLocaleString("es-MX")}}</b><br>Y (Factor): <b>${{(vy[k]||0).toLocaleString("es-MX")}}</b>`,{{sticky:true}});
    }}
  }}).addTo(map);
}}

// ── TAB HEATMAP ──
function updateHeat() {{
  const capa = document.getElementById("sel-heat").value;
  const pts = pt_data[capa] || [];
  document.getElementById("h-pts").textContent = pts.length.toLocaleString("es-MX");
  
  if(layer) map.removeLayer(layer);
  if(heatLayer) map.removeLayer(heatLayer);
  
  layer = L.geoJSON(geojson, {{
    style: {{fillColor:"#ffffff", fillOpacity:0.02, color:"#1f2335", weight:0.5}}
  }}).addTo(map);
  
  const rad = parseInt(document.getElementById("heat-radius").value);
  const mx = parseFloat(document.getElementById("heat-max").value);
  
  heatLayer = L.heatLayer(pts, {{
    radius: rad,
    blur: rad + 5,
    maxZoom: 14,
    max: mx,
    gradient: {{ 0.2: '#2f7bf6', 0.5: '#2ed573', 0.8: '#ffa502', 1.0: '#ff4757' }}
  }}).addTo(map);
}}

function updateHeatProps() {{
  if(currentMode==='heat') updateHeat();
}}

// ── TAB COLONIAS (LISTA Y FLYTO) ──
function renderColonias() {{
  const el = document.getElementById("colonias-lista");
  el.innerHTML = "";
  clearColMarkers();
  
  layer = L.geoJSON(geojson, {{
    style: {{fillColor:"#ffffff", fillOpacity:0.02, color:"#1f2335", weight:0.5}}
  }}).addTo(map);
  
  col_data.forEach((c, idx) => {{
    // Crear item en la lista
    const item = document.createElement("div");
    item.className = "colonia-item";
    item.innerHTML = `<span class="nombre">${{idx+1}}. ${{c.colonia.toUpperCase()}}</span><span class="cifra">${{c.total.toLocaleString("es-MX")}}</span>`;
    
    // Zoom y marcador al hacer clic
    item.onclick = () => {{
      map.flyTo([c.lat, c.lng], 15);
      
      // Limpiar marcadores anteriores de selección
      clearColMarkers();
      
      const m = L.marker([c.lat, c.lng]).addTo(map)
        .bindPopup(`<b style="font-size:13px;">${{c.colonia.toUpperCase()}}</b><br>Delitos Totales 2023: <b>${{c.total.toLocaleString("es-MX")}}</b>`)
        .openPopup();
      colMarkers.push(m);
    }};
    el.appendChild(item);
  }});
}}

function clearColMarkers() {{
  colMarkers.forEach(m => map.removeLayer(m));
  colMarkers = [];
}}

// ── TABS DE CHARTS (TEMPORAL & ENVIPE) ──
let charts = {{}};

function destroyChart(name) {{
  if (charts[name]) {{
    charts[name].destroy();
    delete charts[name];
  }}
}}

const chartConfigBase = (type, labels, data, colors, horizontal=false) => ({{
  type: type,
  data: {{
    labels: labels,
    datasets: [{{
      data: data,
      backgroundColor: colors,
      borderColor: 'transparent',
      borderWidth: 0,
      borderRadius: 4
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    indexAxis: horizontal ? 'y' : 'x',
    plugins: {{
      legend: {{ display: false }}
    }},
    scales: {{
      x: {{
        grid: {{ color: '#1f2335' }},
        ticks: {{ color: '#747d8c', font: {{ family: 'Outfit', size: 10 }} }}
      }},
      y: {{
        grid: {{ color: '#1f2335' }},
        ticks: {{ color: '#747d8c', font: {{ family: 'Outfit', size: 10 }} }}
      }}
    }}
  }}
}});

function renderTemporalCharts() {{
  // Fondo de AGEB sutil
  if(layer) map.removeLayer(layer);
  layer = L.geoJSON(geojson, {{
    style: {{fillColor:"#ffffff", fillOpacity:0.02, color:"#1f2335", weight:0.5}}
  }}).addTo(map);
  
  destroyChart('meses');
  destroyChart('dias');
  destroyChart('delitos');
  
  charts['meses'] = new Chart(document.getElementById('chart-meses'), chartConfigBase('bar', temp_data.labels_meses, temp_data.valores_meses, '#2f7bf6'));
  charts['dias'] = new Chart(document.getElementById('chart-dias'), chartConfigBase('bar', temp_data.labels_dias, temp_data.valores_dias, '#ffa502'));
  
  const delConfig = chartConfigBase('bar', temp_data.labels_delitos, temp_data.valores_delitos, '#ff4757', true);
  delConfig.options.scales.y.ticks.font.size = 9;
  charts['delitos'] = new Chart(document.getElementById('chart-delitos-ranking'), delConfig);
}}

function renderEnvipeCharts() {{
  if(layer) map.removeLayer(layer);
  layer = L.geoJSON(geojson, {{
    style: {{fillColor:"#ffffff", fillOpacity:0.02, color:"#1f2335", weight:0.5}}
  }}).addTo(map);
  
  destroyChart('percDia');
  destroyChart('percNoche');
  destroyChart('prob');
  destroyChart('vicDel');
  destroyChart('vicEdades');
  destroyChart('edu');
  
  const donutOptions = (colors) => ({{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{
        position: 'bottom',
        labels: {{ color: '#747d8c', boxWidth: 10, font: {{ family: 'Outfit', size: 9 }} }}
      }}
    }}
  }});
  
  charts['percDia'] = new Chart(document.getElementById('chart-perc-dia'), {{
    type: 'doughnut',
    data: {{
      labels: env_data.perc_dia_labels,
      datasets: [{{ data: env_data.perc_dia_values, backgroundColor: ['#2ed573', '#ff4757', '#ffa502', '#747d8c'], borderWidth: 0 }}]
    }},
    options: donutOptions()
  }});
  
  charts['percNoche'] = new Chart(document.getElementById('chart-perc-noche'), {{
    type: 'doughnut',
    data: {{
      labels: env_data.perc_noche_labels,
      datasets: [{{ data: env_data.perc_noche_values, backgroundColor: ['#ff4757', '#2ed573'], borderWidth: 0 }}]
    }},
    options: donutOptions()
  }});
  
  charts['prob'] = new Chart(document.getElementById('chart-prob-principales'), chartConfigBase('bar', env_data.prob_labels, env_data.prob_values, '#ffa502', true));
  charts['vicDel'] = new Chart(document.getElementById('chart-vic-delitos'), chartConfigBase('bar', env_data.vic_del_labels, env_data.vic_del_values, '#ff6b81', true));
  charts['vicEdades'] = new Chart(document.getElementById('chart-vic-edades'), chartConfigBase('bar', env_data.edad_labels, env_data.edad_values, '#2f7bf6'));
  
  const eduConfig = chartConfigBase('bar', env_data.edu_labels, env_data.edu_values, '#2ed573');
  eduConfig.options.scales.x.ticks.font.size = 8;
  charts['edu'] = new Chart(document.getElementById('chart-edu'), eduConfig);
}}

// Generar Leyenda Bivariada 3x3
const lgEl=document.getElementById("biv-legend");
for(let r=0;r<3;r++) for(let c=0;c<3;c++){{
  const d=document.createElement("div"); d.className="biv-cell"; d.style.background=BIVCOLS[r][c];
  lgEl.appendChild(d);
}}

// Cargar estado inicial
switchTab('simple');
</script>
</body>
</html>"""

salida = "09_dashboard_completo.html"
with open(salida, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\n✅ Dashboard Maestro Final generado con éxito en:\n   {os.path.abspath(salida)}")
print("¡Puedes abrirlo directamente en tu navegador web para explorar las 6 pestañas integradas!")
