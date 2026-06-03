"""
SCRIPT 08 — Mapa de Correlación Bivariada Interactiva por AGEB
===============================================================
Permite seleccionar una categoría de delito y cruzarla con
un negocio o dato sociodemográfico del Censo para visualizar
la correlación por AGEB mediante un mapa bivariado + scatter plot.
"""
import pandas as pd
import geopandas as gpd
import json, os

# ── 1. AGEBS ─────────────────────────────────────────────────────────
print("Cargando AGEBs...")
agebs = gpd.read_file("guadalajara_AGEB/2025_14039_A07052026_1549.shp")
agebs["CVEGEO"] = agebs["CVEGEO"].astype(str).str.strip()
agebs_gdl = agebs[agebs["CVEGEO"].str[2:5] == "039"].copy()
agebs_gdl = agebs_gdl.to_crs(epsg=4326) if agebs_gdl.crs else agebs_gdl.set_crs(epsg=4326)
print(f"  → {len(agebs_gdl)} AGEBs")

# ── 2. DELITOS ────────────────────────────────────────────────────────
print("Procesando delitos IIEG...")
CATS_DELITO = {
    'Homicidios y Feminicidios': ['homicidio doloso', 'feminicidio'],
    'Lesiones Dolosas':          ['lesiones dolosas'],
    'Robo a Persona':            ['robo a persona', 'robo a cuentahabientes'],
    'Robo a Negocio':            ['robo a negocio', 'robo a bancos'],
}
inc = pd.read_csv("iieg_2023.csv").dropna(subset=["x","y","delito"])
inc["delito"] = inc["delito"].str.strip().str.lower()
def clas_del(d):
    for cat, lst in CATS_DELITO.items():
        if d in lst: return cat
    return None
inc["categoria"] = inc["delito"].apply(clas_del)
inc = inc.dropna(subset=["categoria"])
gdf_inc = gpd.GeoDataFrame(inc, geometry=gpd.points_from_xy(inc.x, inc.y), crs="EPSG:4326").to_crs(agebs_gdl.crs)
j_del = gpd.sjoin(gdf_inc, agebs_gdl[["CVEGEO","geometry"]], how="left", predicate="within")
t_del = j_del.groupby(["CVEGEO","categoria"]).size().unstack(fill_value=0).reset_index()

# ── 3. DENUE ──────────────────────────────────────────────────────────
print("Procesando DENUE...")
denue = pd.read_csv("gdl_denue.csv").dropna(subset=["longitud","latitud","categoria"])
denue = denue[denue["categoria"] != "Otro"]
gdf_dn = gpd.GeoDataFrame(denue, geometry=gpd.points_from_xy(denue.longitud, denue.latitud), crs="EPSG:4326").to_crs(agebs_gdl.crs)
j_dn = gpd.sjoin(gdf_dn, agebs_gdl[["CVEGEO","geometry"]], how="left", predicate="within")
t_dn = j_dn.groupby(["CVEGEO","categoria"]).size().unstack(fill_value=0).reset_index()
t_dn.columns = ["CVEGEO"] + [f"NEG: {c}" for c in t_dn.columns if c != "CVEGEO"]

# ── 4. CENSO ──────────────────────────────────────────────────────────
print("Procesando Censo RESAGEBURB...")
resag = pd.read_csv("RESAGEBURB_14CSV20.csv", encoding="latin1")
resag_gdl = resag[(resag["MUN"]==39) & (resag["MZA"]==0)].copy()
resag_gdl["CVEGEO"] = (
    resag_gdl["ENTIDAD"].astype(str).str.zfill(2) +
    resag_gdl["MUN"].astype(str).str.zfill(3) +
    resag_gdl["LOC"].astype(str).str.zfill(4) +
    resag_gdl["AGEB"].astype(str).str.strip()
)
COLS_CEN = {
    "POBTOT":    "CEN: Población Total",
    "GRAPROES":  "CEN: Escolaridad Promedio (años)",
    "PEA":       "CEN: Población Económ. Activa",
    "POCUPADA":  "CEN: Población Ocupada",
    "PDESOCUP":  "CEN: Población Desocupada",
    "PSINDER":   "CEN: Sin Derechohabiencia",
    "P15YM_AN":  "CEN: Analfabetismo +15 años",
    "P18YM_PB":  "CEN: Con Educación Básica",
    "VPH_INTER": "CEN: Viviendas con Internet",
    "VPH_AUTOM": "CEN: Viviendas con Automóvil",
    "VPH_PC":    "CEN: Viviendas con Computadora",
    "P60YMAS":   "CEN: Adultos Mayores +60 años",
}
cols_sel = ["CVEGEO"] + [c for c in COLS_CEN if c in resag_gdl.columns]
t_cen = resag_gdl[cols_sel].rename(columns=COLS_CEN).copy()
for col in t_cen.columns:
    if col != "CVEGEO":
        t_cen[col] = pd.to_numeric(t_cen[col], errors="coerce").fillna(0)

# ── 4.5 ENTORNO URBANO (FRENTES) ──────────────────────────────────────
print("Procesando Entorno Urbano (inv_frentes.csv)...")
frentes = pd.read_csv("inv_frentes.csv", dtype={"CVEGEO": str, "CVE_MUN": str})
# Filtrar solo Guadalajara (039)
frentes_gdl = frentes[frentes["CVE_MUN"].str.zfill(3) == "039"].copy()

if not frentes_gdl.empty:
    # Extraer el CVEGEO a nivel AGEB (los primeros 13 caracteres de los 16 del frente)
    frentes_gdl["CVEGEO_AGEB"] = frentes_gdl["CVEGEO"].str[:13]
    
    # Crear variables indicadoras (1 si dispone, 0 si no)
    frentes_gdl["alumbrado"] = (frentes_gdl["ALUMPUB_D"] == "Dispone").astype(int)
    frentes_gdl["banqueta"] = (frentes_gdl["BANQUETA_D"] == "Dispone").astype(int)
    frentes_gdl["arboles"] = (frentes_gdl["ARBOLES_D"] == "Dispone").astype(int)
    
    # Agrupar por AGEB para sacar el porcentaje promedio (de 0 a 100)
    t_frentes = frentes_gdl.groupby("CVEGEO_AGEB")[["alumbrado", "banqueta", "arboles"]].mean() * 100
    t_frentes = t_frentes.reset_index().rename(columns={
        "CVEGEO_AGEB": "CVEGEO",
        "alumbrado": "URB: % Frentes con Alumbrado",
        "banqueta":  "URB: % Frentes con Banqueta",
        "arboles":   "URB: % Frentes con Árboles"
    })
else:
    t_frentes = pd.DataFrame(columns=["CVEGEO", "URB: % Frentes con Alumbrado", "URB: % Frentes con Banqueta", "URB: % Frentes con Árboles"])

# ── 5. MERGE ──────────────────────────────────────────────────────────
print("Uniendo capas...")
af = agebs_gdl.copy()
af = af.merge(t_del, on="CVEGEO", how="left")
af = af.merge(t_dn,  on="CVEGEO", how="left")
af = af.merge(t_cen, on="CVEGEO", how="left")
af = af.merge(t_frentes, on="CVEGEO", how="left")
af = af.fillna(0)

capas_delitos  = list(CATS_DELITO.keys())
capas_negocios = [c for c in af.columns if c.startswith("NEG: ")]
capas_censo    = [c for c in af.columns if c.startswith("CEN: ")]
capas_urbano   = [c for c in af.columns if c.startswith("URB: ")]
capas_y        = capas_negocios + capas_censo + capas_urbano

# ── 6. EXPORTAR DATOS ─────────────────────────────────────────────────
print("Exportando GeoJSON y datos...")
geojson_str = af.to_json()
todas = capas_delitos + capas_y
datos_js = {c: af.set_index("CVEGEO")[c].to_dict() for c in todas}

# Construir opciones HTML del selector Y
def build_optgroup(label, lista):
    opts = f'<optgroup label="{label}">'
    for c in lista:
        short = c.replace("NEG: ","").replace("CEN: ","")
        opts += f'<option value="{c}">{short}</option>'
    return opts + "</optgroup>"

opts_y = build_optgroup("🏪 Negocios (DENUE)", capas_negocios) + \
         build_optgroup("📊 Sociodemográfico (Censo 2020)", capas_censo) + \
         build_optgroup("🏙️ Entorno Urbano (Frentes)", capas_urbano)
opts_x = "".join(f'<option value="{c}">{c}</option>' for c in capas_delitos)

# ── 7. GENERAR HTML ───────────────────────────────────────────────────
print("Generando HTML...")
html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Correlación Bivariada — Guadalajara 2023</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:'Inter',sans-serif;background:#0f1117;color:#e8eaf0;height:100vh;display:flex;flex-direction:column}}
    #header{{background:linear-gradient(135deg,#1a1d2e,#12151f);border-bottom:1px solid #2a2d3d;padding:12px 24px;display:flex;align-items:center;gap:12px;flex-shrink:0;z-index:2000}}
    #header h1{{font-size:15px;font-weight:700;background:linear-gradient(90deg,#e74c3c,#9b59b6);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
    #header .sub{{font-size:11px;color:#6c7293;margin-left:auto}}
    #main{{display:flex;flex:1;overflow:hidden}}
    #sidebar{{width:320px;background:#1a1d2e;border-right:1px solid #2a2d3d;display:flex;flex-direction:column;overflow-y:auto;flex-shrink:0}}
    .sec{{padding:14px;border-bottom:1px solid #2a2d3d}}
    .sec h3{{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#6c7293;margin-bottom:8px}}
    select{{width:100%;background:#0f1117;color:#e8eaf0;border:1px solid #2a2d3d;border-radius:7px;padding:8px 10px;font-size:12px;font-family:'Inter',sans-serif;cursor:pointer;outline:none;margin-bottom:6px}}
    select:hover{{border-color:#4a4d6d}}
    select optgroup{{color:#6c7293;font-weight:700}}
    select option{{color:#e8eaf0;background:#1a1d2e}}
    .btn{{width:100%;padding:9px;background:linear-gradient(135deg,#e74c3c,#9b59b6);color:#fff;border:none;border-radius:7px;font-size:13px;font-weight:600;cursor:pointer;font-family:'Inter',sans-serif;transition:opacity .2s}}
    .btn:hover{{opacity:.85}}
    .card{{background:#0f1117;border:1px solid #2a2d3d;border-radius:8px;padding:11px;margin-bottom:7px}}
    .card .lbl{{font-size:10px;color:#6c7293;margin-bottom:3px}}
    .card .val{{font-size:18px;font-weight:700}}
    .card .sub{{font-size:10px;color:#6c7293;margin-top:2px}}
    #r-badge{{font-size:28px;font-weight:700;text-align:center;padding:8px;border-radius:8px;margin-bottom:4px}}
    #r-interp{{font-size:11px;color:#a0a3b1;text-align:center;line-height:1.5}}
    #scatter-canvas{{width:100%;height:180px;border-radius:8px;background:#0f1117;border:1px solid #2a2d3d;display:block}}
    /* Leyenda bivariada 3x3 */
    #biv-legend{{display:grid;grid-template-columns:repeat(3,1fr);gap:2px;width:90px;height:90px;margin:8px auto}}
    .biv-cell{{border-radius:2px}}
    .biv-labels{{display:flex;justify-content:space-between;font-size:9px;color:#6c7293;margin-top:2px}}
    #map{{flex:1;z-index:1}}
    .leaflet-tooltip{{background:#1a1d2e!important;border:1px solid #2a2d3d!important;color:#e8eaf0!important;border-radius:7px!important;padding:7px 11px!important;font-family:'Inter',sans-serif!important;font-size:12px!important;box-shadow:0 4px 15px rgba(0,0,0,.5)!important}}
  </style>
</head>
<body>
<div id="header">
  <div style="width:9px;height:9px;border-radius:50%;background:#e74c3c;box-shadow:0 0 7px #e74c3c"></div>
  <h1>Correlación Bivariada por AGEB — Guadalajara 2023</h1>
  <span class="sub">IIEG · DENUE · Censo 2020 · Entorno Urbano</span>
</div>
<div id="main">
  <div id="sidebar">
    <div class="sec">
      <h3>🔴 Variable X — Delito</h3>
      <select id="sel-x">{opts_x}</select>
      <h3 style="margin-top:10px">🔵 Variable Y — Factor</h3>
      <select id="sel-y">{opts_y}</select>
      <button class="btn" onclick="analizar()">Analizar correlación ▶</button>
    </div>
    <div class="sec">
      <h3>Coeficiente de Correlación (Pearson r)</h3>
      <div id="r-badge">—</div>
      <div id="r-interp">Selecciona las variables y presiona Analizar</div>
    </div>
    <div class="sec">
      <h3>Scatter Plot por AGEB</h3>
      <canvas id="scatter-canvas"></canvas>
      <div id="scatter-labels" style="display:flex;justify-content:space-between;font-size:9px;color:#6c7293;margin-top:3px">
        <span id="x-label-sc">X</span><span id="y-label-sc">Y</span>
      </div>
    </div>
    <div class="sec">
      <h3>Leyenda Bivariada</h3>
      <div style="display:flex;gap:12px;align-items:flex-start">
        <div>
          <div id="biv-legend"></div>
          <div class="biv-labels"><span>Bajo X</span><span>Alto X</span></div>
          <div style="font-size:9px;color:#6c7293;text-align:right;margin-top:1px">↑ Alto Y</div>
        </div>
        <div style="font-size:10px;color:#6c7293;line-height:1.7;margin-top:4px">
          <b style="color:#c0392b">■</b> Alto delito + bajo factor<br>
          <b style="color:#2980b9">■</b> Bajo delito + alto factor<br>
          <b style="color:#8e44ad">■</b> Alto en ambos<br>
          <b style="color:#2d2d3a">■</b> Bajo en ambos
        </div>
      </div>
    </div>
    <div class="sec">
      <h3>Estadísticas</h3>
      <div class="card"><div class="lbl">AGEBs analizadas</div><div class="val" id="st-n">—</div></div>
      <div class="card"><div class="lbl">p-valor (aprox.)</div><div class="val" id="st-p">—</div><div class="sub">Si p &lt; 0.05 la correlación es significativa</div></div>
    </div>
  </div>
  <div id="map"></div>
</div>
<script>
const geojson = {geojson_str};
const datos   = {json.dumps(datos_js)};

const map = L.map("map").setView([20.668,-103.347],12);
L.tileLayer("https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png",{{attribution:"© CARTO"}}).addTo(map);
let layer=null;

// Paleta bivariada 3x3 (fila=Y, col=X)
const BIVCOLS=[
  ["#d3d3e8","#b0a8cb","#8e44ad"], // Y alto
  ["#a8c8e8","#b0b0b0","#c0392b"], // Y medio
  ["#2980b9","#a87050","#e74c3c"]  // Y bajo
];

function buildLegend(){{
  const el=document.getElementById("biv-legend");
  el.innerHTML="";
  // Renderizar de arriba (Y alto) a abajo (Y bajo)
  for(let r=0;r<3;r++) for(let c=0;c<3;c++){{
    const d=document.createElement("div");
    d.className="biv-cell";
    d.style.background=BIVCOLS[r][c];
    el.appendChild(d);
  }}
}}
buildLegend();

function quantileBin(val,q33,q66){{
  if(val<=q33) return 2; // bajo → fila 2 (abajo en grid)
  if(val<=q66) return 1; // medio
  return 0;              // alto → fila 0 (arriba en grid)
}}

function pearson(xs,ys){{
  const n=xs.length;
  if(n<3) return 0;
  const mx=xs.reduce((a,b)=>a+b,0)/n;
  const my=ys.reduce((a,b)=>a+b,0)/n;
  let num=0,dx2=0,dy2=0;
  for(let i=0;i<n;i++){{
    const dx=xs[i]-mx, dy=ys[i]-my;
    num+=dx*dy; dx2+=dx*dx; dy2+=dy*dy;
  }}
  return (dx2===0||dy2===0)?0:num/Math.sqrt(dx2*dy2);
}}

function pValApprox(r,n){{
  if(n<3) return 1;
  const t=r*Math.sqrt((n-2)/(1-r*r+1e-10));
  // Aproximación simple de dos colas (distribución t)
  const p=2*(1-Math.min(0.9999,Math.abs(t)/(Math.abs(t)+Math.sqrt(n-2))));
  return p;
}}

function drawScatter(xvals,yvals,xname,yname){{
  const canvas=document.getElementById("scatter-canvas");
  const ctx=canvas.getContext("2d");
  canvas.width=canvas.offsetWidth*window.devicePixelRatio||280;
  canvas.height=180*window.devicePixelRatio||180;
  ctx.scale(window.devicePixelRatio||1,window.devicePixelRatio||1);
  const W=canvas.offsetWidth||280, H=180;
  ctx.fillStyle="#0f1117"; ctx.fillRect(0,0,W,H);
  if(xvals.length===0) return;
  const pad=20;
  const xmin=Math.min(...xvals),xmax=Math.max(...xvals)||1;
  const ymin=Math.min(...yvals),ymax=Math.max(...yvals)||1;
  // Eje
  ctx.strokeStyle="#2a2d3d"; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad,H-pad); ctx.lineTo(W-pad,H-pad); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pad,pad); ctx.lineTo(pad,H-pad); ctx.stroke();
  // Puntos
  for(let i=0;i<xvals.length;i++){{
    const px=pad+(xvals[i]-xmin)/(xmax-xmin+1e-10)*(W-2*pad);
    const py=(H-pad)-(yvals[i]-ymin)/(ymax-ymin+1e-10)*(H-2*pad);
    ctx.beginPath(); ctx.arc(px,py,2.5,0,Math.PI*2);
    ctx.fillStyle="rgba(155,89,182,0.7)"; ctx.fill();
  }}
  // Línea de tendencia
  const n=xvals.length;
  const mx=xvals.reduce((a,b)=>a+b,0)/n;
  const my=yvals.reduce((a,b)=>a+b,0)/n;
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
  ctx.strokeStyle="#e74c3c"; ctx.lineWidth=1.5; ctx.stroke();
}}

function analizar(){{
  const cx=document.getElementById("sel-x").value;
  const cy=document.getElementById("sel-y").value;
  const vx=datos[cx]||{{}};
  const vy=datos[cy]||{{}};
  const cvegeos=Object.keys(vx).filter(k=>k in vy);
  const xvals=cvegeos.map(k=>vx[k]||0);
  const yvals=cvegeos.map(k=>vy[k]||0);

  // Cuantiles para clasificación bivariada
  const xsort=[...xvals].sort((a,b)=>a-b);
  const ysort=[...yvals].sort((a,b)=>a-b);
  const q33x=xsort[Math.floor(xsort.length*.333)];
  const q66x=xsort[Math.floor(xsort.length*.666)];
  const q33y=ysort[Math.floor(ysort.length*.333)];
  const q66y=ysort[Math.floor(ysort.length*.666)];

  // Calcular Pearson
  const r=pearson(xvals,yvals);
  const p=pValApprox(r,xvals.length);

  // Mostrar r
  const rb=document.getElementById("r-badge");
  rb.textContent=r.toFixed(3);
  const absR=Math.abs(r);
  let col="#66bb6a",interp="";
  if(absR<0.1){{col="#6c7293";interp="Sin correlación aparente (r ≈ 0)"}}
  else if(absR<0.3){{col="#f39c12";interp=r>0?"Correlación positiva débil":"Correlación negativa débil"}}
  else if(absR<0.5){{col="#e67e22";interp=r>0?"Correlación positiva moderada":"Correlación negativa moderada"}}
  else if(absR<0.7){{col="#e74c3c";interp=r>0?"Correlación positiva fuerte":"Correlación negativa fuerte"}}
  else{{col="#c0392b";interp=r>0?"Correlación positiva muy fuerte":"Correlación negativa muy fuerte"}}
  rb.style.background=col+"22"; rb.style.color=col; rb.style.border="1px solid "+col;
  document.getElementById("r-interp").textContent=interp;
  document.getElementById("st-n").textContent=xvals.length.toLocaleString("es-MX");
  document.getElementById("st-p").textContent=p<0.001?"< 0.001":p.toFixed(4);
  document.getElementById("x-label-sc").textContent="→ "+cx;
  document.getElementById("y-label-sc").textContent=cy+" ↑";

  // Scatter
  drawScatter(xvals,yvals,cx,cy);

  // Mapa bivariado
  if(layer) map.removeLayer(layer);
  const xnameShort=cx.replace("NEG: ","").replace("CEN: ","").replace("URB: ","");
  const ynameShort=cy.replace("NEG: ","").replace("CEN: ","").replace("URB: ","");
  layer=L.geoJSON(geojson,{{
    style:function(f){{
      const k=f.properties.CVEGEO;
      const vxv=vx[k]||0, vyv=vy[k]||0;
      const row=quantileBin(vyv,q33y,q66y);
      const col=quantileBin(vxv,q33x,q66x)==0?2:quantileBin(vxv,q33x,q66x)==1?1:0;
      const fillColor=BIVCOLS[row][col];
      return{{fillColor,fillOpacity:0.82,color:"#000",weight:0.4}};
    }},
    onEachFeature:function(f,lyr){{
      const k=f.properties.CVEGEO;
      const vxv=vx[k]||0, vyv=vy[k]||0;
      lyr.bindTooltip(
        `<b>AGEB ${{k}}</b><br>`+
        `<span style="color:#e74c3c">${{xnameShort}}:</span> <b>${{vxv.toLocaleString("es-MX")}}</b><br>`+
        `<span style="color:#2980b9">${{ynameShort}}:</span> <b>${{vyv.toLocaleString("es-MX")}}</b>`,
        {{sticky:true,opacity:1}}
      );
    }}
  }}).addTo(map);
}}

// Ejecutar al cargar
analizar();
</script>
</body>
</html>"""

salida = "mapa_correlacion_bivariada.html"
with open(salida, "w", encoding="utf-8") as f:
    f.write(html)

ruta = os.path.abspath(salida)
print(f"\n✅ Dashboard guardado en:\n   {ruta}")
print(f"\nCapas X (delitos): {capas_delitos}")
print(f"Capas Y (factores): {len(capas_y)} disponibles")
