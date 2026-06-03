"""
API Servidora de Analítica Urbana con Gemini
===========================================
Este script Flask sirve el backend para el Dashboard React Premium.
Carga y procesa todos los conjuntos de datos en memoria para brindar respuestas instantáneas:
  - Cruce de AGEBs (IIEG, DENUE, Censo, Entorno Urbano)
  - Agrupaciones de delincuencia mensual y por día
  - Listas de colonias con geolocalización
  - Cruce de encuestas ENVIPE (incluyendo perfil de víctimas cruzado por edad y sexo)
  - Consultas inteligentes al Analista de IA con la API de Gemini (basadas en datos urbanos locales)
"""

import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial.distance import cdist
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__, static_folder=".")
CORS(app)

# Cache de datos procesados
cache_data = {}

def cargar_y_procesar_todo():
    print("🚀 Cargando y preprocesando datos geoespaciales y estadísticos...")
    
    # 1. Cargar AGEBs
    agebs = gpd.read_file("datos/guadalajara_AGEB/2025_14039_A07052026_1549.shp")
    agebs["CVEGEO"] = agebs["CVEGEO"].astype(str).str.strip()
    agebs_gdl = agebs[agebs["CVEGEO"].str[2:5] == "039"].copy()
    agebs_gdl = agebs_gdl.to_crs(epsg=4326) if agebs_gdl.crs else agebs_gdl.set_crs(epsg=4326)
    
    # Guardar centroides de AGEBs para cálculos de cercanía
    agebs_gdl["centroid_y"] = agebs_gdl.geometry.centroid.y
    agebs_gdl["centroid_x"] = agebs_gdl.geometry.centroid.x
    
    # 2. Delitos IIEG
    grupos_delito = {
        'Homicidios y Feminicidios (Grupo)': ['homicidio doloso', 'feminicidio'],
        'Lesiones Dolosas (Grupo)':          ['lesiones dolosas'],
        'Robo a Persona (Grupo)':            ['robo a persona', 'robo a cuentahabientes'],
        'Robo a Negocio (Grupo)':            ['robo a negocio', 'robo a bancos'],
    }
    
    individuales_delito = {
        'Violencia Familiar': 'violencia familiar',
        'Robo a Persona': 'robo a persona',
        'Robo a Vehículos Particulares': 'robo a vehiculos particulares',
        'Robo a Negocio': 'robo a negocio',
        'Lesiones Dolosas': 'lesiones dolosas',
        'Robo de Autopartes': 'robo de autopartes',
        'Abuso Sexual Infantil': 'abuso sexual infantil',
        'Robo a Interior de Vehículos': 'robo a int de vehiculos',
        'Robo de Motocicleta': 'robo de motocicleta',
        'Robo a Casa Habitación': 'robo a casa habitacion',
        'Homicidio Doloso': 'homicidio doloso',
        'Violación': 'violacion',
        'Robo a Cuentahabientes': 'robo a cuentahabientes',
        'Robo a Carga Pesada': 'robo a carga pesada',
        'Feminicidio': 'feminicidio',
        'Robo a Bancos': 'robo a bancos',
    }

    inc = pd.read_csv("datos/iieg_2023.csv").dropna(subset=["x", "y", "delito"])
    inc["delito"] = inc["delito"].str.strip().str.lower()
    
    gdf_inc = gpd.GeoDataFrame(
        inc,
        geometry=gpd.points_from_xy(inc["x"], inc["y"]),
        crs="EPSG:4326"
    ).to_crs(agebs_gdl.crs)
    
    join_del = gpd.sjoin(gdf_inc, agebs_gdl[["CVEGEO", "geometry"]], how="inner", predicate="within")
    
    # Construir tabla_del con todas las capas solicitadas
    tabla_del = pd.DataFrame({"CVEGEO": agebs_gdl["CVEGEO"].unique()})
    heatmap_data = {}
    
    # 1. Todos los Delitos
    total_counts = join_del.groupby("CVEGEO").size()
    tabla_del["Todos los Delitos"] = tabla_del["CVEGEO"].map(total_counts).fillna(0).astype(int)
    heatmap_data["Todos los Delitos"] = join_del[["y", "x"]].values.tolist()
    heatmap_data["TODOS_DELITOS"] = heatmap_data["Todos los Delitos"]
    
    # 2. Grupos
    for nombre_grupo, delitos_lista in grupos_delito.items():
        sub_df = join_del[join_del["delito"].isin(delitos_lista)]
        counts = sub_df.groupby("CVEGEO").size()
        tabla_del[nombre_grupo] = tabla_del["CVEGEO"].map(counts).fillna(0).astype(int)
        heatmap_data[nombre_grupo] = sub_df[["y", "x"]].values.tolist()
        
    # 3. Individuales
    for nombre_ind, delito_key in individuales_delito.items():
        sub_df = join_del[join_del["delito"] == delito_key]
        counts = sub_df.groupby("CVEGEO").size()
        tabla_del[nombre_ind] = tabla_del["CVEGEO"].map(counts).fillna(0).astype(int)
        heatmap_data[nombre_ind] = sub_df[["y", "x"]].values.tolist()
        
    categorias_delito = {"Todos los Delitos": True}
    for k in grupos_delito.keys():
        categorias_delito[k] = True
    for k in individuales_delito.keys():
        categorias_delito[k] = True

    
    # Temporal
    join_del["fecha_dt"] = pd.to_datetime(join_del["fecha"], errors="coerce")
    join_del = join_del.dropna(subset=["fecha_dt"])
    join_del["mes_num"] = join_del["fecha_dt"].dt.month
    join_del["dia_sem_num"] = join_del["fecha_dt"].dt.dayofweek
    
    meses_nombres = {1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun", 7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"}
    dias_nombres = {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves", 4: "Viernes", 5: "Sábado", 6: "Domingo"}
    
    temporal_meses = join_del["mes_num"].value_counts().sort_index().rename(index=meses_nombres).to_dict()
    temporal_dias = join_del["dia_sem_num"].value_counts().sort_index().rename(index=dias_nombres).to_dict()
    ranking_delitos_todos = join_del["delito"].str.title().value_counts().to_dict()
    
    temp_json = {
        "labels_meses": list(temporal_meses.keys()),
        "valores_meses": list(temporal_meses.values()),
        "labels_dias": list(temporal_dias.keys()),
        "valores_dias": list(temporal_dias.values()),
        "labels_delitos": list(ranking_delitos_todos.keys())[:15],
        "valores_delitos": list(ranking_delitos_todos.values())[:15]
    }
    
    # Colonias completo (todas para el Analista IA)
    colonia_stats = join_del.groupby("colonia").agg(
        total=("delito", "count"),
        lat=("y", "mean"),
        lng=("x", "mean")
    ).reset_index()
    
    # Añadir desglose de tipos de delitos por colonia para el Analista de IA
    delitos_por_colonia = join_del.groupby(["colonia", "delito"]).size().unstack(fill_value=0)
    
    top_colonias_30 = colonia_stats.sort_values(by="total", ascending=False).head(30).to_dict(orient="records")
    colonia_list_all = colonia_stats.sort_values(by="colonia").to_dict(orient="records")
    
    # === GENERACIÓN DE POLÍGONOS DE COLONIAS (DISSOLVE AGEBS) ===
    # Calculamos la distancia de cada centroide AGEB a cada centroide de Colonia
    agebs_coords = agebs_gdl[["centroid_y", "centroid_x"]].values
    cols_coords = colonia_stats[["lat", "lng"]].values
    distances = cdist(agebs_coords, cols_coords)
    nearest_idx = distances.argmin(axis=1)
    
    # Asignar a cada AGEB la colonia más cercana y disolver (agrupar polígonos)
    agebs_gdl["colonia_asignada"] = colonia_stats["colonia"].iloc[nearest_idx].values
    colonias_poly = agebs_gdl.dissolve(by="colonia_asignada").reset_index()
    
    # Unir las estadísticas para incluirlas en el GeoJSON de Colonias
    colonias_poly = colonias_poly.merge(colonia_stats, left_on="colonia_asignada", right_on="colonia", how="left")
    colonias_geojson_dict = json.loads(colonias_poly.to_json())
    
    # 3. DENUE
    denue = pd.read_csv("datos/gdl_denue.csv").dropna(subset=["longitud", "latitud", "categoria"])
    denue = denue[denue["categoria"] != "Otro"].copy()
    gdf_denue = gpd.GeoDataFrame(
        denue,
        geometry=gpd.points_from_xy(denue.longitud, denue.latitud),
        crs="EPSG:4326"
    ).to_crs(agebs_gdl.crs)
    join_dn = gpd.sjoin(gdf_denue, agebs_gdl[["CVEGEO", "geometry"]], how="inner", predicate="within")
    tabla_dn = join_dn.groupby(["CVEGEO", "categoria"]).size().unstack(fill_value=0).reset_index()
    tabla_dn.columns = ["CVEGEO"] + [f"NEG: {c}" for c in tabla_dn.columns if c != "CVEGEO"]
    
    for cat in join_dn["categoria"].unique():
        cat_pts = join_dn[join_dn["categoria"] == cat]
        heatmap_data[f"NEG: {cat}"] = cat_pts[["latitud", "longitud"]].values.tolist()
        
    # 4. Censo 2020 (usando archivo filtrado para Guadalajara)
    try:
        resag_gdl = pd.read_csv("datos/gdl_censo_filtrado.csv", low_memory=False)
    except FileNotFoundError:
        resag = pd.read_csv("datos/RESAGEBURB_14CSV20.csv", encoding="latin1", low_memory=False)
        resag_gdl = resag[(resag["MUN"] == 39) & (resag["MZA"] == 0)].copy()
        resag_gdl.to_csv("datos/gdl_censo_filtrado.csv", index=False)
        
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
    tabla_cen = resag_gdl[cols_sel].copy().rename(columns=cols_census)
    for col in tabla_cen.columns:
        if col != "CVEGEO":
            tabla_cen[col] = pd.to_numeric(tabla_cen[col], errors="coerce").fillna(0)
            
    # 5. Entorno Urbano (usando archivo filtrado para Guadalajara)
    try:
        frentes_gdl = pd.read_csv("datos/gdl_frentes_filtrado.csv", dtype={"CVEGEO": str, "CVE_MUN": str})
    except FileNotFoundError:
        frentes = pd.read_csv("datos/inv_frentes.csv", dtype={"CVEGEO": str, "CVE_MUN": str})
        frentes_gdl = frentes[frentes["CVE_MUN"].str.zfill(3) == "039"].copy()
        frentes_gdl.to_csv("datos/gdl_frentes_filtrado.csv", index=False)
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
        
    # 6. ENVIPE (Cruce de Edades y Sexos)
    df_per = pd.read_csv("datos/gdl_percepcion.csv")
    perc_dia = df_per["SEGURA_DIA"].value_counts().to_dict()
    perc_noche = df_per["SEGURA_NOCHE"].value_counts().to_dict()
    perc_problemas = df_per["PROBLEMA_PRINC"].value_counts().head(10).to_dict()
    
    df_vic = pd.read_csv("datos/gdl_victimas.csv")
    delitos_sufridos_lista = df_vic["DELITOS_SUFRIDOS"].dropna().str.split(", ").explode()
    ranking_victimas_del = delitos_sufridos_lista.value_counts().head(10).to_dict()
    
    def agrupar_edad(e):
        try:
            val = float(e)
            if val < 18: return "Menor de 18"
            elif val <= 29: return "18-29 años"
            elif val <= 44: return "30-44 años"
            elif val <= 59: return "45-59 años"
            else: return "60 años o más"
        except:
            return "No especificado"
            
    df_vic["edad_grupo"] = df_vic["EDAD"].apply(agrupar_edad)
    
    # Cruce completo de Edad y Sexo de Víctimas (Muestreo básico)
    edad_sexo_df = df_vic.groupby(["edad_grupo", "SEXO_STR"]).size().unstack(fill_value=0)
    edad_grupos_labels = ["Menor de 18", "18-29 años", "30-44 años", "45-59 años", "60 años o más"]
    
    edad_mujeres = [int(edad_sexo_df.loc[g, "Mujer"]) if "Mujer" in edad_sexo_df.columns and g in edad_sexo_df.index else 0 for g in edad_grupos_labels]
    edad_hombres = [int(edad_sexo_df.loc[g, "Hombre"]) if "Hombre" in edad_sexo_df.columns and g in edad_sexo_df.index else 0 for g in edad_grupos_labels]
    
    # --- CÁLCULO DE DATOS AVANZADOS (PERFIL DE VÍCTIMA & PIRÁMIDE DE VOLUMEN) ---
    df_delitos = pd.read_csv("datos/gdl_delitos.csv")
    df_full = pd.merge(df_delitos, df_vic[["ID_PER", "EDAD", "SEXO_STR"]], on="ID_PER", how="inner", suffixes=("", "_vic"))
    df_full["SEXO_STR"] = df_full["SEXO_STR_vic"].combine_first(df_full["SEXO_STR"])
    df_full["EDAD"] = df_full["EDAD_vic"].combine_first(df_full["EDAD"])
    
    bins_pir = [17, 29, 44, 59, 100]
    labels_pir = ["18-29 años", "30-44 años", "45-59 años", "60 años o más"]
    df_full["Rango_Edad"] = pd.cut(df_full["EDAD"], bins=bins_pir, labels=labels_pir)
    df_full["Volumen_Estimado"] = df_full["FAC_DEL"].fillna(1)
    
    # 1. Pirámide de victimización por volumen estimado
    pir_df = df_full.groupby(["Rango_Edad", "SEXO_STR"])["Volumen_Estimado"].sum().unstack(fill_value=0)
    pir_hombres = [int(pir_df.loc[g, "Hombre"]) if "Hombre" in pir_df.columns and g in pir_df.index else 0 for g in labels_pir]
    pir_mujeres = [int(pir_df.loc[g, "Mujer"]) if "Mujer" in pir_df.columns and g in pir_df.index else 0 for g in labels_pir]
    
    # 2. Top 5 Delitos por Sexo
    delitos_sexo = df_full.groupby(["SEXO_STR", "TIPO_DELITO_STR"])["Volumen_Estimado"].sum().reset_index()
    hombres_top5 = delitos_sexo[delitos_sexo["SEXO_STR"] == "Hombre"].nlargest(5, "Volumen_Estimado")
    mujeres_top5 = delitos_sexo[delitos_sexo["SEXO_STR"] == "Mujer"].nlargest(5, "Volumen_Estimado")
    
    top5_hombres_labels = [k[:25] for k in hombres_top5["TIPO_DELITO_STR"].tolist()]
    top5_hombres_values = hombres_top5["Volumen_Estimado"].astype(int).tolist()
    top5_mujeres_labels = [k[:25] for k in mujeres_top5["TIPO_DELITO_STR"].tolist()]
    top5_mujeres_values = mujeres_top5["Volumen_Estimado"].astype(int).tolist()
    
    # 3. Evolución de Riesgo (Calle vs Patrimonio)
    delitos_violencia_calle = [
        "Robo en calle (a transeúnte)", "Robo en transporte público", "Robo total de vehículo",
        "Robo de accesorios de vehículo", "Lesiones", "Amenazas/intimidación", "Delito sexual", "Secuestro"
    ]
    delitos_patrimonio_casa = [
        "Extorsión", "Fraude bancario", "Robo en vivienda", "Robo en banco/cajero",
        "Robo en negocio", "Homicidio de familiar", "Otro delito"
    ]
    def clasificar_riesgo(d):
        if d in delitos_violencia_calle: return "Callejero"
        if d in delitos_patrimonio_casa: return "Patrimonial"
        return "Otro"
    
    df_full["Categoria_Riesgo"] = df_full["TIPO_DELITO_STR"].apply(clasificar_riesgo)
    evolucion_df = df_full.groupby(["Rango_Edad", "Categoria_Riesgo"])["Volumen_Estimado"].sum().unstack(fill_value=0)
    evolucion_pct = evolucion_df.div(evolucion_df.sum(axis=1), axis=0) * 100
    
    evolucion_calle = [round(float(evolucion_pct.loc[g, "Callejero"]), 1) if "Callejero" in evolucion_pct.columns and g in evolucion_pct.index else 0.0 for g in labels_pir]
    evolucion_patr = [round(float(evolucion_pct.loc[g, "Patrimonial"]), 1) if "Patrimonial" in evolucion_pct.columns and g in evolucion_pct.index else 0.0 for g in labels_pir]
    
    # 4. Heatmap de Riesgo Específico (Composición delictiva por perfil)
    df_full["Perfil"] = df_full["SEXO_STR"] + " - " + df_full["Rango_Edad"].astype(str).str.replace(" años", "")
    envipe_heatmap_df = df_full.groupby(["TIPO_DELITO_STR", "Perfil"])["Volumen_Estimado"].sum().unstack(fill_value=0)
    heatmap_pct = (envipe_heatmap_df.div(envipe_heatmap_df.sum(axis=0), axis=1) * 100).round(1)
    top_delitos_order = envipe_heatmap_df.sum(axis=1).sort_values(ascending=False).index.tolist()
    heatmap_pct = heatmap_pct.loc[top_delitos_order]
    
    heatmap_profiles = heatmap_pct.columns.tolist()
    heatmap_delitos = heatmap_pct.index.tolist()
    heatmap_matrix = heatmap_pct.values.tolist()
    
    df_sdem = pd.read_csv("datos/gdl_sdem.csv")
    nivel_edu_map = {
        0: 'Sin escolaridad', 1: 'Preescolar', 2: 'Primaria',
        3: 'Secundaria', 4: 'Carrera técnica con secundaria',
        5: 'Normal básica', 6: 'Preparatoria o bachillerato',
        7: 'Carrera técnica con preparatoria', 8: 'Licenciatura',
        9: 'Posgrado (Maestría/Doctorado)', 99: 'No especificado'
    }
    df_sdem["NIVEL_EDU"] = df_sdem["NIV"].map(nivel_edu_map).fillna("No especificado")
    educacion_dist = df_sdem["NIVEL_EDU"].value_counts().to_dict()
    
    env_json = {
        "perc_dia_labels": list(perc_dia.keys()),
        "perc_dia_values": list(perc_dia.values()),
        "perc_noche_labels": list(perc_noche.keys()),
        "perc_noche_values": list(perc_noche.values()),
        "prob_labels": [k[:25] for k in perc_problemas.keys()],
        "prob_values": list(perc_problemas.values()),
        "vic_del_labels": [k[:25] for k in ranking_victimas_del.keys()],
        "vic_del_values": list(ranking_victimas_del.values()),
        "edad_labels": edad_grupos_labels,
        "edad_mujeres": edad_mujeres,
        "edad_hombres": edad_hombres,
        "edu_labels": list(educacion_dist.keys()),
        "edu_values": list(educacion_dist.values()),
        
        # Nuevos datos avanzados
        "pir_labels": labels_pir,
        "pir_hombres": pir_hombres,
        "pir_mujeres": pir_mujeres,
        "top5_hombres_labels": top5_hombres_labels,
        "top5_hombres_values": top5_hombres_values,
        "top5_mujeres_labels": top5_mujeres_labels,
        "top5_mujeres_values": top5_mujeres_values,
        "evolucion_labels": labels_pir,
        "evolucion_calle": evolucion_calle,
        "evolucion_patr": evolucion_patr,
        "heatmap_profiles": heatmap_profiles,
        "heatmap_delitos": heatmap_delitos,
        "heatmap_matrix": heatmap_matrix
    }
    
    # Unir capas para el mapa
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
    
    todas_capas = capas_delitos + capas_negocios + capas_censo + capas_urbano
    datos_js = {capa: agebs_final.set_index("CVEGEO")[capa].to_dict() for capa in todas_capas}
    
    paletas_js = {}
    for c in capas_delitos: paletas_js[c] = "rojo"
    for c in capas_negocios: paletas_js[c] = "azul"
    for c in capas_censo: paletas_js[c] = "verde"
    for c in capas_urbano: paletas_js[c] = "naranja"
    
    geojson_dict = json.loads(agebs_final.to_json())
    
    delitos_breakdown_dict = delitos_por_colonia.to_dict(orient="index")
    
    cache_data["main_payload"] = {
        "geojson": geojson_dict,
        "colonias_geojson": colonias_geojson_dict,
        "delitos_breakdown": delitos_breakdown_dict,
        "datos": datos_js,
        "paletas": paletas_js,
        "pt_data": heatmap_data,
        "temp_data": temp_json,
        "col_data": top_colonias_30,
        "col_list_all": colonia_list_all,
        "env_data": env_json,
        "capas_delitos": capas_delitos,
        "capas_negocios": capas_negocios,
        "capas_censo": capas_censo,
        "capas_urbano": capas_urbano
    }
    
    cache_data["agebs_gdl"] = agebs_final
    cache_data["delitos_por_colonia"] = delitos_por_colonia
    print("✅ Preprocesamiento completado con éxito.")

# Cargar al arrancar
cargar_y_procesar_todo()

@app.route("/api/data", methods=["GET"])
def get_data():
    return jsonify(cache_data["main_payload"])


@app.route("/api/analista", methods=["POST"])
def get_analisis_ia():
    payload = request.json or {}
    colonia = payload.get("colonia", "").strip()
    custom_key = payload.get("api_key", "").strip()
    
    if not colonia:
        return jsonify({"error": "Debe especificar una colonia para analizar."}), 400
        
    # Obtener API key (primero la provista por el cliente en el modal, luego la del servidor)
    api_key = custom_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return jsonify({
            "error": "Falta la clave API de Gemini. Por favor configúrala en el servidor como variable de entorno o provéela desde los ajustes del chat."
        }), 400
        
    try:
        # Configurar Gemini
        genai.configure(api_key=api_key)
        
        # Buscar un modelo compatible de forma auto-curativa
        model_name = None
        candidatos = [
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-pro",
            "gemini-1.5-pro",
            "gemini-2.5-flash",
            "gemini-1.0-pro"
        ]
        
        try:
            # Intentar listar los modelos disponibles en esta API Key
            modelos_disponibles = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            print("Modelos disponibles en tu API Key:", modelos_disponibles)
            available_clean = [m.replace("models/", "") for m in modelos_disponibles]
            for c in candidatos:
                if c in available_clean:
                    model_name = c
                    break
            if not model_name and available_clean:
                model_name = available_clean[0]
        except Exception as e_list:
            print("No se pudo listar modelos (posible falta de permisos en list_models), usando fallbacks secuenciales...", e_list)
            
        if not model_name:
            model_name = "gemini-1.5-flash"
            
        # 1. Obtener estadísticas de la colonia
        col_list = cache_data["main_payload"]["col_list_all"]
        col_data = next((c for c in col_list if c["colonia"].lower() == colonia.lower()), None)
        
        if not col_data:
            return jsonify({"error": f"No se encontraron registros detallados para la colonia '{colonia}'"}), 404
            
        col_total = col_data["total"]
        col_lat = col_data["lat"]
        col_lng = col_data["lng"]
        
        # Desglose de delitos específicos en esta colonia
        delitos_df = cache_data["delitos_por_colonia"]
        colonia_key = next((idx for idx in delitos_df.index if idx.lower() == colonia.lower()), None)
        
        delitos_info = ""
        if colonia_key is not None:
            delitos_col = delitos_df.loc[colonia_key]
            delitos_activos = delitos_col[delitos_col > 0].sort_values(ascending=False).head(8)
            delitos_info = ", ".join([f"{k.title()} ({v})" for k, v in delitos_activos.items()])
        else:
            delitos_info = "Sin desglose detallado disponible."
            
        # 2. Encontrar el AGEB más cercano para enriquecer con datos del censo y entorno urbano
        agebs_final = cache_data["agebs_gdl"]
        best_dist = float('inf')
        best_ageb = None
        
        for idx, row in agebs_final.iterrows():
            dist = (row["centroid_y"] - col_lat)**2 + (row["centroid_x"] - col_lng)**2
            if dist < best_dist:
                best_dist = dist
                best_ageb = row
                
        urb_info = ""
        censo_info = ""
        if best_ageb is not None:
            ageb_id = best_ageb["CVEGEO"]
            # Extraer variables urbanas
            alumbrado = best_ageb.get("URB: % Frentes con Alumbrado", 0)
            banqueta = best_ageb.get("URB: % Frentes con Banqueta", 0)
            arboles = best_ageb.get("URB: % Frentes con Árboles", 0)
            
            # Extraer variables censo
            poblacion = best_ageb.get("CEN: Población Total", 0)
            educacion = best_ageb.get("CEN: Escolaridad Promedio (años)", 0)
            desempleo = best_ageb.get("CEN: Población Desocupada", 0)
            
            urb_info = f"Alumbrado público: {alumbrado:.1f}%, Banquetes peatonales: {banqueta:.1f}%, Frentes arbolados: {arboles:.1f}%"
            censo_info = f"Población estimada del sector: {int(poblacion):,}, Escolaridad promedio: {educacion:.1f} años, Desempleados registrados: {int(desempleo):,}"
        else:
            urb_info = "No disponible"
            censo_info = "No disponible"
            
        # Construir Prompt Enriquecido
        prompt = f"""
        Analiza las condiciones de seguridad pública e infraestructura urbana de la colonia '{colonia.upper()}' en el municipio de Guadalajara, Jalisco.
        
        A continuación, te proporciono la información estadística exacta extraída de nuestro sistema geoespacial de Guadalajara (IIEG + DENUE + Censo INEGI 2020):
        
        DATOS DE INCIDENCIA DELICTIVA (IIEG 2023):
        - Total de denuncias registradas en esta colonia en 2023: {col_total} delitos.
        - Delitos más frecuentes en esta zona: {delitos_info}.
        - Ubicación geográfica centralizada (GPS): Latitud {col_lat:.5f}, Longitud {col_lng:.5f}.
        
        INFRAESTRUCTURA URBANA ESTIMADA EN EL ENTORNO (Cruces de Frentes del INEGI):
        - {urb_info}.
        
        PERFIL SOCIODEMOGRÁFICO ESTIMADO (Censo de Población INEGI):
        - {censo_info}.
        
        DATOS DE CONTEXTO GENERAL METROPOLITANO (Encuesta de Seguridad ENVIPE GDL):
        - El 88.4% de los habitantes de Guadalajara se siente inseguro de noche en su entorno vecinal.
        - Los tres problemas comunitarios reportados como más críticos en las colonias son: 1) Pandillas o bandas delictivas, 2) Venta/distribución de drogas, 3) Consumo público de sustancias.
        
        Escribe un diagnóstico analítico premium para la colonia '{colonia.upper()}' estructurado exactamente de la siguiente manera:
        
        1. 🏷️ **Nivel de Riesgo Estimado**: Determina una categoría de riesgo única ("Bajo", "Moderado", "Alto", o "Crítico") según la tasa delictiva e infraestructura. Explica brevemente por qué.
        2. 🔍 **Diagnóstico de Seguridad**: Analiza el impacto de los delitos más recurrentes y cómo la infraestructura (como la falta de alumbrado o banquetas) influye en este comportamiento delictivo.
        3. 🛡️ **Recomendaciones de Prevención para Vecinos**: Da 4 consejos prácticos orientados a los habitantes de esta zona en su vida diaria.
        4. 🏗️ **Recomendaciones de Intervención Urbana**: Propón 3 acciones específicas que las autoridades municipales de Guadalajara o el IIEG deberían sugerir para mitigar el riesgo a través del diseño urbano de esta colonia.
        
        Mantén un tono profesional, analítico y riguroso. Utiliza formato Markdown de manera excelente (con negritas, listas y símbolos) para que luzca perfecto en nuestro frontend React.
        """
        
        # Bucle de re-intento auto-curativo sobre candidatos
        response = None
        ultimo_error = None
        modelos_a_probar = [model_name] + [c for c in candidatos if c != model_name]
        
        for cand in modelos_a_probar:
            try:
                print(f"-> Probando modelo Gemini: {cand}...")
                model = genai.GenerativeModel(cand)
                response = model.generate_content(prompt)
                print(f"✅ ¡Éxito total consultando el modelo {cand}!")
                break
            except Exception as e_call:
                print(f"⚠️ El modelo {cand} falló o no está soportado: {str(e_call)}")
                ultimo_error = e_call
                continue
                
        if response is None:
            raise ultimo_error if ultimo_error else Exception("Ningún modelo de Gemini pudo ser consultado con éxito.")
            
        reporte_markdown = response.text
        
        return jsonify({
            "colonia": colonia,
            "analisis": reporte_markdown
        })
        
    except Exception as e:
        return jsonify({"error": f"Error al consultar la API de Gemini: {str(e)}"}), 500

@app.route("/", methods=["GET"])
def index():
    # Servir el frontend principal
    return send_from_directory(".", "index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🔥 Servidor Backend de Analítica Urbana escuchando en http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
