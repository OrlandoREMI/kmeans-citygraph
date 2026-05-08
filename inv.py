import geopandas as gpd

# Cargar el shapefile
gdf = gpd.read_file(r"C:\Users\Andrea\Documents\CIIIA\14_Frentes_INV2020_shp\INV2020_IND_EU_FTE_14.shp")

# Quitar geometría y guardar como CSV
df = gdf.drop(columns="geometry")
df.to_csv(r"C:\Users\Andrea\Documents\CIIIA\inv_frentes.csv", index=False)

print(f"Listo — {len(df)} filas, {len(df.columns)} columnas")
print(f"Guardado en: C:\\Users\\Andrea\\Documents\\CIIIA\\inv_frentes.csv")