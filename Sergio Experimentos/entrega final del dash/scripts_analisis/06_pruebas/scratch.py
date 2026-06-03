from esda.moran import Moran_Local
import geopandas as gpd
from libpysal.weights import Queen
import matplotlib.pyplot as plt
from splot.esda import lisa_cluster

# Load a small sample
agebs = gpd.read_file('guadalajara_AGEB/2025_14039_A07052026_1549.shp')
import pandas as pd
df_iter = pd.read_csv('RESAGEBURB_14CSV20.csv', encoding='latin-1', nrows=1000)
# This is getting complicated to recreate perfectly in a scratch file. 
# It's better to just use sed/awk on 04_mas_datos.py or just edit it to print the labels and exit.
