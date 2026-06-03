import pandas as pd
df = pd.read_csv('inv_frentes.csv', dtype=str)
gdl = df[df['CVE_MUN'] == '039']
print(gdl['ALUMPUB_D'].value_counts())
