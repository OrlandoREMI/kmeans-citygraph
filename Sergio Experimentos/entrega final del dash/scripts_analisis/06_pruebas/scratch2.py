import pandas as pd

print("Reading CSV...")
df = pd.read_csv('inv_frentes.csv', dtype=str)
print("Filtering for Guadalajara...")
gdl = df[df['CVE_MUN'] == '039']
print(f"Total rows for GDL: {len(gdl)}")

if len(gdl) > 0:
    print("\nColumns with _D:")
    cols_d = [c for c in gdl.columns if c.endswith('_D')]
    for c in cols_d[:5]:
        print(f"\n{c}:")
        print(gdl[c].value_counts().head(3))
        
    print("\nSample row:")
    print(gdl.iloc[0])
