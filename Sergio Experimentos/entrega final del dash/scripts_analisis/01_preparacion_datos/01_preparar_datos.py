"""
SCRIPT 1 — Filtrar y preparar datos para Guadalajara
======================================================
Fuentes:
  - ENVIPE (TSDem, TMod_Vic, TPer_Vic1, TPer_Vic2, TVivienda)
  - DENUE Jalisco (denue_inegi_14_.csv)

Salidas:
  - gdl_sdem.csv       → perfil sociodemográfico por persona
  - gdl_delitos.csv    → modalidad y tipo de cada delito
  - gdl_victimas.csv   → quién fue víctima y de qué
  - gdl_percepcion.csv → percepción de seguridad por vivienda
  - gdl_denue.csv      → establecimientos con coordenadas, clasificados
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ── Diccionarios ENVIPE ────────────────────────────────────────────────────────

# Tipo de delito (BPCOD en TMod_Vic)
TIPO_DELITO = {
    1:  'Robo total de vehículo',
    2:  'Robo de accesorios de vehículo',
    3:  'Robo en vivienda',
    4:  'Robo en transporte público',
    5:  'Robo en calle (a transeúnte)',
    6:  'Robo en negocio',
    7:  'Robo en banco/cajero',
    8:  'Fraude bancario',
    9:  'Extorsión',
    10: 'Amenazas/intimidación',
    11: 'Lesiones',
    12: 'Secuestro',
    13: 'Delito sexual',
    14: 'Homicidio de familiar',
    15: 'Otro delito',
}

# Lugar donde ocurrió el delito (AREAM_OCU)
LUGAR_DELITO = {
    1:  'Calle o vía pública',
    2:  'En su vivienda',
    3:  'En vivienda de otra persona',
    4:  'En negocio o empresa',
    5:  'En institución bancaria',
    6:  'En transporte público',
    7:  'En vehículo particular',
    8:  'En mercado o tianguis',
    9:  'En escuela',
    10: 'En centro comercial',
    11: 'En hospital o clínica',
    12: 'En bar, cantina o antro',
    13: 'En estacionamiento',
    14: 'En carretera',
    15: 'En parque o área verde',
    16: 'En cajero automático',
    17: 'En gasolinera',
    18: 'En terminal (bus/tren/metro)',
    19: 'En oficina gubernamental',
    21: 'En restaurante',
    24: 'En hotel o motel',
    26: 'En iglesia',
    27: 'En campo/área rural',
    28: 'En zona industrial',
    29: 'En otro lugar',
    31: 'Internet/medios digitales',
    32: 'Por teléfono',
    33: 'Sin especificar',
    36: 'En tienda de conveniencia',
    39: 'En farmacia',
    40: 'En joyería',
    41: 'En casa de empeño',
    43: 'En otro comercio',
}

# Problemas en la colonia (AP2_1 en TVivienda — principal problema)
PROBLEMA_COLONIA = {
    1: 'Robos frecuentes',
    2: 'Pandillas/bandas',
    3: 'Consumo de drogas',
    4: 'Venta de drogas',
    5: 'Violencia entre vecinos',
    6: 'Disparos frecuentes',
    7: 'Extorsión a negocios',
    8: 'Prostitución',
    9: 'Ninguno',
    10: 'Otro',
}

# Tipos de delito sufridos AP4_2_xx (en TPer_Vic1)
DELITO_SUFRIDO = {
    'AP4_2_01': 'Robo vehículo',
    'AP4_2_02': 'Robo accesorios vehículo',
    'AP4_2_03': 'Robo en vivienda',
    'AP4_2_04': 'Robo en transporte público',
    'AP4_2_05': 'Robo en calle',
    'AP4_2_06': 'Secuestro',
    'AP4_2_07': 'Fraude bancario',
    'AP4_2_08': 'Extorsión',
    'AP4_2_09': 'Delito sexual',
    'AP4_2_10': 'Lesiones',
    'AP4_2_11': 'Amenazas',
    'AP4_2_12': 'Homicidio familiar',
    'AP4_2_13': 'Otro',
}

# ── Categorías DENUE para análisis de seguridad ───────────────────────────────
CATEGORIAS_DENUE = {
    'Bar / Cantina':         ['comercio al por menor de bebidas alcohólicas','bar','cantina','pulque'],
    'Antro / Discoteca':     ['discoteca','antro','club nocturno','salón de baile'],
    'Restaurante':           ['restaurante','taquería','torta','antojito','pizza','hamburguesa','mariscos'],
    'Farmacia':              ['farmacia','botica'],
    'Banco / Financiero':    ['banca múltiple','institución de crédito','caja popular','unión de crédito','casa de cambio'],
    'Cajero ATM':            ['cajero automático'],
    'Joyería / Relojería':   ['joyería','reloj','plata','oro','artículos de joyería'],
    'Gasolinera':            ['gasolina y diesel','gasolina y diésel'],
    'Hotel / Motel':         ['hotel','motel','posada'],
    'Tienda conveniencia':   ['conveniencia','minisuper','miscelánea','abarrotes'],
    'Casa de empeño':        ['empeño','monte de piedad','préstamos'],
    'Licorería':             ['licorería','vinos y licores'],
    'Supermercado':          ['supermercado','autoservicio','hiper'],
    'Estacionamiento':       ['estacionamiento'],
    'Casino / Apuestas':     ['casino','apuesta','juego con apuesta'],
    'Telefonía / Electrónica':['telefonía','teléfono','electrónica','cómputo'],
    'Escuela':               ['escuela','colegio','jardín de niños','primaria','secundaria','preparatoria','universidad'],
    'Hospital / Clínica':    ['hospital','clínica','sanatorio','consultorio de medicina'],
    'Farmacia / Droguería':  ['farmacia','droguería'],
    'Policía / Seguridad':   ['seguridad pública','vigilancia privada','custodia'],
}

def clasificar_actividad(nombre_act):
    if pd.isna(nombre_act):
        return 'Otro'
    nombre_lower = nombre_act.lower()
    for cat, keywords in CATEGORIAS_DENUE.items():
        if any(kw in nombre_lower for kw in keywords):
            return cat
    return 'Otro'

# ══════════════════════════════════════════════════════════════════════════════
# 1. DENUE — Establecimientos de Guadalajara
# ══════════════════════════════════════════════════════════════════════════════
print("Procesando DENUE...")
denue = pd.read_csv('denue_inegi_14_.csv', encoding='latin1')
gdl_denue = denue[denue['municipio'].str.upper() == 'GUADALAJARA'].copy()
gdl_denue = gdl_denue.dropna(subset=['latitud', 'longitud'])

gdl_denue['categoria'] = gdl_denue['nombre_act'].apply(clasificar_actividad)

# Conservar solo columnas útiles
gdl_denue = gdl_denue[[
    'id', 'nom_estab', 'nombre_act', 'categoria',
    'per_ocu', 'nomb_asent', 'cod_postal',
    'latitud', 'longitud', 'fecha_alta'
]]

gdl_denue.to_csv('gdl_denue.csv', index=False)
print(f"  → {len(gdl_denue):,} establecimientos en Guadalajara")
print("  Distribución por categoría:")
print(gdl_denue['categoria'].value_counts().head(15))

# ══════════════════════════════════════════════════════════════════════════════
# 2. ENVIPE — Sociodemográfico Guadalajara
# ══════════════════════════════════════════════════════════════════════════════
print("\nProcesando TSDem...")
tsdem = pd.read_csv('TSDem.csv', encoding='latin1')
gdl_sdem = tsdem[tsdem['NOM_MUN'] == 'GUADALAJARA'].copy()

gdl_sdem['SEXO_STR']  = gdl_sdem['SEXO'].map({1: 'Hombre', 2: 'Mujer'})
gdl_sdem['NIVEL_EDU'] = gdl_sdem['NIV'].map({
    0: 'Sin escolaridad', 1: 'Preescolar', 2: 'Primaria',
    3: 'Secundaria', 4: 'Carrera técnica con secundaria',
    5: 'Normal básica', 6: 'Preparatoria o bachillerato',
    7: 'Carrera técnica con preparatoria', 8: 'Licenciatura',
    9: 'Posgrado (Maestría/Doctorado)', 99: 'No especificado'
})

gdl_sdem.to_csv('gdl_sdem.csv', index=False)
print(f"  → {len(gdl_sdem):,} personas en Guadalajara")

# ══════════════════════════════════════════════════════════════════════════════
# 3. ENVIPE — Modalidad del delito (nacional, con etiquetas)
# ══════════════════════════════════════════════════════════════════════════════
print("\nProcesando TMod_Vic...")
tmod = pd.read_csv('TMod_Vic.csv', encoding='latin1')

tmod['TIPO_DELITO_STR'] = tmod['BPCOD'].map(TIPO_DELITO)
tmod['LUGAR_STR']       = tmod['AREAM_OCU'].map(LUGAR_DELITO)
tmod['SEXO_STR']        = tmod['SEXO'].map({1: 'Hombre', 2: 'Mujer'})

# Filtrar Jalisco cruzando por ID_PER con TSDem
ids_jalisco = tsdem[tsdem['NOM_ENT'] == 'JALISCO']['ID_PER'].values
tmod_jal = tmod[tmod['ID_PER'].isin(ids_jalisco)].copy()

ids_gdl = gdl_sdem['ID_PER'].values
tmod_gdl = tmod[tmod['ID_PER'].isin(ids_gdl)].copy()

tmod_jal.to_csv('gdl_delitos.csv', index=False)
print(f"  → {len(tmod_jal):,} registros de delitos en Jalisco")
print(f"  → {len(tmod_gdl):,} registros cruzados con Guadalajara")
print("  Tipos de delito más frecuentes:")
print(tmod_jal['TIPO_DELITO_STR'].value_counts().head(8))
print("  Lugares más frecuentes:")
print(tmod_jal['LUGAR_STR'].value_counts().head(8))

# ══════════════════════════════════════════════════════════════════════════════
# 4. ENVIPE — Victimización Guadalajara
# ══════════════════════════════════════════════════════════════════════════════
print("\nProcesando TPer_Vic1...")
tvic1 = pd.read_csv('TPer_Vic1.csv', encoding='latin1')
gdl_vic = tvic1[tvic1['NOM_MUN'] == 'GUADALAJARA'].copy()

# Crear columna de si fue víctima (1=sí en alguno de los delitos AP4_2_xx)
cols_delito = [c for c in gdl_vic.columns if c.startswith('AP4_2_')]
gdl_vic['FUE_VICTIMA'] = gdl_vic[cols_delito].eq(1).any(axis=1)

# Columna con lista de delitos sufridos
def delitos_sufridos(row):
    d = [DELITO_SUFRIDO[c] for c in cols_delito if row[c] == 1]
    return ', '.join(d) if d else 'Ninguno'

gdl_vic['DELITOS_SUFRIDOS'] = gdl_vic[cols_delito].apply(
    lambda row: delitos_sufridos(row), axis=1
)
gdl_vic['SEXO_STR'] = gdl_vic['SEXO'].map({1: 'Hombre', 2: 'Mujer'})

gdl_vic.to_csv('gdl_victimas.csv', index=False)
print(f"  → {len(gdl_vic):,} personas encuestadas en Guadalajara")
print(f"  → Víctimas declaradas: {gdl_vic['FUE_VICTIMA'].sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. ENVIPE — Percepción de seguridad Guadalajara
# ══════════════════════════════════════════════════════════════════════════════
print("\nProcesando TVivienda...")
tviv = pd.read_csv('TVivienda.csv', encoding='latin1')
gdl_perc = tviv[tviv['NOM_MUN'] == 'GUADALAJARA'].copy()

gdl_perc['SEGURA_DIA']   = gdl_perc['AP1_1'].map({1: 'Sí', 2: 'No', 3: 'No sabe', 4: 'No aplica'})
gdl_perc['SEGURA_NOCHE'] = gdl_perc['AP1_2'].map({1: 'Sí', 2: 'No', 3: 'No sabe'})
gdl_perc['PROBLEMA_PRINC'] = gdl_perc['AP2_1'].map(PROBLEMA_COLONIA)

gdl_perc.to_csv('gdl_percepcion.csv', index=False)
print(f"  → {len(gdl_perc):,} viviendas en Guadalajara")
print(f"  → Colonia segura de día:   {(gdl_perc['SEGURA_DIA']=='Sí').sum()} Sí / {(gdl_perc['SEGURA_DIA']=='No').sum()} No")
print(f"  → Colonia segura de noche: {(gdl_perc['SEGURA_NOCHE']=='Sí').sum()} Sí / {(gdl_perc['SEGURA_NOCHE']=='No').sum()} No")

print("\n✅ Todos los archivos preparados:")
print("   gdl_denue.csv, gdl_sdem.csv, gdl_delitos.csv, gdl_victimas.csv, gdl_percepcion.csv")