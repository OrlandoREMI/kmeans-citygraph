# Análisis Geoespacial de la Incidencia Delictiva y de Movilidad en Guadalajara

Este proyecto de Ciencia de Datos realiza un análisis espacial, demográfico y de movilidad urbana a partir de la incidencia delictiva registrada en Guadalajara, Jalisco. Combina bases de datos del **IIEG**, **DENUE**, **Censo de Población (INEGI)**, la encuesta **ENVIPE** y modelado de grafos viales con **OSMnx** y **NetworkX**.

Adicionalmente, incorpora un **Dashboard Interactivo Web (React + Flask)** potenciado con inteligencia artificial (**Gemini AI**) para la generación de diagnósticos automáticos por colonia y visualización en tiempo real de capas demográficas y delictivas.

---

## 📄 Documentos Principales del Entregable
Los reportes formales y metodológicos compilados se ubican directamente en la raíz para facilitar su consulta:
1. 📈 **[reporte_proyecto.pdf](reporte_proyecto.pdf)**: Reporte de investigación final científico, hipótesis, correlaciones (Spearman, I de Moran) y análisis de movilidad.
2. 📘 **[GUIA_DEL_PROYECTO.pdf](GUIA_DEL_PROYECTO.pdf)**: Manual técnico exhaustivo que describe el flujo de desarrollo, los scripts y la arquitectura.

---

## 📂 Estructura General del Proyecto

```
proyecto_mapas/
├── README.md                      # Esta guía rápida
├── reporte_proyecto.pdf           # Reporte científico compilado (PDF)
├── GUIA_DEL_PROYECTO.pdf          # Guía técnica compilada (PDF)
│
├── 📊 dashboard/                  # Aplicación interactiva autocontenida
│   ├── app.py                     # Servidor backend de análisis y API
│   ├── index.html                 # Interfaz gráfica (React + Leaflet + Chart.js)
│   ├── requirements.txt           # Dependencias mínimas del dashboard
│   └── datos/                     # Bases de datos filtradas y shapefiles locales
│
├── 🔬 scripts_analisis/           # Módulos de scripts de investigación
│   ├── 01_preparacion_datos/      # Limpieza y filtrado inicial de bases
│   ├── 02_correlaciones/          # Matrices y análisis paramétricos/no paramétricos
│   ├── 03_geoespacial/            # Generación de heatmaps y clústeres
│   ├── 04_sociodemografico/       # Perfil del individuo y cruces de educación
│   └── 05_movilidad_vial/         # Grafos de ruta y simulación de patrullajes
│
├── 📁 base_datos_crudas/          # Bases de datos originales completas (INEGI, IIEG)
├── 📈 resultados_graficos/        # Gráficas fijas en alta resolución generadas para los reportes
└── 📄 fuentes_latex/              # Código fuente de los reportes LaTeX (.tex, .toc, .aux)
```

---

## 🚀 Guía de Instalación y Ejecución Rápida

Sigue estos pasos para instalar y ejecutar el **Dashboard Urbano** localmente.

### 1. Requisitos Previos
* **Python 3.10 o superior** instalado en el sistema.
* Una clave API de Gemini (**GEMINI_API_KEY**). Puedes obtener una clave gratuita en [Google AI Studio](https://aistudio.google.com/).

### 2. Configurar el Entorno Virtual
Crea un entorno virtual de Python dentro del directorio `dashboard/` e instala las dependencias necesarias:

```bash
# Entrar a la carpeta del dashboard
cd dashboard/

# Crear entorno virtual
python3 -m venv venv

# Activar el entorno
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Levantar la Aplicación
Define tu clave de API de Gemini como variable de entorno e inicia el servidor Flask:

```bash
# Configurar clave API (Linux/macOS)
export GEMINI_API_KEY="Tu_Clave_API_De_Gemini_Aqui"

# Iniciar servidor
python app.py
```

### 4. Abrir en el Navegador
Una vez levantado el servidor backend, abre tu navegador web favorito y entra a:
👉 **[http://localhost:5000](http://localhost:5000)**

---

## 🧠 Características Destacadas del Dashboard
* **Vista Simple (Choropleth)**: Mapeo interactivo a nivel AGEB de los 16 delitos oficiales y capas sociodemográficas (escolaridad, iluminación, población).
* **Vista Bivariada**: Relación espacial cruzada entre giros económicos del DENUE y tipos delictivos.
* **Vista de Calor (Heatmap)**: Densidad de puntos críticos de incidencia acumulada en el municipio.
* **Temporal & ENVIPE**: Panel expandido a pantalla completa con 12 visualizaciones avanzadas de vulnerabilidad delictiva por rangos de edad, sexo y escolaridad.
* **Analista de IA Integrado**: Chat interactivo lateral que consume estadísticas locales de la colonia seleccionada y utiliza Gemini AI para generar recomendaciones de intervención urbana personalizadas.
