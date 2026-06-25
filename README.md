# 🔍 Agente Auditor (Agent A) - Segunda Línea de Defensa

## Descripción

Prototipo de un **Agente Auditor de IA** diseñado como segunda línea de defensa para evaluar las decisiones de un Agente Autónomo (Agent B) en una compañía aseguradora. El sistema valida el cumplimiento de reglas de negocio, seguridad tecnológica y límites de riesgo antes o durante el impacto en producción.

## Arquitectura

```
┌─────────────────────────────────────────────────────────┐
│                   AGENTE AUDITOR (Agent A)               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────┐    ┌────────────────────────────┐  │
│  │  Capa 1: Reglas │    │  Capa 2: Análisis          │  │
│  │  Deterministas  │    │  Semántico                 │  │
│  │                 │    │                            │  │
│  │  • Montos       │    │  Principal: Gemini AI      │  │
│  │  • Alertas AML  │    │  Fallback: TF-IDF +        │  │
│  │  • Edad/Límites │    │  Cosine Similarity         │  │
│  │  • Sospecha     │    │                            │  │
│  └────────┬────────┘    └─────────────┬──────────────┘  │
│           │                           │                  │
│           └───────────┬───────────────┘                  │
│                       ▼                                  │
│         ┌─────────────────────────┐                      │
│         │  Índice de Fidelidad    │                      │
│         │  Analítica (Compuesto)  │                      │
│         │  70% Reglas + 30% IA    │                      │
│         └─────────────────────────┘                      │
│                       │                                  │
│                       ▼                                  │
│         ┌─────────────────────────┐                      │
│         │  Diagnóstico Final      │                      │
│         │  por Caso               │                      │
│         └─────────────────────────┘                      │
└─────────────────────────────────────────────────────────┘
```

## Requisitos

- Python 3.12+
- Dependencias: ver `requirements.txt`

## Instalación

```bash
# Clonar repositorio
git clone https://github.com/sofiacarc29/Agente-Auditor.git
cd Agente-Auditor

# Crear entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt

# Configurar API Key (opcional, tiene fallback)
# Crear archivo .env con:
# GEMINI_API_KEY=tu_api_key_aqui
```

## Ejecución

```bash
python agente_auditor.py
```

## Estructura del Proyecto

```
Agente-Auditor/
├── agente_auditor.py      # Script principal del Agente Auditor
├── reglas.json            # Archivo de configuración de reglas de negocio
├── datos_entrada.json     # Dataset de entrada (logs de Agent B)
├── requirements.txt       # Dependencias del proyecto
├── .env                   # Variables de entorno (API keys) - NO se sube al repo
├── .gitignore             # Archivos excluidos del repositorio
├── README.md              # Documentación del proyecto
└── resultados/            # Carpeta generada con reportes de auditoría
    ├── reporte_auditoria.txt
    └── resultados_auditoria.json
```

## Configuración de Reglas (`reglas.json`)

El archivo `reglas.json` contiene todas las variables de control externalizadas:

- **Umbrales**: Límites monetarios, porcentajes de deducible, edad máxima
- **Alertas críticas**: Palabras clave AML/SARLAFT que requieren bloqueo inmediato
- **Palabras clave**: Clasificación de acciones (bloqueo, aprobación, escalamiento)
- **Pesos de validación**: Ponderación de cada tipo de control
- **Estados**: Posibles estados de la transacción auditada

## Formato de Salida

```
Caso [X]: [Estado de la Transacción]
- Índice de Fidelidad Analítica: [Puntaje numérico 0.0000 - 1.0000]
- Diagnóstico/Razón: [Justificación generada por la lógica del auditor]
```

## Métrica: Índice de Fidelidad Analítica

Métrica compuesta que evalúa la consistencia entre el contexto normativo y la acción del agente:

- **70% Reglas deterministas**: Controles lógicos y matemáticos (montos, alertas, restricciones)
- **30% Análisis semántico**: Coherencia del lenguaje (Gemini o TF-IDF como fallback)

| Rango | Interpretación |
|-------|---------------|
| 0.85 - 1.00 | Excelente fidelidad, decisión conforme |
| 0.60 - 0.84 | Fidelidad aceptable, posible revisión |
| 0.30 - 0.59 | Baja fidelidad, requiere intervención |
| 0.00 - 0.29 | Violación crítica, bloqueo inmediato |

## Autores

- David Cárdenas

## Tecnologías

- Python 3.12+
- Google Gemini AI (análisis semántico)
- scikit-learn (TF-IDF + Cosine Similarity - fallback)
- python-dotenv (gestión de variables de entorno)
