# Agente Auditor (Agent A) - Segunda Línea de Defensa

## Descripción

Prototipo de un **Agente Auditor de IA** diseñado como segunda línea de defensa para evaluar las decisiones de un Agente Autónomo (Agent B) en una compañía aseguradora. El sistema valida el cumplimiento de reglas de negocio, seguridad tecnológica y límites de riesgo.

## Arquitectura

```
┌─────────────────────────────────────────────────────────┐
│                   AGENTE AUDITOR (Agent A)               │
├─────────────────────────────────────────────────────────┤
│  Capa 1: Reglas deterministas (montos, AML, edad)     │
│  Capa 2: Análisis semántico (Gemini / TF-IDF fallback) │
│  Salida: Índice de Fidelidad Analítica (70% + 30%)      │
└─────────────────────────────────────────────────────────┘
```

## Requisitos

- Python 3.11+
- Dependencias: ver `requirements.txt`

## Instalación

```bash
git clone https://github.com/sofiacarc29/Agente-Auditor.git
cd Agente-Auditor

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt

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
├── agente_auditor.py      # Script principal
├── reglas.json            # Reglas de negocio
├── datos_entrada.json     # Casos de entrada (logs de Agent B)
├── requirements.txt       # Dependencias
├── .env                   # API keys (no se sube al repo)
├── .gitignore
├── README.md
└── resultados/            # Reportes generados
    ├── reporte_auditoria.txt
    └── resultados_auditoria.json
```

## Configuración de Reglas (`reglas.json`)

- **Umbrales**: Límites numéricos (ej. edad máxima para emisión automática)
- **Alertas críticas**: Palabras clave AML/SARLAFT
- **Palabras clave**: Acciones de bloqueo, aprobación y escalamiento
- **Estados**: Posibles resultados de la auditoría

## Formato de Salida

```
Caso [X]: [Estado de la Transacción]
- Índice de Fidelidad Analítica: [0.0000 - 1.0000]
- Diagnóstico/Razón: [Justificación]
```

## Índice de Fidelidad Analítica

- **70%** Reglas deterministas (montos, alertas, restricciones)
- **30%** Análisis semántico (Gemini o TF-IDF como fallback)

| Rango | Interpretación |
|-------|---------------|
| 0.85 - 1.00 | Excelente fidelidad |
| 0.60 - 0.84 | Fidelidad aceptable |
| 0.30 - 0.59 | Baja fidelidad |
| 0.00 - 0.29 | Violación crítica |

## Autor

Sofia Carreño

## Tecnologías

- Python 3.11+
- Google Gemini AI
- scikit-learn (TF-IDF fallback)
- python-dotenv
