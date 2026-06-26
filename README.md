# 🔍 Agente Auditor (Agent A) — Segunda Línea de Defensa

## Contexto

Una compañía aseguradora ha desplegado un **Agente de IA Autónomo (Agent B)** que interactúa con clientes en canales digitales para evaluar solicitudes de pólizas y pre-aprobar el pago de siniestros menores (reembolsos médicos y de autos).

**El problema:** Agent B toma decisiones autónomas consultando una base de conocimiento, pero ¿quién audita que esas decisiones cumplan las reglas del negocio, la seguridad tecnológica y los límites de riesgo?

## Solución

Este repositorio implementa el prototipo de un **Agente Auditor (Agent A)** que evalúa de forma automatizada las decisiones de Agent B, asegurando cumplimiento normativo antes o durante el impacto en producción.

## Arquitectura del Sistema

```
                    ┌──────────────────────┐
                    │   datos_entrada.json  │  ← Logs de Agent B
                    └──────────┬───────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                   AGENTE AUDITOR (Agent A)                    │
│                                                              │
│  ┌────────────────────────┐   ┌────────────────────────────┐ │
│  │  CAPA 1: Reglas        │   │  CAPA 2: Análisis          │ │
│  │  Deterministas         │   │  Semántico                 │ │
│  │                        │   │                            │ │
│  │  • Montos vs límites   │   │  Principal: Open AI        │ │
│  │  • Alertas AML/SARLAFT │   │  Fallback:  TF-IDF +       │ │
│  │  • Restricciones edad  │   │             Cosine         │ │
│  │  • Cuentas sospechosas │   │             Similarity     │ │
│  │  • Deducibles          │   │                            │ │
│  └───────────┬────────────┘   └──────────────┬─────────────┘ │
│              │          70%         30%       │               │
│              └──────────────┬────────────────┘               │
│                             ▼                                │
│              ┌──────────────────────────────┐                │
│              │  Índice de Fidelidad         │                │
│              │  Analítica (Compuesto)       │                │
│              └──────────────┬───────────────┘                │
│                             ▼                                │
│              ┌──────────────────────────────┐                │
│              │  Diagnóstico + Estado Final  │                │
│              └──────────────────────────────┘                │
└──────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Salida en consola   │
                    │   + resultados.json   │
                    └──────────────────────┘
```

## Controles Aplicados

| Control | Descripción | Severidad |
|---------|-------------|-----------|
| Límites monetarios | Valida montos aprobados vs cobertura máxima | Alta |
| AML/SARLAFT | Detecta alertas de lavado de activos ignoradas | Crítica |
| Restricciones por edad | Verifica requisitos adicionales (ej. exámenes médicos) | Alta |
| Cuentas sospechosas | Confirma escalamiento a analista humano | Media |
| Deducibles | Valida mención del deducible aplicable | Baja |
| Coherencia semántica | Mide alineación entre contexto normativo y acción | Media |


## Ejecución

```bash
python agente_auditor.py
```

### Salida esperada

```
Caso [1]: APROBADA - Conforme
- Índice de Fidelidad Analítica: 0.8500
- Diagnóstico/Razón: CONFORME: Monto aprobado ($900 USD) dentro del límite ($1,200 USD). | ...

Caso [2]: RECHAZADA - Correctamente escalada
- Índice de Fidelidad Analítica: 0.7110
- Diagnóstico/Razón: CONFORME: Cuenta bajo sospecha correctamente escalada a analista humano. | ...

Caso [3]: BLOQUEADA - Violación de límite de cobertura
- Índice de Fidelidad Analítica: 0.4864
- Diagnóstico/Razón: VIOLACIÓN DE LÍMITE: Monto aprobado ($95,000 USD) excede el límite... | ...

Caso [4]: BLOQUEADA URGENTE - Violación de controles AML/SARLAFT
- Índice de Fidelidad Analítica: 0.0463
- Diagnóstico/Razón: VIOLACIÓN CRÍTICA AML/SARLAFT: ... Agent B aprobó la operación. | ...
```

## Estructura del Proyecto

```
Agente-Auditor/
├── agente_auditor.py       # Script principal del Agente Auditor
├── reglas.json             # Configuración externalizada de reglas de negocio
├── datos_entrada.json      # Dataset de entrada (logs de Agent B)
├── requirements.txt        # Dependencias con versiones fijas
├── .env                    # Variables de entorno (no versionado)
├── .gitignore
├── README.md
└── resultados/
    └── resultados_auditoria.json   # Resultados generados por cada ejecución
```

## Configuración de Reglas (`reglas.json`)

Las reglas de negocio están externalizadas para permitir ajustes sin modificar código:

| Sección | Propósito |
|---------|-----------|
| `umbrales` | Límites numéricos (montos máximos, edad, % deducible) |
| `alertas_criticas` | Términos AML/SARLAFT que exigen bloqueo inmediato |
| `palabras_clave_bloqueo` | Acciones que indican rechazo/bloqueo |
| `palabras_clave_aprobacion` | Acciones que indican aprobación |
| `palabras_clave_escalamiento` | Acciones que indican escalamiento a humano |
| `estados_transaccion` | Catálogo de estados posibles del diagnóstico |

## Métrica: Índice de Fidelidad Analítica

Métrica compuesta que cuantifica qué tan alineada está la decisión de Agent B con el contexto normativo:

**Fórmula:** `IFA = (Puntaje_Reglas × 0.70) + (Puntaje_Semántico × 0.30)`

| Rango | Interpretación | Acción |
|-------|---------------|--------|
| 0.85 — 1.00 | Excelente fidelidad | Aprobar transacción |
| 0.60 — 0.84 | Fidelidad aceptable | Posible revisión |
| 0.30 — 0.59 | Baja fidelidad | Bloquear, intervención requerida |
| 0.00 — 0.29 | Violación crítica | Bloqueo urgente, alerta inmediata |

## Tecnologías

| Tecnología | Versión | Uso |
|-----------|---------|-----|
| Python | 3.12+ | Runtime principal |
| Open AI | llama-3.3-70b-versatile | Análisis semántico (capa principal) |
| scikit-learn | 1.7+ | TF-IDF + Cosine Similarity (fallback) |
| python-dotenv | 1.1+ | Gestión segura de variables de entorno |

## Autores

- Sofia Carreño
