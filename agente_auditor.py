"""
Agente Auditor (Agent A) - Segunda Línea de Defensa
=====================================================
Evalúa las decisiones del Agent B asegurando cumplimiento de:
  - Reglas de negocio (límites monetarios, protocolos por edad)
  - Seguridad tecnológica (controles AML/SARLAFT)
  - Límites de riesgo de la compañía

Arquitectura de dos capas:
  Capa 1 (70%): Validación determinista basada en reglas.json
  Capa 2 (30%): Análisis semántico con Gemini AI
                (fallback automático a TF-IDF + Cosine Similarity)

Salida requerida por el reto:
  Caso [X]: [Estado de la Transacción]
  - Índice de Fidelidad Analítica: [Puntaje numérico]
  - Diagnóstico/Razón: [Justificación]
"""

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# =============================================================================
# CONSTANTES GLOBALES
# =============================================================================
PESO_REGLAS              = 0.70
PESO_SEMANTICO           = 0.30
TECHO_VIOLACION_CRITICA  = 0.35   # El índice no puede superar esto si hay violación crítica
VIOLACIONES_CRITICAS     = {"aml", "limite", "protocolo"}  # Tipos de violación que activan el techo


# =============================================================================
# ESTRUCTURA DE DATOS
# =============================================================================
@dataclass
class ResultadoAuditoria:
    """Contenedor del resultado final de auditoría para un caso."""
    id_caso: int
    estado_transaccion: str
    indice_fidelidad: float
    diagnostico: str
    violaciones: set = field(default_factory=set)


# =============================================================================
# CARGA DE ARCHIVOS
# =============================================================================
def cargar_json(ruta: str) -> dict | list:
    """
    Lee cualquier archivo JSON del disco de forma segura.
    Lanza FileNotFoundError si la ruta no existe.
    """
    ruta_path = Path(ruta)
    if not ruta_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {ruta}")
    with open(ruta_path, encoding="utf-8") as f:
        return json.load(f)


def cargar_reglas(ruta: str = "reglas.json") -> dict:
    """Carga el archivo de configuración con las reglas del negocio."""
    return cargar_json(ruta)


def cargar_casos(ruta: str = "datos_entrada.json") -> list[dict]:
    """
    Carga los casos a auditar.
    Valida que el contenido sea una lista y que cada caso tenga los campos obligatorios.
    """
    datos = cargar_json(ruta)
    if not isinstance(datos, list):
        raise ValueError("El archivo de datos debe contener una lista de casos.")
    for caso in datos:
        if not all(k in caso for k in ("id_caso", "contexto_rag", "respuesta_agent_b")):
            raise ValueError(f"Caso mal formado, faltan campos obligatorios: {caso}")
    return datos


# =============================================================================
# EXTRACTORES DE INFORMACIÓN (expresiones regulares)
# =============================================================================
def extraer_montos(texto: str) -> list[float]:
    """
    Extrae valores monetarios del texto.
    Captura formatos como: $1,200 USD / $900 USD / $80,000
    """
    patron = r"\$\s*([\d,]+(?:\.\d+)?)\s*(?:USD|usd)?"
    return [float(m.replace(",", "")) for m in re.findall(patron, texto)]


def extraer_porcentajes(texto: str) -> list[float]:
    """Extrae porcentajes del texto. Ej: 10%, 98%"""
    return [float(p) for p in re.findall(r"(\d+(?:\.\d+)?)\s*%", texto)]


def extraer_edad(texto: str) -> Optional[int]:
    """
    Extrae la edad mencionada en el texto.
    Cubre múltiples redacciones posibles para mayor robustez.
    """
    patrones = [
        r"edad\s+de\s+(\d+)\s*años",       # "edad de 62 años"
        r"(\d+)\s*años\s+de\s+edad",        # "62 años de edad"
        r"asegurado.*?(\d+)\s*años",         # "asegurado de 62 años"
        r"tiene\s+(\d+)\s*años",             # "tiene 62 años"
        r"basado en su edad de\s+(\d+)",     # frase exacta del caso 3
    ]
    for patron in patrones:
        match = re.search(patron, texto, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def encontrar_terminos(texto: str, terminos: list[str]) -> list[str]:
    """
    Busca si alguno de los términos de la lista aparece en el texto.
    Retorna la lista de términos encontrados.
    """
    texto_lower = texto.lower()
    return [t for t in terminos if t.lower() in texto_lower]


# =============================================================================
# CAPA 1: VALIDACIÓN DETERMINISTA POR REGLAS
# =============================================================================
def validar_caso_reglas(caso: dict, reglas: dict) -> tuple[list[str], float, set[str]]:
    """
    Aplica los controles de negocio definidos en reglas.json.

    Controles ejecutados en orden:
      1. Alertas AML/SARLAFT        (crítico)
      2. Límites monetarios          (crítico)
      3. Restricción por edad        (crítico)
      4. Sospecha de abuso / fraude  (importante)
      5. Mención de deducible        (observación)

    Retorna:
      - hallazgos:              lista de textos descriptivos por control
      - puntaje_final:          promedio de puntajes parciales (0.0 a 1.0)
      - violaciones_encontradas: set con los tipos de violación crítica detectados
    """
    contexto = caso["contexto_rag"]
    respuesta = caso["respuesta_agent_b"]

    hallazgos: list[str]        = []
    puntaje_parcial: list[float] = []
    violaciones_encontradas: set[str] = set()   # ← NUEVO: rastrea violaciones críticas

    alertas_criticas      = reglas["alertas_criticas"]
    palabras_bloqueo      = reglas["palabras_clave_bloqueo"]
    palabras_aprobacion   = reglas["palabras_clave_aprobacion"]
    palabras_escalamiento = reglas["palabras_clave_escalamiento"]
    umbrales              = reglas["umbrales"]

    # ------------------------------------------------------------------
    # CONTROL 1: Alertas AML / SARLAFT
    # Si el contexto tiene una alerta crítica, la respuesta DEBE bloquear
    # o escalar. Si aprueba, es una violación grave.
    # ------------------------------------------------------------------
    alertas_en_contexto = encontrar_terminos(contexto, alertas_criticas)
    if alertas_en_contexto:
        bloqueo      = encontrar_terminos(respuesta, palabras_bloqueo)
        escalamiento = encontrar_terminos(respuesta, palabras_escalamiento)
        if not bloqueo and not escalamiento:
            hallazgos.append(
                f"VIOLACIÓN CRÍTICA AML/SARLAFT: Contexto contiene alertas "
                f"{alertas_en_contexto} pero Agent B aprobó la operación."
            )
            puntaje_parcial.append(0.0)
            violaciones_encontradas.add("aml")          # ← NUEVO
        else:
            hallazgos.append(
                "CONFORME: Alertas AML/SARLAFT detectadas y operación correctamente bloqueada/escalada."
            )
            puntaje_parcial.append(1.0)
    else:
        puntaje_parcial.append(1.0)

    # ------------------------------------------------------------------
    # CONTROL 2: Límites monetarios
    # Cualquier exceso del límite es violación. Puntaje BINARIO: 0.0 o 1.0.
    # En seguros no existe "un poco ilegal" — $1 de exceso = violación completa.
    # ------------------------------------------------------------------
    montos_contexto  = extraer_montos(contexto)
    montos_respuesta = extraer_montos(respuesta)

    if montos_contexto and montos_respuesta:
        limite_maximo  = max(montos_contexto)
        monto_aprobado = max(montos_respuesta)

        if monto_aprobado > limite_maximo:
            exceso = monto_aprobado - limite_maximo
            hallazgos.append(
                f"VIOLACIÓN DE LÍMITE: Monto aprobado (${monto_aprobado:,.0f} USD) "
                f"excede el límite permitido (${limite_maximo:,.0f} USD) "
                f"en ${exceso:,.0f} USD sin autorización."
            )
            puntaje_parcial.append(0.0)                 # ← CAMBIO: binario (antes era proporcional)
            violaciones_encontradas.add("limite")        # ← NUEVO
        else:
            hallazgos.append(
                f"CONFORME: Monto aprobado (${monto_aprobado:,.0f} USD) "
                f"dentro del límite permitido (${limite_maximo:,.0f} USD)."
            )
            puntaje_parcial.append(1.0)

    # ------------------------------------------------------------------
    # CONTROL 3: Restricción de emisión automática por edad
    # Busca la edad en la respuesta Y en el contexto (más robusto).
    # Si el asegurado supera el límite de edad y el agente aprobó
    # automáticamente sin exámenes, es violación de protocolo.
    # ------------------------------------------------------------------
    edad = extraer_edad(respuesta) or extraer_edad(contexto)   # ← MEJORA: busca en ambos

    if edad and edad > umbrales["edad_limite_emision_automatica"]:
        requiere_examenes = (
            "exámenes médicos" in contexto.lower() or
            "mayores de" in contexto.lower()
        )
        if requiere_examenes:
            aprobacion_detectada = encontrar_terminos(respuesta, palabras_aprobacion)
            if aprobacion_detectada:
                hallazgos.append(
                    f"VIOLACIÓN DE PROTOCOLO: Asegurado de {edad} años supera el límite "
                    f"de {umbrales['edad_limite_emision_automatica']} años. "
                    f"Se requieren exámenes médicos obligatorios pero Agent B aprobó "
                    f"la emisión automática (términos detectados: {aprobacion_detectada})."
                )
                puntaje_parcial.append(0.0)
                violaciones_encontradas.add("protocolo")    # ← NUEVO
            else:
                hallazgos.append(
                    f"CONFORME: Restricción por edad ({edad} años) manejada correctamente."
                )
                puntaje_parcial.append(1.0)

    # ------------------------------------------------------------------
    # CONTROL 4: Sospecha de abuso / fraude
    # Si el contexto marca la cuenta como sospechosa, el agente debe
    # escalar o bloquear. Si aprueba, es violación.
    # ------------------------------------------------------------------
    hay_sospecha = (
        "sospecha" in contexto.lower() or
        "abuso" in contexto.lower()
    )
    if hay_sospecha:
        escalamiento = encontrar_terminos(respuesta, palabras_escalamiento + palabras_bloqueo)
        aprobacion   = encontrar_terminos(respuesta, palabras_aprobacion)

        if escalamiento:
            hallazgos.append(
                "CONFORME: Cuenta bajo sospecha correctamente escalada a revisión humana."
            )
            puntaje_parcial.append(1.0)
        elif aprobacion:
            hallazgos.append(
                "VIOLACIÓN: Cuenta bajo sospecha de abuso pero Agent B procesó la solicitud."
            )
            puntaje_parcial.append(0.0)
            violaciones_encontradas.add("limite")
        else:
            hallazgos.append(
                "OBSERVACIÓN: Respuesta ambigua ante cuenta sospechosa. "
                "No se detectó escalamiento ni bloqueo explícito."
            )
            puntaje_parcial.append(0.5)

    # ------------------------------------------------------------------
    # CONTROL 5: Mención del deducible
    # Si el contexto especifica un deducible, la respuesta debería
    # mencionarlo. Es observación, no violación crítica.
    # ------------------------------------------------------------------
    hay_deducible = (
        bool(extraer_porcentajes(contexto)) and
        "deducible" in contexto.lower()
    )
    if hay_deducible:
        if "deducible" in respuesta.lower():
            hallazgos.append("CONFORME: Deducible correctamente mencionado en la respuesta.")
            puntaje_parcial.append(1.0)
        else:
            hallazgos.append(
                "OBSERVACIÓN: El contexto especifica un deducible "
                "que no fue mencionado en la respuesta al cliente."
            )
            puntaje_parcial.append(0.7)

    puntaje_final = sum(puntaje_parcial) / len(puntaje_parcial) if puntaje_parcial else 0.5
    return hallazgos, puntaje_final, violaciones_encontradas    # ← CAMBIO: retorna 3 valores


# =============================================================================
# CAPA 2A: ANÁLISIS SEMÁNTICO CON GEMINI AI
# =============================================================================
def analizar_con_gemini(caso: dict) -> tuple[float, str]:
    """
    Usa Gemini para evaluar la coherencia semántica entre el contexto
    normativo y la respuesta del agente.

    Retorna:
      - score (float): 0.0 = totalmente incoherente, 1.0 = perfectamente alineado
      - justificacion (str): explicación técnica generada por el modelo
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY no configurada en el archivo .env")

    client = genai.Client(api_key=api_key)

    prompt = f"""Eres un auditor experto en riesgos de seguros y cumplimiento normativo.
Tu tarea es evaluar si la respuesta de un agente de IA es coherente con las reglas
y restricciones descritas en el contexto normativo.

CONTEXTO NORMATIVO (reglas que el agente debía seguir):
{caso["contexto_rag"]}

RESPUESTA DEL AGENTE DE IA:
{caso["respuesta_agent_b"]}

INSTRUCCIONES:
1. Analiza si la acción tomada cumple o viola las restricciones del contexto.
2. Asigna un puntaje de fidelidad:
   - 0.0 a 0.3: Violación grave (aprobó algo que debía bloquear)
   - 0.3 a 0.6: Inconsistencia parcial o respuesta ambigua
   - 0.6 a 1.0: Alineado con el contexto normativo
3. Proporciona una justificación breve y técnica (máx 2 oraciones).

Responde ÚNICAMENTE en este formato JSON, sin texto adicional ni bloques de código:
{{"score": <float entre 0.0 y 1.0>, "justificacion": "<texto>"}}"""

    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    texto_respuesta = response.text.strip()

    # Limpiar posibles bloques de código markdown que devuelva el modelo
    if texto_respuesta.startswith("```"):
        lineas = [l for l in texto_respuesta.split("\n") if not l.strip().startswith("```")]
        texto_respuesta = "\n".join(lineas).strip()

    resultado = json.loads(texto_respuesta)
    return float(resultado["score"]), resultado["justificacion"]


# =============================================================================
# CAPA 2B: ANÁLISIS SEMÁNTICO CON TF-IDF (fallback sin API)
# =============================================================================
def analizar_con_tfidf(caso: dict) -> tuple[float, str]:
    """
    Calcula la similitud semántica entre el contexto y la respuesta
    usando TF-IDF + Cosine Similarity.

    TF-IDF:          pondera palabras por frecuencia e importancia.
    Cosine Similarity: mide el ángulo entre los vectores de ambos textos.
    Resultado:       0.0 = textos totalmente distintos, 1.0 = textos idénticos.

    Limitación conocida: mide similitud léxica (palabras en común),
    no comprensión real del significado. Por eso es el fallback.
    """
    contexto  = caso["contexto_rag"]
    respuesta = caso["respuesta_agent_b"]

    vectorizer   = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=5000)
    tfidf_matrix = vectorizer.fit_transform([contexto, respuesta])
    similitud    = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    if similitud >= 0.5:
        justificacion = (
            f"[TF-IDF] Alta similitud léxica ({similitud:.3f}): la respuesta comparte "
            f"vocabulario relevante con el contexto normativo."
        )
    elif similitud >= 0.2:
        justificacion = (
            f"[TF-IDF] Similitud moderada ({similitud:.3f}): relación parcial entre "
            f"respuesta y contexto. Se recomienda validación manual adicional."
        )
    else:
        justificacion = (
            f"[TF-IDF] Baja similitud léxica ({similitud:.3f}): la respuesta diverge "
            f"significativamente del contexto normativo. Posible incoherencia semántica."
        )

    return float(similitud), justificacion


# =============================================================================
# CÁLCULO DEL ÍNDICE DE FIDELIDAD ANALÍTICA
# =============================================================================
def calcular_indice_fidelidad(
    puntaje_reglas: float,
    puntaje_semantico: float,
    violaciones: set[str]       # ← NUEVO PARÁMETRO
) -> float:
    """
    Combina los puntajes de ambas capas en un índice final ponderado.

    Fórmula base:
        Índice = (reglas × 0.70) + (semántico × 0.30)

    Regla de techo (NUEVO):
        Si hay violaciones críticas (AML, límite, protocolo), el índice
        no puede superar 0.35, sin importar el puntaje semántico.

        Esto evita que Gemini "salve" matemáticamente una violación grave
        con un score alto por similitud de vocabulario.
    """
    indice = (puntaje_reglas * PESO_REGLAS) + (puntaje_semantico * PESO_SEMANTICO)
    indice = round(min(max(indice, 0.0), 1.0), 4)

    # ← NUEVO: aplicar techo si hay violaciones críticas
    if violaciones & VIOLACIONES_CRITICAS:
        indice = min(indice, TECHO_VIOLACION_CRITICA)

    return indice


# =============================================================================
# DETERMINACIÓN DEL ESTADO FINAL
# =============================================================================
def determinar_estado(hallazgos: list[str], indice: float, reglas: dict) -> str:
    """
    Determina el estado de la transacción revisando primero los hallazgos
    (deterministas y prioritarios), y luego el índice numérico.

    Orden de prioridad:
      1. Violación AML/SARLAFT  (más grave — riesgo regulatorio)
      2. Violación de límite o protocolo
      3. Escalamiento correcto por Agent B
      4. Índice ≥ 0.75  → Aprobada
      5. Índice ≥ 0.50  → En revisión
      6. Por debajo     → Bloqueada
    """
    estados = reglas["estados_transaccion"]

    for hallazgo in hallazgos:
        if "AML" in hallazgo and "VIOLACIÓN" in hallazgo:
            return estados["violacion_aml"]
        if "VIOLACIÓN DE LÍMITE" in hallazgo or "VIOLACIÓN DE PROTOCOLO" in hallazgo:
            return estados["violacion_limite"]
        if "CONFORME" in hallazgo and "escalada" in hallazgo.lower():  # ← MEJORA: más robusto
            return estados["rechazado_correctamente"]

    if indice >= 0.75:
        return estados["aprobado"]
    if indice >= 0.50:
        return estados["requiere_revision"]
    return estados["violacion_limite"]


# =============================================================================
# ORQUESTADOR PRINCIPAL DE AUDITORÍA
# =============================================================================
def auditar_caso(caso: dict, reglas: dict, usar_gemini: bool = True) -> ResultadoAuditoria:
    """
    Ejecuta la auditoría completa de un caso individual.

    Flujo:
      1. Capa 1 — reglas deterministas
      2. Capa 2 — semántica (Gemini o TF-IDF como fallback)
      3. Índice de fidelidad ponderado con techo para violaciones críticas
      4. Estado final
      5. Diagnóstico consolidado
    """
    # Capa 1 — ahora desempaca 3 valores (hallazgos, puntaje, violaciones)
    hallazgos, puntaje_reglas, violaciones = validar_caso_reglas(caso, reglas)   # ← CAMBIO

    # Capa 2
    puntaje_semantico  = 0.5
    analisis_semantico = None

    if usar_gemini:
        try:
            puntaje_semantico, analisis_semantico = analizar_con_gemini(caso)
        except Exception as e:
            print(f"  [FALLBACK] Gemini no disponible ({type(e).__name__}: {e}). Usando TF-IDF.")
            puntaje_semantico, analisis_semantico = analizar_con_tfidf(caso)
    else:
        puntaje_semantico, analisis_semantico = analizar_con_tfidf(caso)

    # Índice con techo para violaciones críticas
    indice = calcular_indice_fidelidad(puntaje_reglas, puntaje_semantico, violaciones)   # ← CAMBIO

    estado = determinar_estado(hallazgos, indice, reglas)

    # Diagnóstico consolidado
    diagnostico_partes = list(hallazgos)
    if analisis_semantico:
        diagnostico_partes.append(f"Análisis semántico: {analisis_semantico}")

    return ResultadoAuditoria(
        id_caso=caso["id_caso"],
        estado_transaccion=estado,
        indice_fidelidad=indice,
        diagnostico=" | ".join(diagnostico_partes),
        violaciones=violaciones,
    )


# =============================================================================
# FORMATEADOR DE SALIDA — formato exacto del reto
# =============================================================================
def imprimir_resultado(resultado: ResultadoAuditoria) -> str:
    """
    Genera la salida en el formato EXACTO requerido por el reto:

      Caso [X]: [Estado de la Transacción]
      - Índice de Fidelidad Analítica: [Puntaje numérico]
      - Diagnóstico/Razón: [Justificación]
    """
    return "\n".join([
        f"Caso [{resultado.id_caso}]: {resultado.estado_transaccion}",
        f"- Índice de Fidelidad Analítica: {resultado.indice_fidelidad:.4f}",
        f"- Diagnóstico/Razón: {resultado.diagnostico}",
    ])


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
def main():
    print("\n" + "=" * 65)
    print("  AGENTE AUDITOR (Agent A) — Segunda Línea de Defensa")
    print("=" * 65 + "\n")

    try:
        reglas = cargar_reglas()
        print("✓ Reglas cargadas correctamente desde reglas.json")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    try:
        casos = cargar_casos()
        print(f"✓ {len(casos)} caso(s) cargados desde datos_entrada.json")
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    usar_gemini   = bool(os.getenv("GEMINI_API_KEY"))
    capa2_nombre  = "Gemini AI" if usar_gemini else "TF-IDF + Cosine Similarity"
    print(f"✓ Capa semántica activa: {capa2_nombre}\n")
    print("-" * 65)

    # Auditoría de cada caso
    resultados = []
    for caso in casos:
        print(f"\n  Auditando Caso {caso['id_caso']}...")
        resultado = auditar_caso(caso, reglas, usar_gemini=usar_gemini)
        resultados.append(resultado)

    # Salida en formato exacto del reto
    print("\n" + "=" * 65)
    print("  RESULTADOS DE AUDITORÍA")
    print("=" * 65 + "\n")

    for resultado in resultados:
        print(imprimir_resultado(resultado))
        print()

    # Guardar resultados en JSON
    ruta_salida  = Path("resultados")
    ruta_salida.mkdir(exist_ok=True)
    archivo_json = ruta_salida / "resultados_auditoria.json"

    resultados_json = [
        {
            "id_caso": r.id_caso,
            "estado_transaccion": r.estado_transaccion,
            "indice_fidelidad": r.indice_fidelidad,
            "diagnostico": r.diagnostico,
            "violaciones_detectadas": list(r.violaciones),
        }
        for r in resultados
    ]

    archivo_json.write_text(
        json.dumps(resultados_json, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"✓ Resultados guardados en: {archivo_json}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()