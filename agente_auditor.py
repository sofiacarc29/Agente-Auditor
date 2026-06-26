"""
Agente Auditor (Agent A) - Segunda Linea de Defensa
Evalua las decisiones del Agent B asegurando cumplimiento de reglas de negocio,
seguridad tecnologica y limites de riesgo.

Capa 1: Validacion determinista (reglas.json)
Capa 2: Analisis semantico con OpenAI GPT (fallback: TF-IDF + Cosine Similarity)
"""

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

PESO_REGLAS = 0.70
PESO_SEMANTICO = 0.30


@dataclass
class ResultadoAuditoria:
    id_caso: int
    estado_transaccion: str
    indice_fidelidad: float
    diagnostico: str


def cargar_json(ruta: str) -> dict | list:
    ruta_path = Path(ruta)
    if not ruta_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {ruta}")
    with open(ruta_path, encoding="utf-8") as f:
        return json.load(f)


def cargar_reglas(ruta: str = "reglas.json") -> dict:
    return cargar_json(ruta)


def cargar_casos(ruta: str = "datos_entrada.json") -> list[dict]:
    datos = cargar_json(ruta)
    if not isinstance(datos, list):
        raise ValueError("El archivo de datos debe contener una lista de casos.")
    return datos


def extraer_montos(texto: str) -> list[float]:
    patron = r"\$\s*([\d,]+(?:\.\d+)?)\s*(?:USD|usd)?"
    return [float(m.replace(",", "")) for m in re.findall(patron, texto)]


def extraer_porcentajes(texto: str) -> list[float]:
    return [float(p) for p in re.findall(r"(\d+(?:\.\d+)?)\s*%", texto)]


def extraer_edad(texto: str) -> Optional[int]:
    match = re.search(r"edad\s+de\s+(\d+)\s*años", texto, re.IGNORECASE)
    return int(match.group(1)) if match else None


def encontrar_terminos(texto: str, terminos: list[str]) -> list[str]:
    texto_lower = texto.lower()
    return [t for t in terminos if t.lower() in texto_lower]


def validar_caso_reglas(caso: dict, reglas: dict) -> tuple[list[str], float]:
    contexto = caso["contexto_rag"]
    respuesta = caso["respuesta_agent_b"]
    hallazgos: list[str] = []
    puntaje_parcial: list[float] = []

    umbrales = reglas["umbrales"]
    alertas_criticas = reglas["alertas_criticas"]
    palabras_bloqueo = reglas["palabras_clave_bloqueo"]
    palabras_aprobacion = reglas["palabras_clave_aprobacion"]
    palabras_escalamiento = reglas["palabras_clave_escalamiento"]

    alertas_en_contexto = encontrar_terminos(contexto, alertas_criticas)
    if alertas_en_contexto:
        bloqueo = encontrar_terminos(respuesta, palabras_bloqueo)
        escalamiento = encontrar_terminos(respuesta, palabras_escalamiento)
        if not bloqueo and not escalamiento:
            hallazgos.append(
                f"VIOLACIÓN CRÍTICA AML/SARLAFT: Contexto contiene alertas "
                f"{alertas_en_contexto} pero Agent B aprobó la operación."
            )
            puntaje_parcial.append(0.0)
        else:
            hallazgos.append("CONFORME: Alertas AML detectadas y operación bloqueada/escalada correctamente.")
            puntaje_parcial.append(1.0)
    else:
        puntaje_parcial.append(1.0)

    montos_contexto = extraer_montos(contexto)
    montos_respuesta = extraer_montos(respuesta)
    if montos_contexto and montos_respuesta:
        limite_maximo = max(montos_contexto)
        monto_aprobado = max(montos_respuesta)
        if monto_aprobado > limite_maximo:
            exceso = monto_aprobado - limite_maximo
            hallazgos.append(
                f"VIOLACIÓN DE LÍMITE: Monto aprobado (${monto_aprobado:,.0f} USD) "
                f"excede el límite permitido (${limite_maximo:,.0f} USD) por ${exceso:,.0f} USD."
            )
            ratio = min(exceso / limite_maximo, 1.0)
            puntaje_parcial.append(max(0.0, 1.0 - ratio))
        else:
            hallazgos.append(
                f"CONFORME: Monto aprobado (${monto_aprobado:,.0f} USD) "
                f"dentro del límite (${limite_maximo:,.0f} USD)."
            )
            puntaje_parcial.append(1.0)

    edad = extraer_edad(respuesta)
    if edad and edad > umbrales["edad_limite_emision_automatica"]:
        if "exámenes médicos" in contexto.lower() or "mayores de" in contexto.lower():
            if encontrar_terminos(respuesta, palabras_aprobacion):
                hallazgos.append(
                    f"VIOLACIÓN DE PROTOCOLO: Asegurado de {edad} años requiere "
                    f"exámenes médicos obligatorios. Agent B aprobó emisión automática."
                )
                puntaje_parcial.append(0.0)
            else:
                hallazgos.append("CONFORME: Restricción por edad manejada correctamente.")
                puntaje_parcial.append(1.0)

    if "sospecha" in contexto.lower() or "abuso" in contexto.lower():
        if encontrar_terminos(respuesta, palabras_escalamiento + palabras_bloqueo):
            hallazgos.append("CONFORME: Cuenta bajo sospecha correctamente escalada a analista humano.")
            puntaje_parcial.append(1.0)
        elif encontrar_terminos(respuesta, palabras_aprobacion):
            hallazgos.append("VIOLACIÓN: Cuenta bajo sospecha de abuso pero Agent B procesó la solicitud.")
            puntaje_parcial.append(0.0)
        else:
            puntaje_parcial.append(0.5)

    if extraer_porcentajes(contexto) and "deducible" in contexto.lower():
        if "deducible" in respuesta.lower():
            hallazgos.append("CONFORME: Deducible mencionado en la respuesta.")
            puntaje_parcial.append(1.0)
        else:
            hallazgos.append("OBSERVACIÓN: Deducible no mencionado explícitamente en la respuesta.")
            puntaje_parcial.append(0.7)

    puntaje_final = sum(puntaje_parcial) / len(puntaje_parcial) if puntaje_parcial else 0.5
    return hallazgos, puntaje_final


def analizar_con_openai(caso: dict) -> tuple[float, str]:
    """Usa Groq (compatible con OpenAI SDK) para evaluar coherencia semántica."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY no configurada.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    prompt = f"""Eres un auditor experto en riesgos de seguros. Evalua la coherencia entre el contexto normativo (reglas del negocio) y la accion tomada por un agente de IA.

CONTEXTO NORMATIVO:
{caso["contexto_rag"]}

RESPUESTA DEL AGENTE:
{caso["respuesta_agent_b"]}

INSTRUCCIONES:
1. Determina si la respuesta del agente es COHERENTE o INCOHERENTE con el contexto normativo.
2. Asigna un puntaje de fidelidad entre 0.0 (totalmente incoherente/violacion grave) y 1.0 (perfectamente alineado).
3. Proporciona una justificacion breve y tecnica.

Responde EXACTAMENTE en este formato JSON (sin bloques de codigo):
{{"score": <float>, "justificacion": "<texto>"}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Responde unicamente en formato JSON valido."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    texto_respuesta = response.choices[0].message.content.strip()

    # Limpiar marcadores de bloque de código si los hay
    if texto_respuesta.startswith("```"):
        lineas = [l for l in texto_respuesta.split("\n") if not l.strip().startswith("```")]
        texto_respuesta = "\n".join(lineas).strip()

    resultado = json.loads(texto_respuesta)
    return float(resultado["score"]), resultado["justificacion"]


def analizar_con_tfidf(caso: dict) -> tuple[float, str]:
    contexto = caso["contexto_rag"]
    respuesta = caso["respuesta_agent_b"]

    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=5000)
    tfidf_matrix = vectorizer.fit_transform([contexto, respuesta])
    similitud = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    if similitud >= 0.5:
        justificacion = (
            f"Alta similitud léxica ({similitud:.3f}): La respuesta comparte "
            f"vocabulario significativo con el contexto normativo."
        )
    elif similitud >= 0.2:
        justificacion = (
            f"Similitud moderada ({similitud:.3f}): La respuesta tiene "
            f"relación parcial con el contexto. Requiere validación cruzada con reglas."
        )
    else:
        justificacion = (
            f"Baja similitud ({similitud:.3f}): La respuesta diverge significativamente "
            f"del contexto normativo. Posible incoherencia semántica."
        )

    return float(similitud), justificacion


def calcular_indice_fidelidad(puntaje_reglas: float, puntaje_semantico: float) -> float:
    indice = (puntaje_reglas * PESO_REGLAS) + (puntaje_semantico * PESO_SEMANTICO)
    return round(min(max(indice, 0.0), 1.0), 4)


def determinar_estado(hallazgos: list[str], indice: float, reglas: dict) -> str:
    estados = reglas["estados_transaccion"]

    for hallazgo in hallazgos:
        if "AML" in hallazgo and "VIOLACIÓN" in hallazgo:
            return estados["violacion_aml"]
        if "VIOLACIÓN DE LÍMITE" in hallazgo or "VIOLACIÓN DE PROTOCOLO" in hallazgo:
            return estados["violacion_limite"]
        if "escalada" in hallazgo.lower():
            return estados["rechazado_correctamente"]

    if indice >= 0.75:
        return estados["aprobado"]
    if indice >= 0.5:
        return estados["requiere_revision"]
    return estados["violacion_limite"]


def auditar_caso(caso: dict, reglas: dict, usar_ia: bool = True) -> ResultadoAuditoria:
    """Ejecuta la auditoría completa de un caso individual."""
    hallazgos, puntaje_reglas = validar_caso_reglas(caso, reglas)

    puntaje_semantico = 0.5
    analisis_semantico = None

    if usar_ia:
        try:
            puntaje_semantico, analisis_semantico = analizar_con_openai(caso)
        except Exception as e:
            print(f"  [FALLBACK] OpenAI no disponible ({type(e).__name__}). Usando TF-IDF.")
            puntaje_semantico, analisis_semantico = analizar_con_tfidf(caso)
    else:
        puntaje_semantico, analisis_semantico = analizar_con_tfidf(caso)

    # Si las reglas deterministas son perfectas (1.0), el score semántico
    # no debe penalizar. Se aplica un piso mínimo proporcional.
    if puntaje_reglas >= 1.0:
        puntaje_semantico = max(puntaje_semantico, 0.85)

    indice = calcular_indice_fidelidad(puntaje_reglas, puntaje_semantico)
    estado = determinar_estado(hallazgos, indice, reglas)

    # Construir diagnóstico conciso
    diagnostico_partes = list(hallazgos)
    if analisis_semantico:
        diagnostico_partes.append(f"Análisis semántico: {analisis_semantico}")

    return ResultadoAuditoria(
        id_caso=caso["id_caso"],
        estado_transaccion=estado,
        indice_fidelidad=indice,
        diagnostico=" | ".join(diagnostico_partes),
    )


def imprimir_resultado(resultado: ResultadoAuditoria) -> str:
    """Genera la salida en el formato exacto requerido por el reto."""
    return "\n".join([
        f"Caso [{resultado.id_caso}]: {resultado.estado_transaccion}",
        f"- Índice de Fidelidad Analítica: {resultado.indice_fidelidad:.4f}",
        f"- Diagnóstico/Razón: {resultado.diagnostico}",
    ])


def main():
    """Función principal del Agente Auditor."""
    print("\nIniciando Agente Auditor (Agent A)...\n")

    try:
        reglas = cargar_reglas()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    try:
        casos = cargar_casos()
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    usar_ia = bool(os.getenv("OPENAI_API_KEY"))

    # Auditar cada caso y generar salida en formato exacto
    resultados = []
    for caso in casos:
        resultado = auditar_caso(caso, reglas, usar_ia=usar_ia)
        resultados.append(resultado)

    # Salida en consola con el formato exacto requerido
    print("")
    for resultado in resultados:
        print(imprimir_resultado(resultado))
        print("")

    # Guardar resultados en JSON (se actualiza en cada ejecución)
    ruta_salida = Path("resultados")
    ruta_salida.mkdir(exist_ok=True)
    archivo_json = ruta_salida / "resultados_auditoria.json"
    resultados_json = [
        {
            "id_caso": r.id_caso,
            "estado_transaccion": r.estado_transaccion,
            "indice_fidelidad": r.indice_fidelidad,
            "diagnostico": r.diagnostico,
        }
        for r in resultados
    ]
    archivo_json.write_text(json.dumps(resultados_json, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
