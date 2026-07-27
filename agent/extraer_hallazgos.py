import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0
)


def extraer_hallazgos(papers: list[dict]) -> list[dict]:
    resultados = []
    for paper in papers:
        prompt = f"""Titulo: {paper['titulo']}
Resumen: {paper['resumen']}

Extrae de este resumen de paper cientifico dos cosas, en este formato exacto:

METODOLOGIA: [una oracion describiendo que metodo o enfoque usaron]
RESULTADO: [una oracion describiendo el hallazgo o resultado principal]"""

        respuesta = llm.invoke(prompt)
        texto = respuesta.content.strip()

        metodologia = ""
        resultado = ""
        for linea in texto.split("\n"):
            if linea.startswith("METODOLOGIA:"):
                metodologia = linea.replace("METODOLOGIA:", "").strip()
            elif linea.startswith("RESULTADO:"):
                resultado = linea.replace("RESULTADO:", "").strip()

        paper_con_hallazgos = paper.copy()
        paper_con_hallazgos["metodologia"] = metodologia
        paper_con_hallazgos["resultado"] = resultado
        resultados.append(paper_con_hallazgos)

    return resultados


if __name__ == "__main__":
    from buscar_papers import buscar_papers
    from filtrar_relevancia import filtrar_relevancia

    tema = "self-supervised learning cardiac MRI"
    papers = buscar_papers(tema, cantidad=5)
    relevantes = filtrar_relevancia(papers, tema)

    print(f"Extrayendo hallazgos de {len(relevantes)} papers relevantes...\n")
    con_hallazgos = extraer_hallazgos(relevantes)

    for p in con_hallazgos:
        print(f"Titulo: {p['titulo']}")
        print(f"  Metodologia: {p['metodologia']}")
        print(f"  Resultado: {p['resultado']}")
        print()
