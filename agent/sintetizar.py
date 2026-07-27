import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0.3
)


def sintetizar(papers: list[dict], tema: str) -> str:
    papers_texto = ""
    for i, p in enumerate(papers, 1):
        papers_texto += f"""
[{i}] {p['titulo']}
Metodologia: {p['metodologia']}
Resultado: {p['resultado']}
Link: {p['link']}
"""

    prompt = f"""Tema de investigacion: {tema}

A continuacion hay un listado de papers relevantes con su metodologia y resultado principal:
{papers_texto}

Escribi un resumen ejecutivo de 2 a 3 parrafos que sintetice los hallazgos principales de estos papers en relacion al tema. Cita cada paper usando su numero entre corchetes, por ejemplo [1], cuando menciones su hallazgo especifico. Al final, incluí una lista de referencias con el numero, titulo y link de cada paper."""

    respuesta = llm.invoke(prompt)
    return respuesta.content.strip()


if __name__ == "__main__":
    from buscar_papers import buscar_papers
    from filtrar_relevancia import filtrar_relevancia
    from extraer_hallazgos import extraer_hallazgos

    tema = "self-supervised learning cardiac MRI"
    papers = buscar_papers(tema, cantidad=5)
    relevantes = filtrar_relevancia(papers, tema)
    con_hallazgos = extraer_hallazgos(relevantes)

    print("Generando sintesis final...\n")
    resumen = sintetizar(con_hallazgos, tema)

    print(resumen)
