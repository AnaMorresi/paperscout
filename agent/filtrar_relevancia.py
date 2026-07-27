import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0
)


def filtrar_relevancia(papers: list[dict], tema: str) -> list[dict]:
    relevantes = []
    for paper in papers:
        prompt = f"""Tema de investigacion: {tema}

Titulo: {paper['titulo']}
Resumen: {paper['resumen']}

Un paper es relevante SOLO SI el resumen trata explicitamente sobre imagenes medicas, resonancia magnetica (MRI) o el corazon/cardiaco, Y ADEMAS aprendizaje autosupervisado (self-supervised learning).

Si el paper es sobre aprendizaje autosupervisado en otro dominio (imagenes genericas, clustering, robots, texto, etc.) SIN relacion con imagenes medicas o cardiacas, la respuesta es NO.

Respondé unicamente con SI o NO, sin explicacion."""


        respuesta = llm.invoke(prompt)
        veredicto = respuesta.content.strip().upper()

        print(f"{'OK ' if veredicto.startswith('SI') else 'NO '} {paper['titulo'][:60]}... -> {veredicto}")

        if veredicto.startswith("SI"):
            relevantes.append(paper)

    return relevantes


if __name__ == "__main__":
    from buscar_papers import buscar_papers

    tema = "self-supervised learning cardiac MRI"
    papers = buscar_papers(tema, cantidad=5)

    print(f"Filtrando {len(papers)} papers...\n")
    relevantes = filtrar_relevancia(papers, tema)

    print(f"\n{len(relevantes)} de {len(papers)} papers son relevantes:\n")
    for p in relevantes:
        print(f"- {p['titulo']}")
