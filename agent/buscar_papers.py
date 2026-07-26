import arxiv

def buscar_papers(tema: str, cantidad: int = 5):
    client = arxiv.Client()
    busqueda = arxiv.Search(
        query=f"all:{tema}",
        max_results=cantidad,
        sort_by=arxiv.SortCriterion.Relevance
    )

    resultados = []
    for paper in client.results(busqueda):
        resultados.append({
            "titulo": paper.title,
            "autores": [autor.name for autor in paper.authors],
            "resumen": paper.summary,
            "fecha": paper.published.strftime("%Y-%m-%d"),
            "link": paper.entry_id
        })
    return resultados


if __name__ == "__main__":
    tema = "self-supervised learning cardiac MRI"
    papers = buscar_papers(tema, cantidad=5)

    print(f"Encontrados {len(papers)} papers sobre: {tema}\n")
    for i, p in enumerate(papers, 1):
        print(f"{i}. {p['titulo']}")
        print(f"   Autores: {', '.join(p['autores'])}")
        print(f"   Fecha: {p['fecha']}")
        print(f"   Link: {p['link']}")
        print(f"   Resumen: {p['resumen'][:200]}...")
        print()
