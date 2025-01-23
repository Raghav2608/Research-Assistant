from src.data_ingestion.arxiv.utils import fetch_arxiv_papers, parse_papers, summarise_papers

if __name__ == "__main__":

    search_query = "all:attention"
    start = 0
    max_results = 3

    xml_papers = fetch_arxiv_papers(search_query, start, max_results)
    entries = parse_papers(xml_papers)
    summarising_strings = summarise_papers(entries)
    for i, res in enumerate(summarising_strings):
        print(f"Paper: {i+1}")
        print(res)
        print("Number of characters:", len(res))
        print("\n")