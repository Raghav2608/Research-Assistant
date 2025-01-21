import urllib
import urllib.request
import xmltodict

if __name__ == "__main__":

    search_query = "all:attention"
    start = 0
    max_results = 3

    url = f'http://export.arxiv.org/api/query?search_query={search_query}&start={start}&max_results={max_results}&sortBy=relevance&sortOrder=descending'
    data = urllib.request.urlopen(url)
    result = data.read().decode("utf-8")

    # Convert the XML result into a dictionary of the results
    result = xmltodict.parse(result)
    entries = []
    for entry in result["feed"]["entry"]:
        paper_data = {
            "id": entry["id"],
            "title": entry["title"],
            "summary": entry["summary"].strip(),
            "authors": [author["name"] for author in entry["author"]],
            "published": entry["published"],
            "pdf_link": next(link["@href"] for link in entry["link"] if link.get("@title") == "pdf")
        }
        entries.append(paper_data)
    
    for paper in entries:
        print("ID:", paper["id"])
        print("Title:", paper["title"])
        print("Summary:", paper["summary"])
        print("Authors:", paper["authors"])
        print("Published:", paper["published"])
        print("PDF Link:", paper["pdf_link"])
        print("\n")