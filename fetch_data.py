import urllib
import urllib.request

if __name__ == "__main__":

    search_query = "all:attention"
    start = 0
    max_results = 3

    url = f'http://export.arxiv.org/api/query?search_query={search_query}&start={start}&max_results={max_results}&sortBy=relevance&sortOrder=descending'
    data = urllib.request.urlopen(url)
    result = data.read().decode("utf-8")
    print(result)
    print(type(result))