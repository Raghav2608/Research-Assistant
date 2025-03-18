import urllib.parse

def clean_search_query(search_query:str) -> str:
    """
    Cleans the search query by replacing spaces with '+'.

    Args:
        search_query (str): The search query to clean.
    """
    return urllib.parse.quote_plus(search_query)