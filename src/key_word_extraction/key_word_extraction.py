from keybert import KeyBERT

doc = """
    How can reinforcement learning techniques be adapted to improve real-time decision-making in 
    robotic systems performing tasks with uncertain and dynamically changing environments, 
    such as SLAM-based navigation?
"""

kw_model = KeyBERT()
keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 2))
print(keywords)
# [('computer science', 0.6847), ('computer scientists', 0.6655),
#  ('computers computational', 0.5813), ('computer engineers', 0.5612),
#  ('study computers', 0.5589)]