from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List

class TopicExtractor:
    """
    Class for extracting topics from sentences using a pre-trained T5 model.
    """
    def __init__(self):
        model_name = "google/flan-t5-large"
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=True)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def create_prompt(self, sentence:str) -> str:
        """
        Creates a prompt, embedding the input sentence into it.

        Args:
            sentence (str): The sentence to embed into the prompt.
        """
        prompt = f"Extract the main topic that the user is asking about in the sentence: {sentence}"
        return prompt
    
    def get_result(self, sentence:str) -> str:
        """
        Finds the topic that the user is talking about inside of the sentence.
        For example:
        "What are the latest advancements in Computer Vision"
        Should return:
        "Computer Vision"

        Args:
            sentence (str): A sentence containing the topic to extract.
        """
        prompt = self.create_prompt(sentence)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        output = self.model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)
        key_phrases = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return key_phrases

    def __call__(self, sentences:List[str]) -> List[str]:
        """
        Processes a list of sentences and returns a list of topics for each sentence.

        Args:
            sentences (List[str]): A list of sentences to be processed.
        """
        topics_per_sentence = []
        for sentence in sentences:
            topics = self.get_result(sentence)
            topics_per_sentence.append(topics)
        return topics_per_sentence

if __name__ == "__main__":
    topic_extractor = TopicExtractor()
    sentences = [
                "What are the latest advancements in Computer Vision",
                "What are the latest advancements in Artificial Intelligence",
                "Is there anything new in Machine Learning that I should be aware of?",
                "What are the current trends in Natural Language Processing?",
                "Can you tell me about the recent innovations in Data Science?",
                "What's the latest research in Deep Reinforcement Learning?",
                "What are the top use cases of Generative AI?",
                "What industries are benefiting the most from AI advancements?",
                "How is Machine Learning transforming healthcare",
                "Are there any advancements in transformer models lately?"
                ]
    out = topic_extractor(sentences)
    
    for i,(prompt, topic_answer) in enumerate(zip(sentences, out)):
        print(f"Prompt number: {i}")
        print(f"Prompt: {prompt}")
        print(f"Answer: {topic_answer}")
        print("\n")