from transformers import T5ForConditionalGeneration, T5Tokenizer

class NamedEntityRecognition:
    def __init__(self):
        model_name = "google/flan-t5-base"
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=True)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def create_prompt(self, sentence:str):
        prompt = f"Extract the main topic that the user is asking about in the sentence: {sentence}"
        return prompt
    
    def get_result(self, sentence):
        prompt = self.create_prompt(sentence)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        output = self.model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)
        key_phrases = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return key_phrases

    def __call__(self, sentences):
        entities_per_sentence = []
        for sentence in sentences:
            entities = self.get_result(sentence)
            entities_per_sentence.append(entities)
        return entities_per_sentence

if __name__ == "__main__":
    ner = NamedEntityRecognition()
    sentences = [
                "What are the latest advancements in Computer Vision",
                "What are the latest advancements in Artificial Intelligence",
                "Is there anything new in Machine Learning that I should be aware of?"
                ]
    out = ner(sentences)
    print("Output", out)