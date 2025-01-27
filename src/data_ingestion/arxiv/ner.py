import spacy

class NamedEntityRecognition:
    def __init__(self):
        self.model = spacy.load("en_core_web_sm")

    def __call__(self, sentences):
        return sentences

if __name__ == "__main__":
    ner = NamedEntityRecognition()
    sentences = ["What are the latest advancements in deep learning"]
    out = ner(sentences)
    print("Output", out)