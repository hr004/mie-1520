import spacy

# TODO: try different tokenizers and add more preprocessing
nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner"])


def spacy_tokenizer(sent):
    return [w.text.lower() for w in nlp.tokenizer(sent)]
