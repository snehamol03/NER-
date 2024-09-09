import spacy
import nltk
from newsapi import NewsApiClient
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import requests
import torch

# Download required NLTK models and data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Initialize SpaCy and NewsAPI Client
nlp = spacy.load("en_core_web_sm")
api_key = 'e0b1a3c69fad4768ac220b2a5a501dba'  # Replace with your NewsAPI key
newsapi = NewsApiClient(api_key=api_key)

def fetch_news_article():
    # Fetch a news article using NewsAPI
    top_headlines = newsapi.get_top_headlines(language='en', page_size=1)
    article = top_headlines['articles'][0]
    title = article['title']
    content = article['content']
    print(f"Title: {title}\n")
    print(f"Content: {content}\n")
    return content

def nltk_named_entity_recognition(text):
    # Rule-based NER using NLTK
    sentences = sent_tokenize(text)
    nltk_entities = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        tags = pos_tag(tokens)
        tree = ne_chunk(tags, binary=False)
        for subtree in tree:
            if isinstance(subtree, nltk.Tree):
                entity = " ".join([token for token, pos in subtree.leaves()])
                nltk_entities.append((entity, subtree.label()))
    return nltk_entities

def spacy_named_entity_recognition(text):
    # ML-based NER using SpaCy
    doc = nlp(text)
    spacy_entities = [(ent.text, ent.label_) for ent in doc.ents]
    return spacy_entities

def compare_entities(nltk_entities, spacy_entities):
    print("\n--- NLTK Named Entities ---")
    for entity, label in nltk_entities:
        print(f"{entity} ({label})")

    print("\n--- SpaCy Named Entities ---")
    for entity, label in spacy_entities:
        print(f"{entity} ({label})")

    nltk_set = set(nltk_entities)
    spacy_set = set(spacy_entities)

    common_entities = nltk_set.intersection(spacy_set)
    print("\n--- Common Entities ---")
    for entity in common_entities:
        print(entity)

    only_nltk = nltk_set - spacy_set
    only_spacy = spacy_set - nltk_set

    print("\n--- Entities only in NLTK ---")
    for entity in only_nltk:
        print(entity)

    print("\n--- Entities only in SpaCy ---")
    for entity in only_spacy:
        print(entity)

# Main Program
if __name__ == "__main__":
    # Fetch a news article
    article = fetch_news_article()

    # Extract named entities using NLTK
    nltk_entities = nltk_named_entity_recognition(article)

    # Extract named entities using SpaCy
    spacy_entities = spacy_named_entity_recognition(article)

    # Compare results from both approaches
    compare_entities(nltk_entities, spacy_entities)
