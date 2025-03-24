import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline, T5ForConditionalGeneration, T5Tokenizer
import spacy
import json

# Load fine-tuned NER model
fine_tuned_model_path = "./trained_model"
ner_model = AutoModelForTokenClassification.from_pretrained(fine_tuned_model_path)
ner_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)

# Load T5 model for summarization
t5_model_name = "t5-small"  # Can be replaced with a fine-tuned T5 model if needed
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

# Load spaCy model for relationship mapping
nlp = spacy.load("en_core_web_sm")

# Named Entity Recognition (NER) using fine-tuned model
def extract_key_concepts(text):
    ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")
    entities = ner_pipeline(text)
    key_concepts = [entity['word'] for entity in entities]
    return key_concepts

# Text Summarization using T5
def summarize_text(text):
    input_ids = t5_tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = t5_model.generate(input_ids.input_ids, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Relationship Mapping using spaCy
def extract_relationships(text):
    doc = nlp(text)
    relationships = []
    for ent1 in doc.ents:
        for ent2 in doc.ents:
            if ent1 != ent2:
                relationships.append({"entity_1": ent1.text, "relation": "related_to", "entity_2": ent2.text})
    return relationships

# Mind Map JSON Generation
def generate_mind_map(text):
    key_concepts = extract_key_concepts(text)
    summary = summarize_text(text)
    relationships = extract_relationships(text)

    mind_map = {
        "summary": summary,
        "key_concepts": key_concepts,
        "relationships": relationships
    }

    return json.dumps(mind_map, indent=4)

# Example Usage
input_text = """The global financial crisis was still three years away, but in 2005 a financial pall was already hanging over Bradwell Grove Estate in the Cotswolds. Rising oil prices – affecting in particular chemical fertiliser, had rocketed while crop prices stayed low. Manager of the 3500-acre farm, Charles Hunter-Smart, found that “revenues were not covering high running costs,” largely due to the substantial cost of the chemicals used to drive crop productivity"""
mind_map_json = generate_mind_map(input_text)
print(mind_map_json)