import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline, T5ForConditionalGeneration, T5Tokenizer
import spacy
import json

# Initialize FastAPI app
app = FastAPI()

# Get HF Token from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Hugging Face token not found! Set HF_TOKEN as an environment variable.")

# Load fine-tuned NER model from Hugging Face
model_name = "nitinsri/mira-v1"

try:
    ner_model = AutoModelForTokenClassification.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    ner_tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
except Exception as e:
    raise RuntimeError(f"Error loading NER model: {e}")

# Load T5 model for summarization
t5_model_name = "t5-small"
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

# Load spaCy model for relationship mapping
nlp = spacy.load("en_core_web_sm")

# Define input schema
class TextRequest(BaseModel):
    text: str

# Named Entity Recognition (NER)
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

# API Endpoint
@app.post("/analyze")
async def analyze_text(request: TextRequest):
    try:
        key_concepts = extract_key_concepts(request.text)
        summary = summarize_text(request.text)
        relationships = extract_relationships(request.text)

        mind_map = {
            "summary": summary,
            "key_concepts": key_concepts,
            "relationships": relationships
        }

        return mind_map
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run locally: uvicorn main:app --reload
