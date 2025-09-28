from fastapi import FastAPI
from pydantic import BaseModel
from bigram_model import BigramModel  
import spacy
nlp = spacy.load("en_core_web_md")   

app = FastAPI()

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas.",
    "This is another example sentence.",
    "We are generating text based on bigram probabilities.",
    "Bigram models are simple but effective."
]

text = " ".join(corpus)

model = BigramModel(text)

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

class EmbeddingRequest(BaseModel):
    word: str

@app.get("/")
def read_root():
    return {"message": "Hello, World"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    return {"generated_text": model.generate_text(request.start_word, request.length)}

@app.post("/embedding")
def get_embedding(req: EmbeddingRequest):
    text = req.word.strip()
    if not text:
        return {"error": "empty input"}
    doc = nlp(text)
    if len(doc) == 0:
        return {"error": "no tokens"}
    vec = doc[0].vector  # 取第一个 token 的向量
    return {"word": req.word, "embedding": vec.tolist()}

#/opt/anaconda3/envs/GenAI/bin/python -m uvicorn app.main:app --reload


