import os, io
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from app.scenarios import SCENARIOS
from ai.qa import QASystem
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()

app = FastAPI(title="Know My Rights API", version="0.1.0")

# Lazy init
_qa = None
_clf_model = None
_scenario_texts = None
_scenario_ids = None

def get_qa():
    global _qa
    if _qa is None:
        _qa = QASystem()
    return _qa

def init_classifier():
    global _clf_model, _scenario_texts, _scenario_ids
    if _clf_model is None:
        _clf_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        _scenario_ids, _scenario_texts = [], []
        for sid, meta in SCENARIOS.items():
            _scenario_ids.append(sid)
            synonyms = " ".join(meta.get("synonyms", []))
            text = f"{meta['title']} {sid} {synonyms}"
            _scenario_texts.append(text)
    return _clf_model, _scenario_ids, _scenario_texts

class GuideResponse(BaseModel):
    scenario: str
    steps_markdown: str

class AskRequest(BaseModel):
    question: str
    k: int = 4

class AskResponseItem(BaseModel):
    text: str
    source: str

class AskResponse(BaseModel):
    results: List[AskResponseItem]

class ClassifyRequest(BaseModel):
    text: str

class ClassifyResponse(BaseModel):
    scenario: Optional[str]
    scores: List[float]

class SOSRequest(BaseModel):
    message: str
    contacts: List[str] = []
    location: Optional[str] = None

@app.get("/api/guide", response_model=GuideResponse)
def guide(scenario: str):
    scenario = scenario.lower().strip()
    if scenario not in SCENARIOS:
        raise HTTPException(404, f"Unknown scenario: {scenario}")
    md_path = SCENARIOS[scenario]["markdown"]
    if not os.path.exists(md_path):
        raise HTTPException(500, f"Content file not found: {md_path}")
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()
    return GuideResponse(scenario=scenario, steps_markdown=text)

@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    qa = get_qa()
    res = qa.ask(req.question, k=req.k)
    return AskResponse(results=[AskResponseItem(**r) for r in res])

@app.post("/api/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    model, ids, texts = init_classifier()
    q_emb = model.encode([req.text], normalize_embeddings=True)[0]
    s_embs = model.encode(texts, normalize_embeddings=True)
    sims = (s_embs @ q_emb).tolist()
    best_idx = int(np.argmax(sims))
    best_score = sims[best_idx]
    scenario = ids[best_idx] if best_score > 0.35 else None
    return ClassifyResponse(scenario=scenario, scores=sims)

@app.post("/api/sos")
def sos(req: SOSRequest):
    # Optional: send via Twilio if configured; otherwise just return payload
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER")

    payload = {
        "status": "dry_run",
        "sent_to": [],
        "message": req.message,
        "location": req.location,
    }

    if sid and token and from_number:
        try:
            from twilio.rest import Client
            client = Client(sid, token)
            for to in req.contacts[:3]:
                msg = client.messages.create(
                    body=f"SOS: {req.message} | Location: {req.location or 'N/A'}",
                    from_=from_number,
                    to=to,
                )
                payload["sent_to"].append({"to": to, "sid": msg.sid})
            payload["status"] = "sent"
        except Exception as e:
            payload["status"] = f"error: {e}"

    return payload
