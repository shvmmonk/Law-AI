from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import pdfplumber
import io
import os
from analyzer import analyze_legal_document, analyze_courtroom_speech
from groq import Groq

load_dotenv()

app = FastAPI(title="Legal Aid AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root():
    return {"message": "Legal Aid AI backend is running!"}

@app.get("/health")
def health():
    return {"status": "ok", "groq_key_loaded": bool(os.getenv("GROQ_API_KEY"))}

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...), language: str = Form("hindi")):
    contents = await file.read()
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(contents)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except:
        return {"error": "Could not read this file. Please upload a valid PDF."}
    if not text.strip():
        return {"error": "Could not extract text. Try a clearer PDF."}
    result = analyze_legal_document(text, language)
    return {"filename": file.filename, "analysis": result}

class ChatRequest(BaseModel):
    question: str
    context: str
    language: str = "hindi"

@app.post("/chat")
async def chat(req: ChatRequest):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    lang = "Respond in simple Hindi (Devanagari)." if req.language == "hindi" else "Respond in simple English."
    prompt = f"""You are a legal assistant. Based on this document analysis:
{req.context}
Answer: {req.question}
{lang} Keep it short and helpful."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return {"answer": response.choices[0].message.content}

class CourtroomRequest(BaseModel):
    speech: str
    language: str = "hindi"

@app.post("/courtroom")
async def courtroom(req: CourtroomRequest):
    result = analyze_courtroom_speech(req.speech, req.language)
    return result