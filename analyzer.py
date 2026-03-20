from groq import Groq
from dotenv import load_dotenv
import os
import re

load_dotenv()

def analyze_legal_document(text: str, language: str = "hindi") -> dict:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    lang_instruction = "Respond ONLY in simple Hindi (Devanagari script)." if language == "hindi" else "Respond ONLY in simple English."

    prompt = f"""
You are an expert legal assistant for Indian law.
Analyze this legal document and respond in this exact format:

SUMMARY:
(plain language summary)

RED FLAGS:
(list unfair or risky clauses)

NEXT STEPS:
(what the user should do)

RISK SCORE:
(number 1-10 only)

IPC SECTIONS:
(list relevant IPC/CrPC/legal sections that apply)

{lang_instruction}

Document:
\"\"\"{text[:4000]}\"\"\"
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    result = response.choices[0].message.content
    sections = {"summary": "", "red_flags": "", "next_steps": "", "risk_score": "5", "ipc_sections": ""}

    if "SUMMARY:" in result:
        sections["summary"] = result.split("SUMMARY:")[1].split("RED FLAGS:")[0].strip()
    if "RED FLAGS:" in result:
        sections["red_flags"] = result.split("RED FLAGS:")[1].split("NEXT STEPS:")[0].strip()
    if "NEXT STEPS:" in result:
        sections["next_steps"] = result.split("NEXT STEPS:")[1].split("RISK SCORE:")[0].strip()
    if "RISK SCORE:" in result:
        raw = result.split("RISK SCORE:")[1].split("IPC SECTIONS:")[0].strip()
        match = re.search(r'\d+', raw)
        sections["risk_score"] = match.group() if match else "5"
    if "IPC SECTIONS:" in result:
        sections["ipc_sections"] = result.split("IPC SECTIONS:")[1].strip()

    return sections


def analyze_courtroom_speech(speech: str, language: str = "hindi") -> dict:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    lang_instruction = "Respond in simple Hindi (Devanagari script)." if language == "hindi" else "Respond in simple English."

    prompt = f"""
You are an expert AI lawyer assistant for Indian courts.
Someone just said this in a courtroom or legal hearing:

\"\"\"{speech}\"\"\"

Your job:
1. Understand what was said
2. Suggest the best legal RESPONSE or COUNTER-ARGUMENT
3. Mention if an OBJECTION should be raised
4. Cite relevant IPC / CrPC / Indian law sections
5. Suggest what the lawyer should say next

{lang_instruction}

Respond in this exact format:

SITUATION:
(what is happening legally)

RESPONSE:
(what the lawyer should say out loud in court right now)

OBJECTION:
(should we object? if yes, exact objection wording. if no, write "No objection needed")

LEGAL SECTIONS:
(relevant IPC/CrPC sections)

STRATEGY:
(next move suggestion)
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )

    result = response.choices[0].message.content
    out = {"situation": "", "response": "", "objection": "", "legal_sections": "", "strategy": ""}

    if "SITUATION:" in result:
        out["situation"] = result.split("SITUATION:")[1].split("RESPONSE:")[0].strip()
    if "RESPONSE:" in result:
        out["response"] = result.split("RESPONSE:")[1].split("OBJECTION:")[0].strip()
    if "OBJECTION:" in result:
        out["objection"] = result.split("OBJECTION:")[1].split("LEGAL SECTIONS:")[0].strip()
    if "LEGAL SECTIONS:" in result:
        out["legal_sections"] = result.split("LEGAL SECTIONS:")[1].split("STRATEGY:")[0].strip()
    if "STRATEGY:" in result:
        out["strategy"] = result.split("STRATEGY:")[1].strip()

    return out