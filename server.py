import sqlite3
import json
import os
import traceback
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

app = FastAPI(title="KR Guidance AI Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

DB_PATH = "eapcet_cutoffs.db"

class ChatMessage(BaseModel):
    role: str
    content: str

class UserQuery(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []


def query_database(college: str, branch: str, category: str, gender: str, rank: int, year: int = 2025):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    clean_cat = category.upper().replace("-", "_").strip()
    clean_gender = gender.upper().strip()

    if clean_cat.endswith("_BOYS") or clean_cat.endswith("_GIRLS"):
        target_category = clean_cat
    else:
        target_category = f"{clean_cat}_{clean_gender}"

    cursor.execute("""
    SELECT phase, closing_rank FROM cutoffs
    WHERE college_code = ? AND branch_code = ? AND category = ? AND year = ?
    ORDER BY CASE phase
        WHEN 'Phase 1' THEN 1
        WHEN 'Phase 2' THEN 2
        WHEN 'Final Phase' THEN 3
        ELSE 4
    END
    """, (college.upper(), branch.upper(), target_category, year))

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return None

    results = {}
    qualified = False
    best_phase = None

    for phase, closing in rows:
        is_safe = rank <= closing
        margin = closing - rank
        results[phase] = {
            "closing_rank": closing,
            "safe": is_safe,
            "margin": margin
        }
        if is_safe and not qualified:
            qualified = True
            best_phase = phase

    return {
        "candidate_rank": rank,
        "college": college.upper(),
        "branch": branch.upper(),
        "category": target_category,
        "year": year,
        "phases": results,
        "is_safe": qualified,
        "secured_in": best_phase
    }


EXTRACTION_SYSTEM_PROMPT = """
You are an expert admission parameter extractor for TG EAPCET.
Read the entire conversation history and the latest user message to combine all candidate details (supports English and Telugu/Tanglish).

College Code Mapping:
- GRIET / Gokaraju -> GRRR
- VNR / VJIET -> VNRV
- CBIT -> CBIT
- Vardhaman -> VMEG
- Vasavi -> VCE
- CVR -> CVRH
- BVRIT / BVRIT Narsapur -> BVRI
- MGIT -> MGIT
- MVSR -> MVSR
- KMIT -> KMIT

Branch Mapping:
- CSE / Computer Science -> CSE
- CSD / Data Science -> CSD
- CSM / AIML / AI & ML -> CSM
- IT / Information Technology -> INF
- ECE -> ECE
- EEE -> EEE

Gender Mapping:
- "maa babu", "abbayi", "boy", "son", "male" -> BOYS
- "maa ammayi", "papa", "girl", "daughter", "female" -> GIRLS
- Default if unspecified -> BOYS

Category Mapping:
- OC, BC_A, BC_B, BC_C, BC_D, BC_E, SC, ST, EWS

Output strictly valid JSON with these keys:
{
  "is_complete": true,
  "missing_fields": [],
  "college": "CBIT",
  "branch": "ECE",
  "category": "ST",
  "gender": "BOYS",
  "rank": 18000,
  "year": 2025
}
"""

@app.get("/")
def health_check():
    return {"status": "ok", "service": "KR Guidance AI Engine"}

@app.post("/chat")
def chat_endpoint(payload: UserQuery):
    try:
        llm_messages = [{"role": "system", "content": EXTRACTION_SYSTEM_PROMPT}]
        if payload.history:
            for msg in payload.history[-6:]:
                llm_messages.append({"role": msg.role, "content": msg.content})
        llm_messages.append({"role": "user", "content": payload.message})

        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            response_format={"type": "json_object"},
            messages=llm_messages,
            temperature=0.0
        )

        raw_content = response.choices[0].message.content.strip()
        if raw_content.startswith("```"):
            raw_content = raw_content.split("\n", 1)[1].rsplit("\n", 1)[0].strip()

        extracted = json.loads(raw_content)

        college = extracted.get("college")
        branch = extracted.get("branch")
        category = extracted.get("category")
        rank_raw = extracted.get("rank")

        missing_prompts = []
        if not rank_raw:
            missing_prompts.append("- EAPCET Rank (e.g., 18000)")
        if not college:
            missing_prompts.append("- Target College (e.g., CBIT, GRIET, VNR)")
        if not branch:
            missing_prompts.append("- Target Branch (e.g., CSE, CSD, CSM, ECE, IT)")
        if not category:
            missing_prompts.append("- Reservation Category & Gender (e.g., BC-B Boy, SC Girl, OC Boy)")

        if missing_prompts:
            return {
                "type": "slot_filling",
                "reply": "Namaskaram! Mee EAPCET admission chances analyze cheyataniki kinda details ivvandi:\n" + "\n".join(missing_prompts)
            }

        try:
            rank = int(str(rank_raw).replace(",", "").strip())
        except ValueError:
            return {
                "type": "slot_filling",
                "reply": "Please provide a valid numerical rank (e.g., 18000)."
            }

        gender = extracted.get("gender", "BOYS")
        year = int(extracted.get("year") or 2025)

        db_result = query_database(
            college=college,
            branch=branch,
            category=category,
            gender=gender,
            rank=rank,
            year=year
        )

        if not db_result:
            return {
                "type": "not_found",
                "reply": f"Official closing ranks for {college} - {branch} ({category} {gender}) could not be found in the {year} database. Please check the college/branch code."
            }

        phases = db_result["phases"]
        p1 = phases.get("Phase 1", {})
        p2 = phases.get("Phase 2", {})
        p_final = phases.get("Final Phase", {})

        if db_result["is_safe"]:
            status_str = f"Safe / High Probability (Secured in {db_result['secured_in']})"
        else:
            final_margin = abs(p_final.get("margin", 9999))
            if final_margin <= 1000:
                status_str = f"Borderline / Competitive (Missed Final Phase by {final_margin} ranks)"
            else:
                status_str = f"Reach / Tough (Exceeds Final Phase cutoff by {final_margin} ranks)"

        output = f"""Candidate Rank: {db_result['candidate_rank']}
Target: {db_result['college']} - {db_result['branch']} ({db_result['category']})
Official Closing Ranks ({year}):
- Phase 1: {p1.get('closing_rank', 'N/A')}
- Phase 2: {p2.get('closing_rank', 'N/A')}
- Final Phase: {p_final.get('closing_rank', 'N/A')}
Status: {status_str}

For personalized choice filling or in person sessions in Siddipet with KR Sir, visit krguidance.in."""

        return {
            "type": "result",
            "reply": output,
            "data": db_result
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "type": "error",
            "reply": "Server is busy processing requests. Please try again in a moment.",
            "error_detail": str(e)
        }