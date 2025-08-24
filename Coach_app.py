import re
import json
import random
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List

import chainlit as cl
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)
sid = SentimentIntensityAnalyzer()

MEMORY_PATH = Path("coach_memory.json")
QUOTES_PATH = Path("quotes.json")

DEFAULT_QUOTES = [
    "Small steps every day lead to big changes.",
    "You don’t have to be extreme, just consistent.",
    "Action cures fear — start tiny.",
    "Your future is decided by what you do today, not tomorrow.",
    "Rest is part of the process."
]

SUGGESTIONS: Dict[str, List[str]] = {
    "stressed": [
        "2-minute box breathing: inhale 4s, hold 4s, exhale 4s, hold 4s.",
        "Write down 3 worries, then one tiny action for each.",
        "Stretch your neck and shoulders for 60 seconds."
    ],
    "tired": [
        "Stand up, sip water, and do 10 slow squats.",
        "2-minute sunlight break at a window/balcony.",
        "Swap tasks: do a 5-minute easy win to regain momentum."
    ],
    "sad": [
        "Text a friend one nice thing.",
        "Play a comforting song and breathe for a minute.",
        "Step outside for 3 minutes and name 5 things you see."
    ],
    "happy": [
        "Bank the energy: tackle a 10-minute task you've delayed.",
        "Share gratitude: write 2 lines to someone you appreciate.",
        "Queue a playlist and do a focused 15-minute sprint."
    ],
    "neutral": [
        "Pick one 10-minute task and set a timer.",
        "Tidy your workspace for 3 minutes.",
        "Drink water and plan your next 30 minutes."
    ]
}

MOOD_KEYWORDS: Dict[str, List[str]] = {
    "stressed": ["overwhelmed", "anxious", "stress", "pressure", "panic"],
    "tired": ["sleepy", "exhausted", "fatigue", "drained", "tired"],
    "sad": ["down", "blue", "depressed", "lonely", "sad"],
    "happy": ["great", "excited", "joy", "grateful", "happy"]
}

ALL_MOODS = ["stressed", "tired", "sad", "happy", "neutral"]

TOKEN_RE = re.compile(r"[a-zA-Z']+")

def load_memory():
    if MEMORY_PATH.exists():
        try:
            return json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"recent_moods": []}

def save_memory(mem: Dict):
    MEMORY_PATH.write_text(json.dumps(mem, ensure_ascii=False, indent=2), encoding="utf-8")

def remember_mood(mood: str):
    mem = load_memory()
    moods = deque(mem.get("recent_moods", []), maxlen=10)
    moods.append({"mood": mood, "ts": datetime.utcnow().isoformat()})
    mem["recent_moods"] = list(moods)
    save_memory(mem)

def recent_mood_bias(mood: str) -> float:
    """Penalty if the same mood occurred in the last 60 minutes (reduces repetition)."""
    mem = load_memory()
    last_same = [m for m in mem.get("recent_moods", []) if m["mood"] == mood]
    if not last_same:
        return 0.0
    try:
        last_ts = datetime.fromisoformat(last_same[-1]["ts"])
        if datetime.utcnow() - last_ts < timedelta(minutes=60):
            return -0.2
    except Exception:
        pass
    return 0.0

def detect_mood(text: str):
    text_l = text.lower()
    scores = {k: 0.0 for k in ALL_MOODS}
    reasons = []

    for mood, kws in MOOD_KEYWORDS.items():
        hits = [kw for kw in kws if kw in text_l]
        if hits:
            scores[mood] += 0.6
            reasons.append(f"keywords for **{mood}**: {', '.join(hits)}")

    comp = sid.polarity_scores(text_l)["compound"]
    if comp >= 0.5:
        scores["happy"] += 0.5
        reasons.append(f"positive sentiment (compound={comp:.2f}) → **happy**")
    elif comp >= 0.05:
        scores["happy"] += 0.2
        reasons.append(f"slightly positive sentiment (compound={comp:.2f}) → **happy**")
    elif comp <= -0.5:
        scores["sad"] += 0.5
        reasons.append(f"negative sentiment (compound={comp:.2f}) → **sad**")
    elif comp <= -0.05:
        scores["stressed"] += 0.2
        reasons.append(f"slightly negative sentiment (compound={comp:.2f}) → **stressed**")
    else:
        scores["neutral"] += 0.2
        reasons.append(f"neutral sentiment (compound={comp:.2f}) → **neutral**")

    for m in ALL_MOODS:
        adj = recent_mood_bias(m)
        scores[m] += adj
        if adj < 0:
            reasons.append(f"recently seen **{m}** → applying penalty")

    mood = max(scores, key=scores.get)
    return mood, comp, reasons, scores

def pick_quote() -> str:
    if QUOTES_PATH.exists():
        try:
            quotes = json.loads(QUOTES_PATH.read_text(encoding="utf-8"))
            if isinstance(quotes, list) and quotes:
                random.shuffle(quotes)
                return quotes[0]
        except Exception:
            pass
    q = DEFAULT_QUOTES[:]
    random.shuffle(q)
    return q[0]

def suggest_for_mood(mood: str, n=3):
    pool = SUGGESTIONS.get(mood, SUGGESTIONS["neutral"]).copy()
    random.shuffle(pool)
    tips = pool[:n]
    return tips

@cl.on_chat_start
async def start():
    welcome = (
        "**AI Life Coach Agent**\n\n"
        "Tell me how you're feeling (e.g., *tired, stressed, happy*), and I'll suggest quick, helpful actions.\n\n"
        "**Commands**\n"
        "• `why` → explain my mood detection\n"
        "• `correct: <mood>` → override (moods: stressed, tired, sad, happy, neutral)\n"
        "• `reset` → clear recent mood memory\n"
        "• `save` → save memory to disk\n"
    )
    await cl.Message(content=welcome).send()

@cl.on_message
async def handle_message(message: str):
    text = message.strip()

    if text.lower() == "save":
        save_memory(load_memory()) 
        await cl.Message(content="Memory saved to disk.").send()
        return

    if text.lower() == "reset":
        save_memory({"recent_moods": []})
        await cl.Message(content="Cleared recent mood memory.").send()
        return

    if text.lower().startswith("correct:"):
        mood = text.split(":", 1)[1].strip().lower()
        if mood not in ALL_MOODS:
            await cl.Message(content=f"Unknown mood `{mood}`. Use one of: {', '.join(ALL_MOODS)}").send()
            return
        remember_mood(mood)
        tips = suggest_for_mood(mood)
        quote = pick_quote()
        reply = (
            f"Thanks — updated mood to **{mood}**.\n\n"
            f"**Suggestions:**\n" + "\n".join([f"- {t}" for t in tips]) +
            f"\n\n**Quote:** _{quote}_"
        )
        await cl.Message(content=reply).send()
        return

    if text.lower() == "why":
        await cl.Message(content="Send a sentence describing how you feel (e.g., 'I'm overwhelmed and anxious'). I'll explain my reasoning next time.").send()
        return

    mood, comp, reasons, scores = detect_mood(text)
    remember_mood(mood)
    tips = suggest_for_mood(mood)
    quote = pick_quote()

    resp = (
        f"**Detected mood:** **{mood}**  \n"
        f"**Sentiment (compound):** {comp:.2f}\n\n"
        f"**Suggestions:**\n" + "\n".join([f"- {t}" for t in tips]) +
        f"\n\n**Quote:** _{quote}_\n\n"
        f"Type `why` if you want my reasoning, or `correct: <mood>` to override."
    )
    await cl.Message(content=resp).send()

    if "why" in text.lower():
        expl = (
            "**Why I chose this mood**\n"
            + "\n".join([f"- {r}" for r in reasons])
            + "\n\n**Score snapshot:** "
            + ", ".join([f"{k}={scores[k]:.2f}" for k in ALL_MOODS])
        )
        await cl.Message(content=expl).send()
