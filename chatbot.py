"""
AliChatbot - Friendly, context-aware, self-learning chatbot for FlexiPDF.

Features:
- Unified memory.json that holds:
    - user metadata (name, country, city, language)
    - facts (user-taught facts and general facts)
    - relationships (friend/girlfriend/etc.)
    - conversations (timestamped)
    - flexipdf_knowledge (help items about the app)
- Short-term context (last 20 messages) for contextual replies
- Auto-learning from patterns like:
    "My name is Ali", "My city is Kohat", "Love is emotion"
- Teach FlexiPDF by:
    Ali learn 'pdf split' means divide PDF into multiple files
- Safe loading + auto-repair for older memory.json shapes
- reset_memory() helper method to wipe memory
"""

import json
import os
import re
from datetime import datetime
from collections import deque
from typing import Optional

try:
    # Optional: use sentence-transformers for later embedding-based retrieval if available
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import faiss
    HAS_EMBED = True
except Exception:
    HAS_EMBED = False


class AliChatbot:
    def __init__(self, memory_path: str = "data/memory.json", ctx_size: int = 20):
        self.memory_path = memory_path
        os.makedirs(os.path.dirname(memory_path), exist_ok=True)

        # context memory (short-term)
        self.context = deque(maxlen=ctx_size)

        # default data structure
        self.data = {
            "user_name": None,
            "country": None,
            "city": None,
            "language": None,
            "facts": {},              # key -> value (free facts)
            "relationships": {},      # e.g. girlfriend -> Rubab
            "conversations": [],      # list of {user, ai, time}
            "flexipdf_knowledge": {}, # app-help items
            "meta": {"created_at": None}
        }

        # load from disk or create default
        self._load_or_init_memory()

        # optional embedding model for future semantic search (not required)
        self.embedding_model = None
        self.faiss_index = None
        if HAS_EMBED:
            try:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                self._build_faiss_index()
            except Exception:
                self.embedding_model = None
                self.faiss_index = None

        # seed helpful FlexiPDF knowledge (merge with existing)
        self._seed_flexipdf_knowledge()

    # -------------------------
    # Memory load / save / repair
    # -------------------------
    def _load_or_init_memory(self):
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    # repair - if it's a list or otherwise malformed, wrap it
                    if not isinstance(raw, dict):
                        self.data["conversations"] = raw if isinstance(raw, list) else []
                    else:
                        for k in self.data.keys():
                            if k in raw and isinstance(raw[k], type(self.data[k])):
                                self.data[k] = raw[k]
                            elif k in raw:
                                if k == "conversations" and isinstance(raw[k], list):
                                    self.data[k] = raw[k]
                        if "meta" in raw and isinstance(raw["meta"], dict):
                            self.data["meta"].update(raw["meta"])
            except Exception as e:
                print(f"[Ali] memory load error — resetting memory.json ({e})")
                self.data["conversations"] = []
                self._save_memory()
        else:
            self.data["meta"]["created_at"] = datetime.now().isoformat(timespec="seconds")
            self._save_memory()

    def _save_memory(self):
        try:
            with open(self.memory_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Ali] failed to save memory.json: {e}")

    def reset_memory(self):
        """Wipe memory (useful for testing)."""
        self.data = {
            "user_name": None,
            "country": None,
            "city": None,
            "language": None,
            "facts": {},
            "relationships": {},
            "conversations": [],
            "flexipdf_knowledge": {},
            "meta": {"created_at": datetime.now().isoformat(timespec="seconds")}
        }
        self.context.clear()
        self._save_memory()
        if self.embedding_model:
            self._build_faiss_index()
        return "Memory reset — fresh start ✨"

    # -------------------------
    # Faiss index (optional)
    # -------------------------
    def _build_faiss_index(self):
        if not self.embedding_model:
            self.faiss_index = None
            return
        texts = [c.get("user", "") for c in self.data.get("conversations", []) if c.get("user")]
        if not texts:
            self.faiss_index = None
            return
        try:
            embeddings = np.array(self.embedding_model.encode(texts)).astype("float32")
            dim = embeddings.shape[1]
            idx = faiss.IndexFlatL2(dim)
            idx.add(embeddings)
            self.faiss_index = {"index": idx, "texts": texts}
        except Exception as e:
            print(f"[Ali] failed building faiss index: {e}")
            self.faiss_index = None

    def _semantic_search(self, query: str, k: int = 3):
        if not (self.embedding_model and self.faiss_index):
            return []
        q_emb = np.array(self.embedding_model.encode([query])).astype("float32")
        D, I = self.faiss_index["index"].search(q_emb, k)
        results = []
        for i in I[0]:
            if i < len(self.faiss_index["texts"]):
                results.append(self.faiss_index["texts"][i])
        return results

    # -------------------------
    # FlexiPDF default knowledge
    # -------------------------
    def _seed_flexipdf_knowledge(self):
        defaults = {
            "pdf_to_word": "Convert PDF files into editable Word (.docx) documents.",
            "word_to_pdf": "Convert Word (.docx) files into clean, printable PDF files.",
            "pdf_to_image": "Export pages of a PDF into high-quality image files (PNG/JPG).",
            "images_to_pdf": "Merge multiple images into a single PDF document.",
            "merge_pdfs": "Combine multiple PDF files into a single PDF.",
            "split_pdf": "Split a PDF into multiple smaller PDF files by page ranges.",
            "compress_pdf": "Reduce PDF file size while keeping acceptable quality.",
            "ocr_pdf": "(If available) Use OCR to recognize text inside scanned PDF pages.",
            "ai_assistant": "I’m Ali — the built-in AI assistant that helps with FlexiPDF tasks and answers questions."
        }
        for k, v in defaults.items():
            if k not in self.data.get("flexipdf_knowledge", {}):
                self.data["flexipdf_knowledge"][k] = v
        self._save_memory()

    # -------------------------
    # Internal helpers: remember conversation
    # -------------------------
    def _remember_conversation(self, user_text: str, ai_text: str):
        entry = {
            "user": user_text,
            "ai": ai_text,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        if "conversations" not in self.data:
            self.data["conversations"] = []
        self.data["conversations"].append(entry)
        self.context.append(entry)
        self._save_memory()
        if self.embedding_model:
            self._build_faiss_index()

    # -------------------------
    # All other methods as in your previous code...
    # Learning patterns, queries, fallback, get_response
    # (Keep everything same as your previous chatbot.py)


    # -------------------------
    # Learning patterns (structured)
    # -------------------------
    def _learn_structured_fact(self, text: str) -> Optional[str]:
        """
        Detect and store structured user facts:
        name, country, city, language, favorites, hobby, age, relationships
        """
        # patterns -> (regex, storage_key, post_process_fn (optional))
        patterns = [
            (r"\bmy name is\s+([A-Za-z\s'\-]+)", "user_name"),
            (r"\bi am called\s+([A-Za-z\s'\-]+)", "user_name"),
            (r"\bi'm called\s+([A-Za-z\s'\-]+)", "user_name"),
            (r"\bmy country is\s+([A-Za-z\s]+)", "country"),
            (r"\bi am from\s+([A-Za-z\s]+)", "country"),
            (r"\bfrom\s+([A-Za-z\s]+)", "country"),
            (r"\bmy city is\s+([A-Za-z\s]+)", "city"),
            (r"\bi live in\s+([A-Za-z\s]+)", "city"),
            (r"\bmy favorite color is\s+([A-Za-z\s]+)", "favorite_color"),
            (r"\bmy hobby is\s+([A-Za-z\s]+)", "hobby"),
            (r"\bi like\s+([A-Za-z\s]+)", "likes"),
            (r"\bmy age is\s+(\d{1,3})", "age"),
        ]

        for pattern, key in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                val = m.group(1).strip().rstrip(".!?")
                # Normalize capitalization for simple values
                if key in ["user_name"]:
                    stored = val.title()
                else:
                    stored = val.capitalize()
                self.data[key] = stored
                self._save_memory()
                # friendly confirmation
                if key == "user_name":
                    return f"Nice to meet you, {stored}! 💫 I'll remember your name."
                elif key == "country":
                    return f"Oh nice! 🌍 {stored} — I'll remember that."
                elif key == "city":
                    return f"{stored} sounds like a lovely city 🏙️ — got it!"
                else:
                    return f"Got it! I’ve learned that your {key.replace('_',' ')} is {stored} 🧠"
        # relationships
        rel_patterns = {
            r"\bmy friend is\s+([A-Za-z\s'\-]+)": "friend",
            r"\bmy best friend is\s+([A-Za-z\s'\-]+)": "best_friend",
            r"\bmy girlfriend is\s+([A-Za-z\s'\-]+)": "girlfriend",
            r"\bmy boyfriend is\s+([A-Za-z\s'\-]+)": "boyfriend",
            r"\bmy teacher is\s+([A-Za-z\s'\-]+)": "teacher",
            r"\bmy mother (?:name )?is\s+([A-Za-z\s'\-]+)": "mother",
            r"\bmy father (?:name )?is\s+([A-Za-z\s'\-]+)": "father",
            r"\bmy sister (?:name )?is\s+([A-Za-z\s'\-]+)": "sister",
            r"\bmy brother (?:name )?is\s+([A-Za-z\s'\-]+)": "brother",
            r"\bmy crush is\s+([A-Za-z\s'\-]+)": "crush"
        }
        for pattern, key in rel_patterns.items():
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                val = m.group(1).strip().rstrip(".!?").title()
                if "relationships" not in self.data:
                    self.data["relationships"] = {}
                self.data["relationships"][key] = val
                self._save_memory()
                heart = "💞" if key in ["girlfriend", "boyfriend", "crush"] else "🤝"
                return f"Nice — I’ve learned your {key.replace('_',' ')} is {val} {heart}"

        return None

    # -------------------------
    # Auto-learn general facts (X is Y)
    # -------------------------
    def _learn_fact_statement(self, text: str) -> Optional[str]:
        """
        Learn generic factual statements of the form "<subject> is <definition>"
        Avoid learning questions like "what is ..." by checking for 'what' near the start.
        """
        # avoid transforming questions into facts
        if re.match(r"\s*(what|who|why|how)\b", text, re.IGNORECASE):
            return None

        m = re.match(r"\s*([^?.!]+?)\s+is\s+(.+)", text, re.IGNORECASE)
        if m:
            subject = m.group(1).strip().rstrip(".!?").lower()
            meaning = m.group(2).strip().rstrip(".!?")
            # store normalized subject -> meaning
            # protect against too-short subjects (single letter)
            if len(subject) < 2 or len(meaning) < 1:
                return None
            self.data["facts"][subject] = meaning
            self._save_memory()
            return f"Got it! I’ve learned that **{subject}** is **{meaning}** 🧠"
        return None

    # -------------------------
    # Teach FlexiPDF knowledge (special command)
    # user can say: Ali learn 'pdf split' means Divide PDF...
    # -------------------------
    def _learn_flexipdf_knowledge(self, text: str) -> Optional[str]:
        # pattern: Ali learn 'key' means some explanation
        m = re.search(r"ali learn\s+'([^']+)'\s+means\s+(.+)", text, re.IGNORECASE)
        if m:
            key = m.group(1).strip().lower().replace(" ", "_")
            explanation = m.group(2).strip().rstrip(".!?")
            if "flexipdf_knowledge" not in self.data:
                self.data["flexipdf_knowledge"] = {}
            self.data["flexipdf_knowledge"][key] = explanation
            self._save_memory()
            return f"Nice! I learned that **{key.replace('_',' ')}** means: {explanation} ✅"
        # also allow: teach me about <topic>: <explanation>
        m2 = re.search(r"teach ali about\s+'?([^:']+)'?:\s*(.+)", text, re.IGNORECASE)
        if m2:
            key = m2.group(1).strip().lower().replace(" ", "_")
            explanation = m2.group(2).strip().rstrip(".!?")
            if "flexipdf_knowledge" not in self.data:
                self.data["flexipdf_knowledge"] = {}
            self.data["flexipdf_knowledge"][key] = explanation
            self._save_memory()
            return f"Thanks — I learned about **{key.replace('_',' ')}**: {explanation} ✅"
        return None

    # -------------------------
    # Query handlers
    # -------------------------
    def _answer_fact_query(self, text: str) -> Optional[str]:
        """
        Answer questions like:
          - what is love
          - who is <name>
          - define <term>
        Looks up learned facts first, then flexipdf_knowledge
        """
        # what is / who is / define
        m = re.match(r"\s*(what is|who is|define)\s+(.+)\??", text, re.IGNORECASE)
        if m:
            term = m.group(2).strip().rstrip(".!?").lower()
            # exact facts
            if term in self.data.get("facts", {}):
                return f"{term.capitalize()} is {self.data['facts'][term]} 💡"
            # flexipdf knowledge
            if term in self.data.get("flexipdf_knowledge", {}):
                return f"{term.replace('_',' ').capitalize()}: {self.data['flexipdf_knowledge'][term]}"
            # simple synonyms - try strip 'the ' or trailing words
            term_clean = re.sub(r"^(the|a|an)\s+", "", term)
            if term_clean in self.data.get("facts", {}):
                return f"{term_clean.capitalize()} is {self.data['facts'][term_clean]} 💡"
            # not found
            return None
        return None

    def _answer_relationship_query(self, text: str) -> Optional[str]:
        # who is my girlfriend / who is my friend / my girlfriend?
        for rel in self.data.get("relationships", {}).keys():
            if f"who is my {rel.replace('_',' ')}" in text.lower() or rel in text.lower():
                name = self.data["relationships"].get(rel)
                if name:
                    heart = "💞" if rel in ["girlfriend", "boyfriend", "crush"] else "🤝"
                    return f"Your {rel.replace('_',' ')} is {name} {heart}"
        # also allow direct question like "who is my girlfriend"
        m = re.search(r"who is my ([a-z_ ]+)\??", text, re.IGNORECASE)
        if m:
            key = m.group(1).strip().replace(" ", "_")
            name = self.data.get("relationships", {}).get(key)
            if name:
                return f"Your {key.replace('_',' ')} is {name} 💖"
        return None

    def _answer_personal_query(self, text: str) -> Optional[str]:
        # my name / my city / my country / favorite color / hobby
        q_map = {
            "my name": "user_name",
            "who am i": "user_name",
            "my country": "country",
            "what is my country": "country",
            "my city": "city",
            "what is my city": "city",
            "favorite color": "favorite_color",
            "my hobby": "hobby",
            "my age": "age"
        }
        for phrase, key in q_map.items():
            if phrase in text.lower():
                val = self.data.get(key)
                if val:
                    return f"Your {key.replace('_',' ')} is {val} 🌟"
                else:
                    return f"I don’t know your {key.replace('_',' ')} yet. Tell me by saying 'My {key.replace('_',' ')} is ...'"
        return None

    # -------------------------
    # Contextual reference (smart)
    # -------------------------
    def _contextual_reference(self, text: str) -> Optional[str]:
        # If user asks "remember what I said" or "you told me yesterday" or "about X earlier"
        lower = text.lower()
        if any(tok in lower for tok in ["yesterday", "earlier", "before", "last time", "you told me"]):
            # reply referencing last meaningful context
            for entry in reversed(self.context):
                u = entry.get("user", "")
                if len(u.split()) > 2:
                    return f"You mentioned earlier: \"{u}\" — would you like to continue that topic?"
            return "I remember bits of our last chats — what would you like to continue?"
        # if user asks "what did I tell you about X" attempt a semantic search (if available)
        m = re.search(r"what did i tell you about (.+)\??", text, re.IGNORECASE)
        if m:
            topic = m.group(1).strip()
            # semantic lookup by scanning saved conversations for that keyword
            matches = [c for c in reversed(self.data.get("conversations", [])) if topic.lower() in c.get("user", "").lower()]
            if matches:
                sample = matches[0]
                return f"You told me: \"{sample['user']}\" — I replied: \"{sample['ai']}\" on {sample['time']}"
            # if embeddings available use semantic search
            if HAS_EMBED and self.embedding_model and self.faiss_index:
                results = self._semantic_search(topic, k=3)
                if results:
                    # return first semantically close user message
                    return f"I found something related: \"{results[0]}\" — does that match what you meant?"
            return "I don't see that in my recent chats. Want to tell me about it now?"
        return None

    # -------------------------
    # Friendly default reply generator
    # -------------------------
    def _friendly_fallback(self, user_text: str) -> str:
        name = self.data.get("user_name") or "friend"
        templates = [
            f"Hmm 🤔 interesting, {name}. Tell me more about that.",
            f"That sounds cool, {name}! 💫 Want to expand on that?",
            f"I’m listening, {name} — go on 👂",
            f"Thanks for sharing, {name}. How does that make you feel?",
            f"Nice — tell me more or ask me for help with FlexiPDF tools!"
        ]
        return templates[0]  # keep predictable; can randomize if you want

    # -------------------------
    # Main entry: get_response
    # -------------------------
    def get_response(self, user_input: str) -> str:
        if not user_input or not user_input.strip():
            return "Type something for me to respond to ✍️"

        text = user_input.strip()

        # 1) Quick structured learning: name/city/country/relationships
        learned = self._learn_structured_fact(text)
        if learned:
            self._remember_conversation(text, learned)
            return learned

        # 2) FlexiPDF knowledge teaching
        learned_fp = self._learn_flexipdf_knowledge(text)
        if learned_fp:
            self._remember_conversation(text, learned_fp)
            return learned_fp

        # 3) Generic fact learning "X is Y" (avoids questions)
        learned_fact = self._learn_fact_statement(text)
        if learned_fact:
            self._remember_conversation(text, learned_fact)
            return learned_fact

        # 4) Attempt to answer relationship queries
        rel_ans = self._answer_relationship_query(text)
        if rel_ans:
            self._remember_conversation(text, rel_ans)
            return rel_ans

        # 5) Answer facts / definitional queries
        fact_ans = self._answer_fact_query(text)
        if fact_ans:
            self._remember_conversation(text, fact_ans)
            return fact_ans

        # 6) Personal queries (name/city/country/fav/hobby)
        personal = self._answer_personal_query(text)
        if personal:
            self._remember_conversation(text, personal)
            return personal

        # 7) Contextual reference
        ctx_ref = self._contextual_reference(text)
        if ctx_ref:
            self._remember_conversation(text, ctx_ref)
            return ctx_ref

        # 8) FlexiPDF help queries (direct)
        # common phrases mapping to keys
        help_map = {
            "how do i convert pdf to word": "pdf_to_word",
            "convert pdf to word": "pdf_to_word",
            "how to convert pdf to word": "pdf_to_word",
            "pdf to word": "pdf_to_word",
            "word to pdf": "word_to_pdf",
            "pdf to image": "pdf_to_image",
            "images to pdf": "images_to_pdf",
            "merge pdf": "merge_pdfs",
            "split pdf": "split_pdf",
            "compress pdf": "compress_pdf",
            "what can you do": "ai_assistant",
            "who made you": "about"
        }
        lower = text.lower()
        for phrase, key in help_map.items():
            if phrase in lower:
                answer = self.data.get("flexipdf_knowledge", {}).get(key)
                if answer:
                    resp = f"{answer} If you'd like, I can guide you step-by-step."
                    self._remember_conversation(text, resp)
                    return resp

        # 9) Emotion detection quick replies
        if re.search(r"\b(i am|i'm)\s+(sad|happy|angry|tired|bored|excited)\b", lower, re.IGNORECASE):
            em = re.search(r"\b(i am|i'm)\s+(\w+)\b", lower, re.IGNORECASE)
            if em:
                mood = em.group(2).lower()
                mood_replies = {
                    "sad": "Oh no 😢 I'm here for you — want to tell me what's wrong?",
                    "happy": "Yay 😄 I'm happy for you!",
                    "angry": "Take a deep breath 😤 — I’m with you.",
                    "tired": "Maybe a short break will help 💤",
                    "bored": "Let’s do something fun — want a joke or a fact?",
                    "excited": "Awesome! Tell me what's got you excited 🎉"
                }
                reply = mood_replies.get(mood, None)
                if reply:
                    self._remember_conversation(text, reply)
                    return reply

        # 10) Greeting & small talk
        if re.match(r"^(hi|hello|hey|yo|hiya)\b", lower, re.IGNORECASE):
            nm = self.data.get("user_name") or "friend"
            reply = f"Hey {nm}! 👋 I'm Ali — your FlexiPDF AI assistant. How can I help you today?"
            self._remember_conversation(text, reply)
            return reply

        # 11) If user asks "what do you know about <topic>" or "tell me about <topic>"
        m_topic = re.match(r"(?:tell me about|what do you know about)\s+(.+)", lower, re.IGNORECASE)
        if m_topic:
            topic = m_topic.group(1).strip().lower()
            # try facts then flexipdf
            if topic in self.data.get("facts", {}):
                ans = f"{topic.capitalize()} is {self.data['facts'][topic]}"
                self._remember_conversation(text, ans)
                return ans
            if topic in self.data.get("flexipdf_knowledge", {}):
                ans = self.data["flexipdf_knowledge"][topic]
                self._remember_conversation(text, ans)
                return ans
            return_text = f"I don't have much on {topic} yet — would you like to teach me? Say: Ali learn '{topic}' means <your explanation>"
            self._remember_conversation(text, return_text)
            return return_text

        # 12) fallback - friendly
        fallback = self._friendly_fallback(text)
        self._remember_conversation(text, fallback)
        return fallback


# If run as a quick test (optional)
if __name__ == "__main__":
    bot = AliChatbot()
    print("Ali loaded. Try chatting:")
    while True:
        try:
            msg = input("You: ").strip()
            if not msg:
                continue
            if msg.lower() in ["exit", "quit"]:
                print("Ali: Bye 👋")
                break
            print("Ali:", bot.get_response(msg))
        except KeyboardInterrupt:
            print("\nAli: Bye 👋")
            break
