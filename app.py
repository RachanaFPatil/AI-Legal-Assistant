import streamlit as st
import fitz  # PyMuPDF
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
import tempfile
from sentence_transformers import SentenceTransformer, util
import requests
from bs4 import BeautifulSoup
import json
import re
import spacy
from collections import defaultdict
from time import sleep

st.set_page_config(page_title="AI Legal Assistant - LegalBERT Full Summary", layout="wide")

nlp_spacy = spacy.load("en_core_web_sm")
# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models once
@st.cache_resource
def load_models():
    model_path=r"C:\Users\DELL\Downloads\legal\legal-bert-base-uncased"
    legalbert_tokenizer = AutoTokenizer.from_pretrained(model_path)
    legalbert_model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    summary_tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    summary_model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6").to(DEVICE)
    nlp = spacy.load("en_core_web_sm")
    return legalbert_tokenizer, legalbert_model, embed_model, summary_model, summary_tokenizer, nlp

(legalbert_tokenizer, legalbert_model, embed_model, summary_model, summary_tokenizer, nlp) = load_models()

CLAUSES = {
    "Termination": ("terminate termination ends agreement dissolved expire", 3),
    "Confidentiality": ("confidential nondisclosure secrecy privacy", 2),
    "Indemnity": ("indemnify indemnification liability responsible hold harmless", 3),
    "Arbitration": ("arbitration arbitrate mediator dispute resolution binding", 2),
    "Jurisdiction": ("jurisdiction governing law court venue state country", 2),
    "Payment Terms": ("payment fee compensation paid refund reimbursement due invoice", 1)
}

# Extract PDF and return both doc object and page-wise text
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    pages = [page.get_text() for page in doc]
    full_text = "\n".join(pages)
    return doc, pages, full_text

def chunk_text(text, max_words=1200, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words - overlap):
        chunk = " ".join(words[i:i + max_words])
        if len(chunk.strip()) > 100:
            chunks.append(chunk)
    return chunks

def summarize_chunk(chunk):
    inputs = summary_tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    summary_ids = summary_model.generate(inputs["input_ids"], max_length=150, min_length=40, num_beams=4)
    return summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def generate_summary(text):
    chunks = chunk_text(text)
    total = len(chunks)
    summaries = []
    progress_bar = st.progress(0, text="Summarizing document...")
    for i, chunk in enumerate(chunks):
        summaries.append(summarize_chunk(chunk))
        progress_bar.progress((i + 1) / total, text=f"Summarizing... {int((i+1)/total*100)}%")
    return " ".join(summaries)

def detect_clauses(text):
    found, missing, risk_score = [], [], 0
    text_lower = text.lower()
    for clause, (keywords, weight) in CLAUSES.items():
        if any(word in text_lower for word in keywords.split()):
            found.append(clause)
        else:
            missing.append(clause)
            risk_score += weight
    return found, missing, risk_score

def extract_people_and_roles(text):
    lines = text.splitlines()
    roles = {}
    
    role_keywords = {
        "witness": "Witness",
        "guarantor": "Guarantor",
        "liable": "Liable/Responsible Party",
        "responsible": "Liable/Responsible Party",
        "signatory": "Signatory",
        "signed by": "Signatory",
        "first party": "First Party",
        "party of the first part": "First Party",
        "second party": "Second Party",
        "party of the second part": "Second Party",
        "authorized representative": "Authorized Representative",
        "authorised representative": "Authorized Representative"
    }

    prefixes = {"shri", "mr", "mrs", "ms", "smt", "dr", "kumari"}

    for i, line in enumerate(lines):
        lower = line.lower()

        for keyword, role in role_keywords.items():
            if keyword in lower:
                # Look at next 1–4 lines for possible names
                for j in range(1, 5):
                    if i + j >= len(lines):
                        break

                    possible_name = lines[i + j].strip()
                    clean_name = re.sub(r'[^A-Za-z\s.]', '', possible_name).strip()
                    words = clean_name.split()

                    if not words or len(words) < 2 or len(words) > 6:
                        continue

                    prefix_match = words[0].lower().strip(".") in prefixes
                    name_format_ok = all(w[0].isupper() for w in words if w.lower() not in prefixes)

                    if prefix_match or name_format_ok:
                        if clean_name not in roles:
                            roles[clean_name] = role
                        break  # Done with this role

    return roles

def chat_with_contract(question, page_texts):
    all_sentences = []
    sentence_to_page = []

    for i, page in enumerate(page_texts):
        sentences = re.split(r'(?<=[.!?])\s+', page.strip())
        for sent in sentences:
            sent = sent.strip()
            if 30 <= len(sent) <= 600 and not sent.isupper():
                sent = re.sub(r'\s+', ' ', sent)
                all_sentences.append(sent)
                sentence_to_page.append(i + 1)

    if not all_sentences:
        return "⚠️ No useful content found", "Please check your PDF."

    # --- Hybrid Keyword Match ---
    keywords = set(re.sub(r"[^\w\s]", "", question.lower()).split())
    top_hits = []
    for idx, sent in enumerate(all_sentences):
        sent_words = set(re.sub(r"[^\w\s]", "", sent.lower()).split())
        common = keywords & sent_words
        if len(common) >= 2:
            top_hits.append((len(common), idx))

    if top_hits:
        top_hits.sort(reverse=True)
        best_idx = top_hits[0][1]
        return f"📌 Answer (by keyword) on Page {sentence_to_page[best_idx]}", all_sentences[best_idx]

    # --- Semantic Search Fallback ---
    question_embed = embed_model.encode(question, convert_to_tensor=True)
    sent_embeds = embed_model.encode(all_sentences, convert_to_tensor=True)
    results = util.semantic_search(question_embed, sent_embeds, top_k=3)[0]
    best_idx = results[0]["corpus_id"]

    return f"📌 Answer (by meaning) on Page {sentence_to_page[best_idx]}", all_sentences[best_idx]

@st.cache_data
def indian_kanoon_search(query):
    url = f"https://www.indiankanoon.org/search/?formInput={query}"
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        results = soup.select(".result_title a")
        return [(r.text.strip(), "https://www.indiankanoon.org" + r['href']) for r in results[:5]]
    except:
        return []

#st.set_page_config(page_title="AI Legal Assistant - LegalBERT Full Summary", layout="wide")
st.title("⚖️ AI Legal Assistant - LegalBERT Summary & Analysis")

st.subheader("🔎 Indian Case Law (via Indian Kanoon)")
query = st.text_input("Search Indian legal cases:")
if query:
    with st.spinner("Searching Indian Kanoon..."):
        cases = indian_kanoon_search(query)
        for title, link in cases:
            st.markdown(f"- [{title}]({link})")

pdf = st.file_uploader("📂 Upload a legal PDF document", type=["pdf"])

if pdf:
    doc, page_texts, full_text = extract_text_from_pdf(pdf)

    st.subheader("📑 Full Document Summary")
    if "cached_summary" not in st.session_state:
        with st.spinner("Summarizing the document..."):
            st.session_state.cached_summary = generate_summary(full_text)
    st.success("✅ Summary generated!")
    st.markdown(st.session_state.cached_summary)
    st.download_button("📥 Download Summary", st.session_state.cached_summary, file_name="legal_summary.txt")

    st.subheader("📌 Clause Detection & Risk")
    found, missing, risk = detect_clauses(full_text)
    st.success("✅ Found Clauses: " + ", ".join(found))
    st.warning("❌ Missing Clauses: " + ", ".join(missing))
    st.info(f"*⚠️ Risk Score:* {risk} / 10")
    if risk >= 7:
        st.error("🔴 High Risk: Many critical clauses are missing. Consider legal review. 🚫 Not safe to sign without legal advice.")
    elif risk >= 4:
        st.warning("🟠 Moderate Risk: Some important clauses are missing. ⚠️ Review carefully before signing.")
    else:
        st.success("🟢 Low Risk: Most critical clauses are present. ✅ Document appears safe to sign.")

    st.subheader("👥 People and Their Roles")
    people_roles = extract_people_and_roles(full_text)

    if people_roles:
        for person, role in people_roles.items():
            st.markdown(f"- **{person}** → *{role}*")
    else:
        st.warning("❌ No names or roles found. Check the document format.")


    st.subheader("💬 Legal Assistant Chatbot")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Ask about the contract or a legal term...")
    if user_input:
        with st.spinner("Searching the document for an answer..."):
            heading, reply = chat_with_contract(user_input, page_texts)
            st.session_state.chat_history.append(("🧑‍💼 You", user_input))
            st.session_state.chat_history.append(("🤖 LegalBot", f"{heading}\n> {reply}"))

    for sender, msg in st.session_state.chat_history:
        with st.chat_message("user" if sender == "🧑‍💼 You" else "assistant"):
            st.markdown(msg)

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []