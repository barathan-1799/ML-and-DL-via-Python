## build_and_save.py

# Importing dependencies
import os
import re
import json
import mailbox
import warnings
from typing import List
import spacy
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)
from langchain_huggingface import HuggingFacePipeline
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# Suppress warnings
warnings.filterwarnings("ignore")
def warn(*args, **kwargs):
    pass
warnings.warn = warn

# (1) Creating helper functions to parse and anonymize emails
nlp = spacy.load("en_core_web_sm")

def extract_body(msg):
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain" and "attachment" not in str(part.get("Content-Disposition") or ""):
                return part.get_payload(decode=True).decode("utf-8", errors="ignore")
        return ""
    else:
        return msg.get_payload(decode=True).decode("utf-8", errors="ignore")

def split_thread(body: str):
    pattern = r"On.*\nwrote:\n\n"
    if re.search(pattern, body) is None:
        return "Not available", "Not available"
    parts = re.split(pattern, body, maxsplit=1)
    reply    = parts[0].strip()
    original = parts[1].strip() if len(parts) > 1 else ""
    return original, reply

def anonymize_email(text: str) -> str:
    # strip common signatures
    sigs = [
        r"(?mi)^--+[\s\S]*$",             
        r"(?mi)^(Best regards|Regards|Sincerely|Cheers|Thank you),.*$",
    ]
    for pat in sigs:
        text = re.sub(pat, "", text)

    # redact IDs & emails & phones
    id_patts = {
        r"\b\d{3}-\d{2}-\d{4}\b": "[REDACTED SSN]",
        r"\b\d{4}-\d{4}-\d{4}-\d{4}\b": "[REDACTED CARD]",
        r"\bID[: ]*\d+\b": "[REDACTED ID]",
        r"\bWBA\d{12}\b": "[REDACTED VIN]",
    }
    for patt, placeholder in id_patts.items():
        text = re.sub(patt, placeholder, text)

    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
                  "[REDACTED EMAIL]", text)
    text = re.sub(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b",
                  "[REDACTED PHONE]", text)

    # redact PERSON entities
    doc = nlp(text)

    for i in range(len(list(doc.ents))):
        doc = nlp(text)
        spans = [(ent.start_char, ent.end_char) for ent in doc.ents if ent.label_ == "PERSON"]
    
        for start, end in sorted(spans, reverse=True):
            text = text[:start] + "[REDACTED NAME]" + text[end:]

    # cleanup
    return re.sub(r"\n{2,}", "\n\n", text).strip()

# (2) Build JSONL for retriever
def build_jsonl(mbox_path: str, jsonl_path: str):
    mbox = mailbox.mbox(mbox_path)
    count = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for msg in mbox:
            body = extract_body(msg)
            orig, reply = split_thread(body)
            orig, reply = anonymize_email(orig), anonymize_email(reply)
            if orig not in ("Not available","") and reply not in ("Not available","") and orig != reply:
                rec = {
                    "id": msg.get("Message-ID", str(count)),
                    "body": orig,
                    "reply": reply,
                    "subject": msg["subject"],
                    "from": msg["from"],
                    "to": msg["to"],
                    "date": msg["date"],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
    print(f"→ Wrote {count} examples to {jsonl_path}")

# (3) Construct the HuggingFacePipeline
def get_llm(model_id: str = "google/flan-t5-base") -> HuggingFacePipeline:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        min_length=100,
        do_sample=True,
        top_p=0.9,
        temperature=0.3,
        device=0 if torch.cuda.is_available() else -1,
    )
    return HuggingFacePipeline(pipeline=pipe)

# (4) Build and save retriever
def load_documents(jsonl_path: str) -> List[Document]:
    docs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            docs.append(Document(page_content=rec["body"], metadata=rec))
    return docs

def chunk_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=5, length_function=len)
    return splitter.split_documents(docs)

def build_and_save_retriever(
    jsonl_path: str,
    persist_dir: str = "chroma_db",
    k: int = 3
):
    docs   = load_documents(jsonl_path)
    chunks = chunk_docs(docs)
    embed  = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device":"cpu"})
    db     = Chroma.from_documents(chunks, embed, persist_directory=persist_dir)
    db.persist()
    print(f"→ Chroma store persisted at: {persist_dir}")
    return db.as_retriever(search_kwargs={"k": k})

# (5) Putting it all together
def main():
    mbox_path   = "Sent.mbox"
    jsonl_path  = "passages_again_2.jsonl"
    model_dir   = "saved_rag_t5"
    chroma_dir  = "chroma_db"

    # 1) Build JSONL
    build_jsonl(mbox_path, jsonl_path)

    # 2) Build & save model
    llm = get_llm()
    os.makedirs(model_dir, exist_ok=True)
    llm.pipeline.model.save_pretrained(model_dir)
    llm.pipeline.tokenizer.save_pretrained(model_dir)
    print(f"→ Saved RAG-T5 model & tokenizer in: {model_dir}")

    # 3) Build & save retriever
    _ = build_and_save_retriever(jsonl_path, persist_dir=chroma_dir, k=3)

if __name__ == "__main__":
    main()