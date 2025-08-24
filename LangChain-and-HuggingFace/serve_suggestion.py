## serve_suggestion.py

# Importing dependencies
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Config
MODEL_DIR  = "saved_rag_t5"
CHROMA_DIR = "chroma_db"
PORT       = 8000

# Load LLM
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
device    = 0 if torch.cuda.is_available() else -1
hf_pipe   = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    min_length=100,
    do_sample=True,
    top_p=0.9,
    temperature=0.3,
    device=device,
)
llm = HuggingFacePipeline(pipeline=hf_pipe)

# Load retriever
embed = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device":"cpu"})
db    = Chroma(persist_directory=CHROMA_DIR, embedding_function=embed)
retriever = db.as_retriever(search_kwargs={"k": 3})

# Build QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False,
)

# Instantiating FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # lock this down to your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Req(BaseModel):
    thread: str

class Resp(BaseModel):
    suggestion: str

@app.post("/suggest", response_model=Resp)
def suggest(req: Req):
    answer = qa_chain.run(req.thread)
    return Resp(suggestion=answer)

if __name__ == "__main__":
    uvicorn.run("serve_suggestion:app", host="0.0.0.0", port=PORT, log_level="info")
