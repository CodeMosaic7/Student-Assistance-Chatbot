import os
import torch
import pandas as pd
import nltk
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

print("Loaded necessary libraries")

# PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
# PINECONE_API_ENV=os.getenv('PINECONE_API_ENV','pinecone')

# Initialize Pinecone Database
def database_initialization():
    pc = Pinecone(api_key='pcsk_2u3TiU_FUtVL6AFu7ghLZQ625pJxvqvCVREALktLm3wVMg4gkmSMtNGcmDY18txQRLsYWx')
    index = pc.Index("student-assistance")
    print("Pinecone initialized")
    return index

index = database_initialization()

# Load HuggingFace Embeddings
def load_huggingface_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-MiniLM-L6-v2')
    print("Embeddings loaded")
    query = "What is the history of Wikipedia?"
    query_embedding = embeddings.embed_documents([query])
    print(len(query_embedding))
    return embeddings

embeddings = load_huggingface_embeddings()
print(embeddings)

# Load and Upsert CSV Data
def load_and_upsert_csv_data(csv_path="Sample_data.csv"):
    df = pd.read_csv(csv_path)
    csv_texts = df.apply(lambda row: " ".join(row.astype(str)), axis=1).tolist()
    csv_embeddings = embeddings.embed_documents(csv_texts)
    csv_records = [
        {"id": f"csv_{idx}", "values": vector, "metadata": {"text": text}}
        for idx, (text, vector) in enumerate(zip(csv_texts, csv_embeddings))
    ]
    index.upsert(vectors=csv_records)
    print("CSV data upserted into Pinecone successfully!")

load_and_upsert_csv_data()

# Load and Upsert Web Data
def load_and_upsert_web_data():
    URLs = [
        'https://en.wikipedia.org/wiki/Main_Page',
        'https://openai.com/',
        'https://aistudio.google.com/',
        'https://www.superdatascience.com/machine-learning'
    ]
    loader = UnstructuredURLLoader(urls=URLs)
    data = loader.load()
    
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(data)
    chunks_texts = [chunk.page_content for chunk in chunks]
    chunk_embeddings = embeddings.embed_documents(chunks_texts)
    
    records = [
        {"id": str(idx), "values": vector, "metadata": {"text": text}}
        for idx, (text, vector) in enumerate(zip(chunks_texts, chunk_embeddings))
    ]
    
    index.upsert(vectors=records)
    print("Website data upserted into Pinecone successfully!")

load_and_upsert_web_data()

# HuggingFace Model Initialization
def initialize_huggingface_model():
    login(token="your_huggingface_api_key")
    print("Logged into Hugging Face")
    
    model_name = "meta-llama/Llama-3.3-70B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=True,
        device_map='auto',
        load_in_8bits=True
    )
    
    pipe = pipeline(
        'text-generation', model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16,
        device_map="auto", max_new_tokens=512, do_sample=True,
        top_k=50, top_p=0.95, num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id
    )
    
    llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': 0})
    print("Model loaded")
    return llm

llm = initialize_huggingface_model()

# Sample Query
response = llm.predict("What is the history of Wikipedia?")
print(response)
