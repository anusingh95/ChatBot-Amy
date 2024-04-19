from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API")

# Check if GOOGLE_API_KEY is loaded, if not raise an error
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

# Configure the Google Generative AI API
genai.configure(api_key=GOOGLE_API_KEY)

pdf_destination = "/home/anon/Desktop/Chatbot/data"

pdf_docs = [os.path.join(pdf_destination, pdf_file) for pdf_file in os.listdir(pdf_destination) if pdf_file.endswith('.pdf')]

text = ""
for pdf in pdf_docs:
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_text(text)

embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001",google_api_key=GOOGLE_API_KEY )
vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
vector_store.save_local("chatbot")
