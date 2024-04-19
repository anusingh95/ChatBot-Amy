import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

GOOGLE_API_KEY = st.secrets['GOOGLE_API']

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found.")

# Configure the Google Generative AI API
genai.configure(api_key=GOOGLE_API_KEY)

# Predefined questions
predefined_questions = [
    "Hi Amy",
    "How does your pricing work?",
    "How do I contact customer support?",
    "What are your operating hours?",
    "How do I update my contact information?"
]

def get_conversational_chain():
    prompt_template = """
    Answer the question and don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   temperature=0.3, google_api_key=GOOGLE_API_KEY)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question, chats):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    
    new_db = FAISS.load_local("chatbot", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    
    chats.insert(0, {"User": user_question, "Amy": response["output_text"]})
    
    return chats

def cool_header():
    st.title("🚀 Chat with Amy using Gemini 💁")
    st.markdown("Amy is a chatbot designed to answer queries of customers of a software company that develops AI products.")
    st.markdown("[Source Code](https://github.com/yourusername/yourrepo)")
    
def display_chat(chats):
    for chat in chats:
        st.write(f"You: {chat['User']}")
        st.write(f"Amy: {chat['Amy']}")
def display_pdf(pdf_path):
    with st.expander("View PDF of Questions Amy Responds to", expanded=False):
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                pdf_reader = PdfReader(f)
                for page in pdf_reader.pages:
                    st.write(page.extract_text())
        else:
            st.write("PDF not found.")
def main():
    st.set_page_config(page_title="ChatBot", page_icon=":robot_face:")
    cool_header()
    
    st.sidebar.title("Frequently Asked Questions")
    for idx, question in enumerate(predefined_questions):
        if st.sidebar.button(f"{question}"):
            user_question = question
            chats = st.session_state.get('chats', [])
            chats = user_input(user_question, chats)
            st.session_state['chats'] = chats
    
    user_question = st.text_input("User:")
    chats = st.session_state.get('chats', [])
    
    if st.button("Ask", key="ask_button"):
        with st.spinner("Amy is thinking..."):
            if user_question:
                chats = user_input(user_question, chats)
                st.session_state['chats'] = chats
                user_question = ""  # Clear input field after asking
                
    display_chat(chats)
    display_pdf("ChatbotQ.pdf")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
