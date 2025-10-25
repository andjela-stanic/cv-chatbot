import streamlit as st
import os
import langchain
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter as CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# --------------------------------------------------
# 🔐 SETUP API KEY
# --------------------------------------------------
# Pre pokretanja u terminalu uradi:
# export OPENAI_API_KEY="tvoj_api_kljuc"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------------------------------------------------
# 📘 LOAD CV
# --------------------------------------------------
with open("cv.md", "r", encoding="utf-8") as f:
    cv_text = f.read()

# Split text into manje delove
splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=40)
chunks = splitter.split_text(cv_text)

# Kreiranje embeddinga za pretragu
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(chunks, embeddings)

# --------------------------------------------------
# 🧠 HELPER FUNKCIJE
# --------------------------------------------------
def retrieve_context(question):
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])
    return context

def ask_openai(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are Andjela Stanić's professional AI assistant. "
                                          "Answer naturally, confidently, and conversationally."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6,
        max_tokens=250
    )
    return response.choices[0].message.content.strip()

# --------------------------------------------------
# 🎨 STREAMLIT UI
# --------------------------------------------------
st.set_page_config(page_title="Andjela's Career Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 Andjela's Career Chatbot (GPT-4o-mini)")
st.write("Ask me anything about my career, experience, or achievements!")

st.sidebar.header("📌 Quick Links")
st.sidebar.markdown("[🌐 LinkedIn Profile](https://www.linkedin.com/in/andjela-stanic/)")
st.sidebar.info("This chatbot uses Andjela's CV and OpenAI GPT-4o-mini for intelligent answers.")

query = st.text_input("Enter your question:")

if not query:
    st.info("👋 Hi! I'm Andjela’s virtual CV assistant. Ask me about her work, skills, or projects.")
else:
    with st.spinner("Thinking..."):
        context = retrieve_context(query)
        prompt = f"""
        Based on the following information about Andjela Stanić, answer the question naturally and professionally.

        CV context:
        {context}

        Question: {query}

        Keep it concise (2–4 sentences), in English, and sound like Andjela speaking about herself.
        """
        answer = ask_openai(prompt)
    st.subheader("Answer:")
    st.write(answer)

