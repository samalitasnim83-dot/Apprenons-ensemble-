import streamlit as st
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from dotenv import load_dotenv
import os
import tempfile

# Charger la clé API
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# Configuration de la page
st.set_page_config(
    page_title="Apprenons",
    page_icon="📚",
    layout="centered"
)

# Style CSS
st.markdown("""
<style>
    .main { background-color: #f0f7ff; }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        padding: 10px 25px;
    }
    .chat-message {
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
    }
    .user-message {
        background-color: #DCF8C6;
        text-align: right;
    }
    .bot-message {
        background-color: #ffffff;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)

# Prompt pédagogique
SYSTEM_PROMPT = """
Tu es Apprenons, un assistant pédagogique bienveillant pour les élèves 
du cycle primaire tunisien qui apprennent le français.

Tes règles ABSOLUES :
1. Tu ne donnes JAMAIS la réponse directement
2. Tu commences TOUJOURS par encourager l'élève
3. Tu donnes des indices progressifs pour guider l'élève
4. Tu utilises un langage simple adapté aux enfants
5. Tu utilises des emojis pour rendre la conversation agréable 😊
6. Si l'élève trouve la bonne réponse, tu le félicites chaleureusement 🎉
7. Tu te bases en priorité sur les documents fournis par l'enseignant
8. Tu ne parles QUE de français : conjugaison, grammaire, orthographe, vocabulaire
9. Si l'élève pose une question hors sujet, tu le rediriges gentiment vers le français

Style de réponse :
- D'abord : encouragement
- Ensuite : indice 1
- Si l'élève bloque : indice 2
- Si l'élève bloque encore : indice 3
- Seulement après 3 indices : donner la réponse avec explication
"""

# Initialisation session
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "page" not in st.session_state:
    st.session_state.page = "eleve"

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3976/3976626.png", width=100)
    st.title("📚 Apprenons")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["💬 Espace Élève", "🔐 Espace Enseignant"],
        index=0
    )
    if page == "🔐 Espace Enseignant":
        st.session_state.page = "enseignant"
    else:
        st.session_state.page = "eleve"

# PAGE ENSEIGNANT
if st.session_state.page == "enseignant":
    st.title("🔐 Espace Enseignant")
    password = st.text_input("Mot de passe", type="password")

    if password == "apprenons2025":
        st.success("✅ Connecté avec succès !")
        st.markdown("### 📤 Uploader vos documents")
        uploaded_files = st.file_uploader(
            "Glissez vos PDF ici",
            type="pdf",
            accept_multiple_files=True
        )
        if uploaded_files:
            if st.button("📥 Charger les documents"):
                with st.spinner("Chargement des documents..."):
                    all_docs = []
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_file.read())
                            tmp_path = tmp.name
                        loader = PyPDFLoader(tmp_path)
                        docs = loader.load()
                        all_docs.extend(docs)
                        os.unlink(tmp_path)

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    chunks = splitter.split_documents(all_docs)
                    embeddings = FakeEmbeddings(size=768)
                    st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
                    st.success(f"✅ {len(uploaded_files)} document(s) chargé(s) avec succès !")

    elif password != "":
        st.error("❌ Mot de passe incorrect")

# PAGE ÉLÈVE
else:
    st.title("👋 Bonjour ! Je suis Apprenons")
    st.markdown("*Ton assistant pour apprendre le français* 📚")
    st.markdown("---")

    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
            👦 {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
            🤖 {message["content"]}
            </div>
            """, unsafe_allow_html=True)

    user_input = st.chat_input("Pose ta question ici... 😊")

    if user_input:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        context = ""
        if st.session_state.vectorstore:
            docs = st.session_state.vectorstore.similarity_search(user_input, k=3)
            context = "\n".join([doc.page_content for doc in docs])

        full_prompt = SYSTEM_PROMPT
        if context:
            full_prompt += f"\n\nDocuments de référence :\n{context}"

        messages_for_api = [{"role": "system", "content": full_prompt}]
        for msg in st.session_state.messages[-6:]:
            messages_for_api.append({
                "role": msg["role"] if msg["role"] != "assistant" else "assistant",
                "content": msg["content"]
            })

        with st.spinner("Apprenons réfléchit... 🤔"):
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages_for_api,
                max_tokens=1024,
                temperature=0.7
            )
            answer = response.choices[0].message.content

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        st.rerun()
