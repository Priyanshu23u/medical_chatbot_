import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Custom prompt and memory
from src.prompt import system_prompt
from src.memory import get_chat_history, add_to_history, clear_history

# --- Load env
load_dotenv()
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

# --- Models & retriever
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
base_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

chat_model = ChatGroq(
    model_name="llama3-8b-8192",
    groq_api_key=GROQ_API_KEY
)

# --- History-aware retriever
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question which might reference context, "
               "formulate a standalone question. Do NOT answer, just reformulate."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
history_aware_retriever = create_history_aware_retriever(chat_model, base_retriever, contextualize_q_prompt)

# --- QA prompt
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    ("system", "Context: {context}")
])
question_answer_chain = create_stuff_documents_chain(chat_model, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- Streamlit config
st.set_page_config(page_title="🩺 Medical Chatbot", page_icon="💬", layout="wide")

# --- 🌱 Custom CSS
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
.chat-container {
    background-color: #f4f4f8;
    border-radius: 12px;
    padding: 20px;
    max-height: 70vh;
    overflow-y: auto;
}
.message-row {
    display: flex;
    margin-bottom: 10px;
    width: 100%;
}
.user-row {
    justify-content: flex-end;
}
.bot-row {
    justify-content: flex-start;
}
.avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    margin: 0 8px;
    flex-shrink: 0;
}
.message-bubble {
    padding: 10px 14px;
    border-radius: 18px;
    max-width: 70%;
    word-wrap: break-word;
    white-space: pre-wrap;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}
.user-bubble {
    background-color: #4a90e2;
    color: white;
    border-bottom-right-radius: 4px;
    text-align: left;
}
.bot-bubble {
    background-color: #fff;
    color: #333;
    border-bottom-left-radius: 4px;
    border: 1px solid #ddd;
    text-align: left;
}
.timestamp {
    font-size: 10px;
    color: #888;
    margin-top: 2px;
    text-align: right;
}
.typing {
    font-size: 12px;
    color: #888;
    margin-top: 5px;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)

# --- Header
st.markdown("<h2 style='text-align:center; color:#4a90e2;'>💬 Medical Chatbot</h2>", unsafe_allow_html=True)
st.divider()

# --- State
if "chat_history_ui" not in st.session_state:
    st.session_state.chat_history_ui = []
if "typing" not in st.session_state:
    st.session_state.typing = False

# --- Sidebar
with st.sidebar:
    st.header("⚙️ Options")
    if st.button("🧹 Clear chat"):
        clear_history()
        st.session_state.chat_history_ui = []
        st.rerun()

# --- Chat UI
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for role, text, timestamp in st.session_state.chat_history_ui:
        time_str = timestamp.strftime("%H:%M")
        if role == "user":
            st.markdown(f"""
            <div class="message-row user-row">
                <div style="display:flex; flex-direction:column; align-items:flex-end;">
                    <div class="message-bubble user-bubble">{text}</div>
                    <div class="timestamp">{time_str}</div>
                </div>
                <img src="https://cdn-icons-png.flaticon.com/512/847/847969.png" class="avatar">
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="message-row bot-row">
                <img src="https://cdn-icons-png.flaticon.com/512/4712/4712027.png" class="avatar">
                <div style="display:flex; flex-direction:column; align-items:flex-start;">
                    <div class="message-bubble bot-bubble">{text}</div>
                    <div class="timestamp">{time_str}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    # Typing indicator
    if st.session_state.typing:
        st.markdown("""
        <div class="message-row bot-row">
            <img src="https://cdn-icons-png.flaticon.com/512/4712/4712027.png" class="avatar">
            <div class="typing">🤖 Bot is typing...</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Input
user_input = st.chat_input("Type your question here...")

if user_input:
    now = datetime.now()
    st.session_state.chat_history_ui.append(("user", user_input, now))

    # Show typing indicator
    st.session_state.typing = True
    st.rerun()

# --- Process after typing indicator shown
if st.session_state.typing and len(st.session_state.chat_history_ui) > 0 and st.session_state.chat_history_ui[-1][0] == "user":
    last_user_input = st.session_state.chat_history_ui[-1][1]
    current_chat_history = get_chat_history()

    response = rag_chain.invoke({
        "input": last_user_input,
        "chat_history": current_chat_history.messages
    })
    answer = response["answer"]

    # Add to chat history
    add_to_history(last_user_input, answer)
    st.session_state.chat_history_ui.append(("bot", answer, datetime.now()))

    # Remove typing indicator
    st.session_state.typing = False
    st.rerun()

# --- Footer
st.divider()
st.markdown("<small style='color:gray;'>🩺 Built with FAISS + Groq + LangChain + Streamlit</small>", unsafe_allow_html=True)
