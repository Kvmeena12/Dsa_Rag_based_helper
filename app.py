import os
import time
import streamlit as st
from dotenv import load_dotenv

# -------------------- LOAD ENV --------------------
load_dotenv()

from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="DSA ChatHelper",
    page_icon="🧠",
    layout="wide"
)

# -------------------- STYLING --------------------
st.markdown("""
<style>
   .stApp {
    background:
        radial-gradient(circle at 20% 20%, rgba(99, 102, 241, 0.18), transparent 35%),
        radial-gradient(circle at 80% 30%, rgba(236, 72, 153, 0.15), transparent 35%),
        radial-gradient(circle at 50% 80%, rgba(14, 165, 233, 0.12), transparent 40%),
        linear-gradient(180deg, #020617, #020617);
    color: #e5e7eb;
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: rgba(15, 23, 42, 0.9);
    border-right: 1px solid rgba(255,255,255,0.1);
}

.glass-card {
    background: rgba(30, 41, 59, 0.4);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
    padding: 20px;
    margin-bottom: 20px;
}

.chat-bubble-user {
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    color: white;
    padding: 12px 18px;
    border-radius: 18px 18px 2px 18px;
    margin: 10px 0;
    width: fit-content;
    max-width: 80%;
    margin-left: auto;
}

.chat-bubble-bot {
    background: #1e293b;
    border: 1px solid #334155;
    color: #f1f5f9;
    padding: 12px 18px;
    border-radius: 18px 18px 18px 2px;
    margin: 10px 0;
    width: fit-content;
    max-width: 85%;
}

.sidebar-item {
    padding: 10px;
    border-radius: 8px;
    background: rgba(255,255,255,0.05);
    margin-bottom: 8px;
    font-size: 0.85rem;
    border-left: 3px solid #3b82f6;
}

code {
    color: #f472b6 !important;
}

[data-testid="stHeader"],
[data-testid="stToolbar"] {
    display: none !important;
}

.block-container {
    padding-top: 0rem !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------- INITIALIZE MODELS --------------------
@st.cache_resource
def init_models():

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY")
    )

    index = pc.Index(
        os.getenv("PINECONE_INDEX_NAME")
    )

    rewriter_llm = ChatGroq(
        api_key=os.getenv("LLM_GROQ_API_KEY"),
        model="llama3-8b-8192",
        temperature=0
    )

    answer_llm = ChatGroq(
        api_key=os.getenv("LLM_GROQ_API_KEY"),
        model="llama3-8b-8192",
        temperature=0.2
    )

    return embeddings, index, rewriter_llm, answer_llm

embeddings, index, rewriter_llm, answer_llm = init_models()

# -------------------- SESSION STATE --------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------- QUERY REWRITER --------------------
def rewrite_query(question, history):

    history_text = "\n".join(
        f"{role}: {msg}" for role, msg in history[-3:]
    )

    prompt = f"""
You are a technical query classifier.

Categories:
1. DSA_QUERY
2. CS_QUERY
3. NOT_TECH_QUERY

Rules:
- Rewrite technical questions clearly.
- If non technical -> output exactly NOT_DSA_QUERY

History:
{history_text}

Question:
{question}

Output:
"""

    try:
        return rewriter_llm.invoke(prompt).content.strip()
    except:
        return question

# -------------------- SIDEBAR --------------------
left, right = st.columns([1, 3])

with left:

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    st.image(
        "https://cdn-icons-png.flaticon.com/512/2103/2103633.png",
        width=60
    )

    st.title("DSA Practice Assistant")

    ds_type = st.selectbox(
        "Data Structure:",
        [
            "General computer science",
            "DSA/Coding",
            "String",
            "Arrays",
            "Trees",
            "Linked Lists",
            "Graphs",
            "DP"
        ]
    )

    st.info(f"Currently focusing on: **{ds_type}**")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.markdown("---")
    st.subheader("Recent Topics")

    user_queries = [
        m for r, m in st.session_state.history
        if r == "user"
    ]

    for q in user_queries[-5:]:
        st.markdown(
            f"<div class='sidebar-item'>{q[:30]}...</div>",
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- CHAT AREA --------------------
with right:

    if not st.session_state.history:

        st.markdown("""
        <div class='chat-bubble-bot'>
        👋 <b>Welcome to DSA ChatHelper!</b><br><br>
        Ask anything about:
        <ul>
            <li>Algorithms</li>
            <li>Data Structures</li>
            <li>Programming</li>
            <li>Time Complexity</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class='chat-bubble-bot'>
        👋 <b>DSA Assistant Active</b><br>
        I’m using previous conversation context.
        </div>
        """, unsafe_allow_html=True)

    for role, msg in st.session_state.history:

        div_class = (
            "chat-bubble-user"
            if role == "user"
            else "chat-bubble-bot"
        )

        st.markdown(
            f"<div class='{div_class}'>{msg}</div>",
            unsafe_allow_html=True
        )

    query = st.chat_input(
        "Ask about an algorithm or time complexity..."
    )

    if query:
        st.session_state.history.append(("user", query))
        st.rerun()

# -------------------- MAIN LOGIC --------------------
if (
    st.session_state.history and
    st.session_state.history[-1][0] == "user"
):

    user_query = st.session_state.history[-1][1]

    with st.spinner("🔍 Processing..."):

        standalone_q = rewrite_query(
            user_query,
            st.session_state.history[:-1]
        )

        # -------------------- NON TECH --------------------
        if standalone_q == "NOT_DSA_QUERY":

            warning_reply = """
⚠️ I only answer:
- DSA
- Programming
- Coding
- Computer Science

Try asking:
- What is Binary Search?
- Explain Merge Sort
- Time complexity of DFS
"""

            st.session_state.history.append(
                ("assistant", warning_reply)
            )

            st.rerun()

        # -------------------- SAFE EMBEDDINGS --------------------
        query_vec = None

        for attempt in range(3):

            try:

                query_vec = embeddings.embed_query(
                    standalone_q[:300]
                )

                break

            except Exception as e:

                if attempt < 2:
                    time.sleep(2)

                else:
                    st.error(f"Embedding Error: {e}")
                    st.stop()

        # -------------------- VECTOR SEARCH --------------------
        try:

            results = index.query(
                vector=query_vec,
                top_k=5,
                include_metadata=True
            )

            context = "\n\n".join(
                m["metadata"]["text"]
                for m in results["matches"]
                if "text" in m["metadata"]
            )

        except Exception as e:

            context = ""
            st.warning(f"Pinecone issue: {e}")

        # -------------------- ANSWER PROMPT --------------------
        system_prompt = f"""
You are an expert DSA Tutor.

Rules:
1. Use context if available.
2. If context unavailable but question is coding related -> answer from your knowledge.
3. Include time complexity when needed.
4. Keep answers concise.
5. Give code only if asked.
6. Focus area: {ds_type}

Context:
{context}

Question:
{standalone_q}
"""

        # -------------------- GENERATE ANSWER --------------------
        try:

            response = answer_llm.invoke(
                system_prompt
            ).content

        except Exception as e:

            response = f"LLM Error: {e}"

        st.session_state.history.append(
            ("assistant", response)
        )

        st.rerun()

# -------------------- FOOTER --------------------
st.markdown("""
<style>
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: rgba(2, 6, 23, 0.96);
    color: #e5e7eb;
    text-align: center;
    padding: 12px 8px;
    font-size: 13px;
    border-top: 1px solid rgba(255,255,255,0.08);
}

.footer a {
    color: #93c5fd;
    text-decoration: none;
    margin: 0 6px;
}

.footer a:hover {
    text-decoration: underline;
}
</style>

<div class="footer">
Helping you understand DSA — not just memorize it<br>
Built by <b>Kvmeena</b> |
<a href="https://github.com/Kvmeena12" target="_blank">GitHub</a> |
<a href="https://www.linkedin.com/in/kvmeena/" target="_blank">LinkedIn</a>
</div>
""", unsafe_allow_html=True)
