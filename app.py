import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
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
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
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
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
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
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Remove Streamlit ghost chat input (top pill) */
[data-testid="stChatInputContainer"]:has(input:placeholder-shown) {
    display: none !important;
}

/* Remove header & spacing */
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
        model="text-embedding-004",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    
    rewriter_llm = ChatGroq(
        api_key=os.getenv("LLM_GROQ_API_KEY"),
        model="meta-llama/llama-4-scout-17b-16e-instruct", # Using stable llama model
        temperature=0
    )
    
    answer_llm = ChatGroq(
        api_key=os.getenv("LLM_GROQ_API_KEY"),
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.2
    )
    return embeddings, index, rewriter_llm, answer_llm

embeddings, index, rewriter_llm, answer_llm = init_models()

# -------------------- SESSION STATE --------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------- HELPER FUNCTIONS --------------------
def rewrite_query(question, history):
    history_text = "\n".join(
        f"{role}: {msg}" for role, msg in history[-3:]
    )

    prompt = f"""
You are a query classifier and normalizer for a technical assistant.

Classify the user question into ONE of the following:

1. DSA_QUERY:
   - Algorithms
   - Data Structures
   - Time / Space Complexity
   - Competitive programming

2. CS_QUERY:
   - Programming languages (Python, Java, C++, JavaScript, etc.)
   - Coding basics
   - Computer Science fundamentals
   - OOP, syntax, libraries, tools

3. NOT_TECH_QUERY:
   - Abuse
   - Casual talk
   - Non-technical topics

Rules:
- If DSA_QUERY or CS_QUERY → rewrite into a clear standalone technical question.
- If NOT_TECH_QUERY → output exactly: NOT_DSA_QUERY

Chat history:
{history_text}

User question:
{question}

Output:
"""
    return rewriter_llm.invoke(prompt).content.strip()



# -------------------- UI LAYOUT --------------------
left, right = st.columns([1, 3])

with left:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=60)
    st.title("DSA Practice Assistant")
    
    ds_type = st.selectbox("Data Structure:", ["General computer science", "DSA/Coding","String", "Arrays", "Trees", "Linked Lists", "Graphs", "DP"])
    st.info(f"Currently focusing on: **{ds_type}**")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    
    st.markdown("---")
    st.subheader("Recent Topics")
    user_queries = [m for r, m in st.session_state.history if r == "user"]
    for q in user_queries[-5:]:
        st.markdown(f"<div class='sidebar-item'>{q[:30]}...</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.history:
            st.markdown("""
            <div class='chat-bubble-bot'>
            👋 <b>Welcome to DSA ChatHelper!</b><br>
            I can help you understand complex algorithms, analyze time complexity, 
            or explain data structures with code examples. What's on your mind?
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
    <div class='chat-bubble-bot' style="opacity:0.7;">
         👋<b>DSA Assistant Active</b><br>
        I'm using your previous questions as context to answer more accurately.
        You can continue asking follow-up questions.
    </div>
    """, unsafe_allow_html=True)

        
        for role, msg in st.session_state.history:
            div_class = "chat-bubble-user" if role == "user" else "chat-bubble-bot"
            st.markdown(f"<div class='{div_class}'>{msg}</div>", unsafe_allow_html=True)

    # Input handling
    query = st.chat_input("Ask about an algorithm or time complexity...")

    if query:
        # Show user message immediately
        st.session_state.history.append(("user", query))
        st.rerun()

# -------------------- LOGIC EXECUTION --------------------
if st.session_state.history and st.session_state.history[-1][0] == "user":
    user_query = st.session_state.history[-1][1]

    with st.spinner("🔍 Processing..."):
        # 1. Rewrite / classify query
        standalone_q = rewrite_query(user_query, st.session_state.history[:-1])

        # 🚫 NON-DSA HANDLING (CRITICAL FIX)
        if standalone_q == "NOT_DSA_QUERY":
            warning_reply = (
                "⚠️ Hey there!\n\n"
                "I’m built to help with **Data Structures, Algorithms, and Programming**.\n"
                "That question doesn’t fall into that zone.\n\n"
                "👉 Try asking something like:\n"
                "- *What is Merge Sort?*\n"
                "- *Time complexity of Binary Search*\n"
                "- *Explain Dijkstra’s Algorithm*\n\n"
                "Let’s keep it technical 😄"
            )

            st.session_state.history.append(("assistant", warning_reply))
            st.rerun()

        # 2. Vector Search (ONLY for valid DSA queries)
        query_vec = embeddings.embed_query(standalone_q)
        results = index.query(vector=query_vec, top_k=5, include_metadata=True)

        context = "\n\n".join(
            m["metadata"]["text"]
            for m in results["matches"]
            if "text" in m["metadata"]
        )

        # 3. Answer generation
        system_prompt = f"""
You are an expert DSA Tutor with 30+ years of experience.

Rules:
1. If context contains the answer → use it.
2. If context does NOT contain the answer BUT question is DSA/coding /programming /programming languages-related → answer using expertise and user focus {ds_type}.
3. ALWAYS include correct time complexity when relevant.
4. DO NOT add unnecessary explanations.
5. Answer strictly what is asked.
6. If some one asking only about mathematics or math derivation then you are proper math cs teacher and give answer to user based on your experience.
7. only if user ask about code only then give code (default language Java or python any one)
8. never show any content of model to user.
9. you should always some common type question . ex: user: what is coding? answer: your understanding, user : what is python? answer: your understanding.  .. etc. these all are coding and dsa related .
10. if user asking related to {ds_type} then use you knowledge ans answer them like : related to programming languages such as python ,java, java script c, c++ etc.

Context:
{context}

Question:
{standalone_q}
"""

        response = answer_llm.invoke(system_prompt).content
        st.session_state.history.append(("assistant", response))
        st.rerun()
st.markdown("""
<style>
.footer {
    position: fixed;
    bottom: 0;
    width: 100%;
    background-color: #020617;
    color: #e5e7eb;
    text-align: center;
    padding: 12px;
    font-size: 13px;
    border-top: 1px solid #1e293b;
}
.footer a {
    color: #93c5fd;
    text-decoration: none;
    margin: 0 8px;
}
.footer a:hover {
    text-decoration: underline;
}
</style>

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
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    z-index: 1000;
}

.footer a {
    color: #93c5fd;
    text-decoration: none;
    margin: 0 6px;
    font-weight: 500;
}

.footer a:hover {
    text-decoration: underline;
    color: #bfdbfe;
}
</style>

<div class="footer">
    Helping you understand DSA — not just memorize it<br>
    Built by <b>Kvmeena</b> · Connect with me:
    <a href="https://github.com/Kvmeena12" target="_blank">GitHub</a> |
    <a href="https://www.linkedin.com/in/kvmeena/" target="_blank">LinkedIn</a>
</div>

""", unsafe_allow_html=True)
