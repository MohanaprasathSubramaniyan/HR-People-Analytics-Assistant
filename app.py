import streamlit as st
import pandas as pd
import os

from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# ===================== 1. PAGE CONFIGURATION =====================
st.set_page_config(
    page_title="HR & People Analytics Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== 2. "DEEP ROYAL" UI STYLING (CSS) =====================
st.markdown("""
<style>
    /* IMPORT FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #e2e8f0;
    }

    /* BACKGROUND: Deep Void Black */
    .stApp {
        background-color: #020617; 
        background-image: radial-gradient(at 50% 0%, #0f172a 0px, transparent 50%);
    }

    /* SIDEBAR: Pure Black */
    section[data-testid="stSidebar"] {
        background-color: #000000; 
        border-right: 1px solid #1e293b;
    }
    
    /* TEXT COLORS */
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #f8fafc !important; 
    }
    .stCaption { color: #94a3b8 !important; }

    /* -------------------------------------------
       GLOBAL BUTTONS (The New "Deep Royal" Gradient)
    ------------------------------------------- */
    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid #1e1b4b; /* Subtle dark border */
        
        /* THE NEW MIX: Royal Blue to Deep Indigo (Professional, not neon) */
        background: linear-gradient(90deg, #2563eb 0%, #4f46e5 100%);
        
        color: #f1f5f9 !important; /* Soft white */
        font-size: 1.05rem;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); /* Subtle shadow, no glow */
        transition: all 0.3s ease;
    }

    /* HOVER EFFECT */
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(37, 99, 235, 0.4);
        background: linear-gradient(90deg, #3b82f6 0%, #6366f1 100%); /* Slightly lighter on hover */
    }

    div.stButton > button:active {
        transform: scale(0.98);
    }

    /* -------------------------------------------
       NAVIGATION BUTTONS (Big ones)
    ------------------------------------------- */
    div.row-widget.stButton > button[kind="secondary"] {
        height: 70px;
        font-size: 1.1rem;
    }

    /* -------------------------------------------
       SIDEBAR BUTTONS (Smaller, Darker Mix)
    ------------------------------------------- */
    section[data-testid="stSidebar"] div.stButton > button {
        height: 45px;
        /* Darker gradient for sidebar to be less distracting */
        background: linear-gradient(90deg, #1e3a8a 0%, #312e81 100%);
        border: 1px solid #312e81;
        font-size: 0.95rem;
    }
    
    section[data-testid="stSidebar"] div.stButton > button:hover {
        background: linear-gradient(90deg, #2563eb 0%, #4338ca 100%);
    }

    /* PRIMARY SEARCH BUTTON (Cyan Accent) */
    button[kind="primary"] {
        background: linear-gradient(90deg, #0891b2 0%, #2563eb 100%) !important;
        box-shadow: 0 0 10px rgba(8, 145, 178, 0.2);
        border: none;
        height: auto;
    }

    /* GLASS CARDS */
    .glass-card {
        background: rgba(15, 23, 42, 0.6); 
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 30px;
        margin-top: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }

    /* INPUT FIELDS */
    .stTextInput > div > div > input {
        background-color: #0f172a;
        color: white;
        border-radius: 12px;
        border: 1px solid #334155;
        padding: 14px;
        font-size: 1rem;
    }
    .stTextInput > div > div > input:focus {
        border-color: #6366f1; 
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.3);
    }

    /* RESULT BOX */
    .result-box {
        background: rgba(15, 23, 42, 0.8);
        border-left: 5px solid #6366f1;
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
    }

</style>
""", unsafe_allow_html=True)

# ===================== 3. SYSTEM CONFIGURATION =====================
DATA_FOLDER = r"C:\HR_Project"
DB_PATH = os.path.join(DATA_FOLDER, "chroma_db")
CSV_PATH = os.path.join(DATA_FOLDER, "employees.csv")

# ===================== 4. DATA LOADING =====================
@st.cache_data
def load_data():
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    return None

df = load_data()

# ===================== 5. SESSION STATE LOGIC =====================
if "app_mode" not in st.session_state:
    st.session_state["app_mode"] = "policy"

if "trigger_policy_search" not in st.session_state:
    st.session_state["trigger_policy_search"] = False
if "trigger_data_search" not in st.session_state:
    st.session_state["trigger_data_search"] = False
if "active_data_view" not in st.session_state:
    st.session_state["active_data_view"] = None 

# --- ACTION FUNCTIONS ---
def set_mode(mode):
    st.session_state["app_mode"] = mode
    st.session_state["active_data_view"] = None 

def trigger_policy(text):
    st.session_state["policy_input_widget"] = text 
    st.session_state["trigger_policy_search"] = True
    st.session_state["app_mode"] = "policy"

def trigger_data(text):
    st.session_state["data_input_widget"] = text
    st.session_state["trigger_data_search"] = True
    st.session_state["app_mode"] = "data"
    st.session_state["active_data_view"] = None 

def set_data_view(view_name):
    st.session_state["active_data_view"] = view_name
    st.session_state["app_mode"] = "data" 

# ===================== 6. LOGIC FUNCTIONS =====================
def get_policy_answer(query):
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
    llm = Ollama(model="llama3")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    result = qa_chain.invoke({"query": query})
    return result["result"], result["source_documents"]

def get_data_answer(query):
    llm = Ollama(model="llama3")
    agent = create_pandas_dataframe_agent(
        llm, df, verbose=True, allow_dangerous_code=True
    )
    result = agent.invoke(query)
    return result["output"]

# ===================== 7. STATIC SIDEBAR (Always Visible) =====================
with st.sidebar:
    st.markdown("## 🏢 **HR Pro Platform**")
    st.markdown("---")
    
    st.markdown("""
    <div style='background: rgba(255,255,255,0.05); padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 20px;'>
        <div style='color: white; margin-bottom: 8px;'>🟢 <strong>System:</strong> Online</div>
        <div style='color: white; margin-bottom: 8px;'>🔒 <strong>Mode:</strong> Secure Local</div>
        <div style='color: white;'>🧠 <strong>Model:</strong> LLaMA 3</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📝 **Project Info**")
    st.caption("Advanced People Analytics Dashboard\nv2.0 Enterprise Edition")

# ===================== 8. MAIN DASHBOARD =====================

# Title
st.markdown("""
<div style="margin-bottom: 30px; text-align: center;">
    <h1 style="font-size: 2.8rem; margin-bottom: 10px; font-weight: 800;">
        HR & People Analytics Assistant
    </h1>
    <p style="color: #94a3b8 !important; font-size: 1.1rem;">AI-powered insights for policy compliance and workforce intelligence</p>
</div>
""", unsafe_allow_html=True)

# --- NAVIGATION ---
col_nav1, col_nav2 = st.columns(2)
with col_nav1:
    if st.button("HR Policy Assistant", use_container_width=True, type="secondary"):
        set_mode("policy")
with col_nav2:
    if st.button("Employee Data Analysis", use_container_width=True, type="secondary"):
        set_mode("data")

# --- VISUAL INDICATOR (Gradient Line) ---
if st.session_state["app_mode"] == "policy":
    st.markdown(f"<div style='height: 4px; width: 50%; background: #6366f1; box-shadow: 0 0 15px #6366f1; margin-left: 0%; border-radius: 2px; transition: all 0.3s;'></div>", unsafe_allow_html=True)
else:
    st.markdown(f"<div style='height: 4px; width: 50%; background: #6366f1; box-shadow: 0 0 15px #6366f1; margin-left: 50%; border-radius: 2px; transition: all 0.3s;'></div>", unsafe_allow_html=True)


# =========================================================
#  MODE 1: HR POLICY ASSISTANT
# =========================================================
if st.session_state["app_mode"] == "policy":
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📄 Policy Knowledge Base")
    st.markdown("Ask questions about *Benefits, Code of Conduct, Severance, and Remote Work*.")

    col_input, col_prompts = st.columns([2, 1])

    with col_input:
        query = st.text_input(
            "Type your question here...", 
            placeholder="e.g., What is the policy on moonlighting?",
            key="policy_input_widget"
        )
        
        search_clicked = st.button("🔍 Find Answer", type="primary", key="policy_search_btn")
        
        if search_clicked or st.session_state["trigger_policy_search"]:
            if query:
                st.session_state["trigger_policy_search"] = False
                
                with st.spinner("🤖 Reading documents & synthesizing answer..."):
                    try:
                        answer, sources = get_policy_answer(query)
                        
                        st.markdown(f"""
                        <div class="result-box">
                            <h4 style="margin-top:0; color: #a5b4fc !important;">✨ AI Response:</h4>
                            <p style="font-size: 1.1rem; line-height: 1.6; color: #e2e8f0 !important;">{answer}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        st.caption("📚 References used for this answer:")
                        for doc in sources:
                            with st.expander(f"Source: {doc.metadata.get('source', 'Unknown File')}"):
                                st.write(doc.page_content)
                    except Exception as e:
                        st.error(f"Error: {e}")

    with col_prompts:
        st.markdown("**⚡ Quick Prompts**")
        
        st.button("What are the health benefits?", use_container_width=True, 
                  on_click=trigger_policy, args=("Explain the health insurance benefits.",))
            
        st.button("What is the remote work policy?", use_container_width=True, 
                  on_click=trigger_policy, args=("What is the remote work policy?",))
            
        st.button("What is the code of conduct?", use_container_width=True, 
                  on_click=trigger_policy, args=("What is the code of conduct?",))

    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
#  MODE 2: EMPLOYEE DATA ANALYSIS
# =========================================================
elif st.session_state["app_mode"] == "data":

    # --- DYNAMIC SIDEBAR INJECTION ---
    with st.sidebar:
        st.markdown("### 📂 Data Explorer")
        st.caption("Click to view raw data tables:")
        
        st.button("👥 View Full Staff List", use_container_width=True, on_click=set_data_view, args=("all",))
        st.button("💰 View Salary Sheet", use_container_width=True, on_click=set_data_view, args=("salary",))
        st.button("🏢 View Departments", use_container_width=True, on_click=set_data_view, args=("dept",))
        
        st.markdown("---")

    # --- MAIN CONTENT ---
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Workforce Analytics Engine") 
    st.markdown("Analyze *Salary, Demographics, Departments, and Attrition* using natural language.")

    # --- 1. HANDLE DATA EXPLORER VIEW (Tables) ---
    if st.session_state["active_data_view"] == "all":
        st.info("👥 Displaying Full Employee Directory")
        st.dataframe(df, use_container_width=True)
        
    elif st.session_state["active_data_view"] == "salary":
        st.info("💰 Displaying Compensation Data")
        cols = [c for c in ['Name', 'Department', 'Position', 'Salary'] if c in df.columns]
        st.dataframe(df[cols], use_container_width=True)
        
    elif st.session_state["active_data_view"] == "dept":
        st.info("🏢 Displaying Department Summary")
        if 'Department' in df.columns:
            dept_summary = df['Department'].value_counts().reset_index()
            dept_summary.columns = ['Department', 'Employee Count']
            st.dataframe(dept_summary, use_container_width=True)
        else:
            st.warning("No 'Department' column found in dataset.")

    # --- 2. STANDARD CHAT INTERFACE ---
    col_input, col_prompts = st.columns([2, 1])

    with col_input:
        query = st.text_input(
            "Describe the analysis you need...", 
            placeholder="e.g., What is the average salary by department?",
            key="data_input_widget"
        )

        analyze_clicked = st.button("📈 Run Analysis", type="primary", key="data_search_btn")

        if analyze_clicked or st.session_state["trigger_data_search"]:
            if query:
                st.session_state["trigger_data_search"] = False
                
                with st.spinner("🔢 Calculating statistics..."):
                    try:
                        answer = get_data_answer(query)
                        
                        st.markdown(f"""
                        <div class="result-box">
                            <h4 style="margin-top:0; color: #67e8f9 !important;">📊 Data Insight:</h4>
                            <p style="font-size: 1.1rem; font-weight: 500; color: #e2e8f0 !important;">{answer}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error: {e}")

    with col_prompts:
        st.markdown("**⚡ Quick Analysis**")
        
        st.button("What is the average salary by Dept?", use_container_width=True,
                  on_click=trigger_data, args=("What is the average salary by department?",))
            
        st.button("How many employees per Dept?", use_container_width=True,
                  on_click=trigger_data, args=("How many employees are in each department?",))
            
        st.button("What is the attrition risk?", use_container_width=True,
                  on_click=trigger_data, args=("Show me the age distribution of employees.",))

    st.markdown('</div>', unsafe_allow_html=True)