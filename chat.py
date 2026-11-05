# import os
# import hashlib
# import tempfile
# from pathlib import Path
# from dotenv import load_dotenv
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_openai import OpenAIEmbeddings
# from langchain_qdrant import QdrantVectorStore
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from openai import OpenAI
# from qdrant_client import QdrantClient
# import streamlit as st

# load_dotenv()

# st.title("📄 PDF RAG Assistant")

# # Initialize client & models (same as your scripts)
# client = OpenAI()
# embedding_models = OpenAIEmbeddings(model="text-embedding-3-large")

# VECTORDB_URL = os.getenv("CLUSTER_URL","http://qdrant:6333")
# QDRANTDB_API_KEY = os.getenv("QDRANTDB_API_KEY",None)


# # --- PART 1: Document Upload & Indexing (Your First Script) ---
# uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

# if uploaded_file is not None:
#    file_bytes = uploaded_file.getvalue()
#    file_hash = hashlib.md5(file_bytes).hexdigest()
#    dynamic_collection_name = f"pdf_{file_hash}"
#    st.session_state["active_collection"] = dynamic_collection_name
   
#    q_client = QdrantClient(url=VECTORDB_URL, api_key=QDRANTDB_API_KEY)
        
#   # Index only once per uploaded file session to avoid redundant processing
#    if (
#       "indexed_file" in st.session_state
#       and st.session_state["indexed_file"] == uploaded_file.name
#    ) or q_client.collection_exists(dynamic_collection_name):
#     # Already exists either in browser memory OR on the Qdrant server!
#     st.session_state["indexed_file"] = uploaded_file.name
   
#    else:
  
#   # PyPDFLoader requires a physical file path, so we save the uploaded file temporarily
#       with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#           tmp_file.write(uploaded_file.getvalue())
#           pdf_path = tmp_file.name

  
    
#       with st.spinner("Indexing documents (wiping old collection & embedding)..."):
#         try:
        
#           loader = PyPDFLoader(file_path=pdf_path)
#           docs = loader.load()

#           # Chunkings (Your exact code)
#           text_splitter = RecursiveCharacterTextSplitter(
#               chunk_size=1000, chunk_overlap=200
#           )
#           split_docs = text_splitter.split_documents(documents=docs)

#           vector_store = QdrantVectorStore.from_documents(
#               documents=split_docs,
#               url=VECTORDB_URL,
#               api_key = QDRANTDB_API_KEY,
#               collection_name=dynamic_collection_name,
#               embedding=embedding_models,
#               batch_size=128,  # Uploads 128 chunks at a time
#               timeout=120,
#           )

#           st.session_state["indexed_file"] = uploaded_file.name
#           st.success("Indexing of documents done...")
#         finally:
#           if os.path.exists(pdf_path):
#             os.unlink(pdf_path)


# # --- PART 2: Querying & Chat UI (Your Second Script) ---

# # Initialize chat history state
# if "messages" not in st.session_state:
#   st.session_state["messages"] = []

# # Display prior chat history
# for message in st.session_state["messages"]:
#   with st.chat_message(message["role"]):
#     st.markdown(message["content"])

# # Take User Query via Streamlit chat input
# query = st.chat_input("Ask a question about your PDF...")

# if query:
#   if "active_collection" not in st.session_state:
#     st.warning("⚠️ Please upload and index a PDF file first before asking questions.")
#   else:
#     st.session_state["messages"].append({"role": "user", "content": query})
#     with st.chat_message("user"):
#       st.markdown(query)

#     with st.spinner("Searching document & generating response..."):
#       try:
#         current_collection = st.session_state["active_collection"]
#         # Vector similarity search [query] in DB (Your exact code)
#         vector_db = QdrantVectorStore.from_existing_collection(
#             url=VECTORDB_URL,
#             api_key = QDRANTDB_API_KEY,
#             collection_name=current_collection,
#             embedding=embedding_models,
#         )

#         search_results = vector_db.similarity_search(query=query)

#         context = "\n\n\n".join(
#             [
#                 f"page content: {result.page_content}\n Page Number: {result.metadata.get('page_label', 'N/A')} \n File Location: {result.metadata.get('source', 'N/A')}"
#                 for result in search_results
#             ]
#         )

#         SYSTEM_PROMPT = f"""
#   You are a helpful AI assistant who answers user query based on available context 
#   retrieved from a PDF file with page_contents along with page number.

#   You should only answer the user based on the following context and navigate the user to open the right page number to know more.

#     Context:
#     {context}
#   """

#         response_stream = client.chat.completions.create(
#             model="gpt-4.1",
#             messages=[
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user", "content": query},   
#             ],
#             stream=True,
#         )

#         def stream_generator():
#             for chunk in response_stream:
#               delta = chunk.choices[0].delta.content
#               if delta is not None:
#                 yield delta

#         with st.chat_message("assistant"):
#          answer = st.write_stream(stream_generator())
            
#       except Exception as e:
#         import traceback
#         print(traceback.format_exc())  # Prints full error in your terminal
#         answer = f"⚠️ Error details: {e}"
        
#         with st.chat_message("assistant"):
#             st.markdown(answer)

#       st.session_state["messages"].append(
#           {"role": "assistant", "content": answer}
#       )
    
import os
import hashlib
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from qdrant_client import QdrantClient
import streamlit as st

load_dotenv()

# --- STREAMLIT PAGE CONFIG & MODERN DARK UI STYLING ---
st.set_page_config(
    page_title="PDF RAG Intelligence Assistant",
    page_icon="⚡",
    layout="wide",
)

st.markdown("""
<style>
    /* Modern Dark Theme Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
        font-family: 'Inter', system-ui, sans-serif;
    }
    
    /* Centered Dashboard Title Banner */
    .dashboard-header-container {
        text-align: center;
        margin-bottom: 2rem;
    }

    .dashboard-header {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.2rem;
    }
    
    .dashboard-subtitle {
        color: #94a3b8;
        font-size: 1rem;
    }

    /* File Uploader Container */
    div[data-testid="stFileUploader"] {
        background-color: rgba(30, 41, 59, 0.7);
        border: 2px dashed #818cf8;
        border-radius: 12px;
        padding: 1rem;
    }

    /* User Chat Bubble */
    .user-bubble {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid #334155;
        border-left: 4px solid #38bdf8;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 1rem;
        color: #cbd5e1;
        margin-bottom: 0.75rem;
    }
    
    /* Catchy, Larger AI Assistant Response Bubble with Eye-Friendly Muted White Text */
    .ai-bubble {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.75) 0%, rgba(49, 46, 129, 0.55) 100%);
        border: 1px solid #6366f1;
        border-left: 5px solid #ec4899;
        border-radius: 12px;
        padding: 16px 20px;
        font-size: 1.15rem !important; /* Prominent, easy-to-read font size */
        font-weight: 400;
        line-height: 1.6;
        color: #e2e8f0; /* Soft, comfortable dull white for eye strain reduction */
        box-shadow: 0 8px 32px rgba(168, 85, 247, 0.15);
        margin-bottom: 1rem;
    }

    .badge-user {
        color: #38bdf8;
        font-weight: 700;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 3px;
    }

    .badge-ai {
        color: #f472b6;
        font-weight: 800;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Centered Dashboard Header
st.markdown("""
<div class="dashboard-header-container">
    <h1 class="dashboard-header">⚡ PDF RAG Intelligence Assistant</h1>
    <p class="dashboard-subtitle">Upload your PDF document and query context securely with live AI streaming.</p>
</div>
""", unsafe_allow_html=True)

# Initialize client & models
client = OpenAI()
embedding_models = OpenAIEmbeddings(model="text-embedding-3-small")  # Using smaller embedding model for faster indexing and lower cost

VECTORDB_URL = os.getenv("CLUSTER_URL", "http://qdrant:6333")
QDRANTDB_API_KEY = os.getenv("QDRANTDB_API_KEY", None)

# --- CACHED QDRANT CLIENT (Prevents SSL handshake timeouts & socket exhaustion) ---
@st.cache_resource
def get_qdrant_client(url, api_key):
    return QdrantClient(url=url, api_key=api_key, timeout=60.0)

q_client = get_qdrant_client(VECTORDB_URL, QDRANTDB_API_KEY)


# --- PART 1: Document Upload & Indexing ---
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    dynamic_collection_name = f"pdf_{file_hash}"
    st.session_state["active_collection"] = dynamic_collection_name
    
    try:
        collection_exists = q_client.collection_exists(dynamic_collection_name)
    except Exception:
        collection_exists = False

    if (
        "indexed_file" in st.session_state
        and st.session_state["indexed_file"] == uploaded_file.name
    ) or collection_exists:
        st.session_state["indexed_file"] = uploaded_file.name
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name

        with st.spinner("✨ Indexing documents into vector database..."):
            try:
                loader = PyPDFLoader(file_path=pdf_path)
                docs = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                split_docs = text_splitter.split_documents(documents=docs)

                QdrantVectorStore.from_documents(
                    documents=split_docs,
                    url=VECTORDB_URL,
                    api_key=QDRANTDB_API_KEY,
                    collection_name=dynamic_collection_name,
                    embedding=embedding_models,
                    batch_size=128,
                    timeout=120,
                )

                st.session_state["indexed_file"] = uploaded_file.name
                st.success("🎉 Document successfully indexed and ready for queries!")
            finally:
                if os.path.exists(pdf_path):
                    os.unlink(pdf_path)

st.divider()

# --- PART 2: Querying & Chat UI ---

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display prior chat history cleanly with custom styled modern cards
for message in st.session_state["messages"]:
    if message["role"] == "user":
        st.markdown(
            f"""
            <div class="user-bubble">
                <div class="badge-user">👤 You</div>
                {message["content"]}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="ai-bubble">
                <div class="badge-ai">🤖 AI Assistant</div>
                {message["content"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

# Take User Query via Streamlit chat input
query = st.chat_input("Ask a question about your PDF...")

if query:
    if "active_collection" not in st.session_state:
        st.warning("⚠️ Please upload and index a PDF file first before asking questions.")
    else:
        st.session_state["messages"].append({"role": "user", "content": query})
        
        st.markdown(
            f"""
            <div class="user-bubble">
                <div class="badge-user">👤 You</div>
                {query}
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.spinner("🔍 Searching document & generating response..."):
            try:
                current_collection = st.session_state["active_collection"]
                vector_db = QdrantVectorStore.from_existing_collection(
                    url=VECTORDB_URL,
                    api_key=QDRANTDB_API_KEY,
                    collection_name=current_collection,
                    embedding=embedding_models,
                )

                search_results = vector_db.similarity_search(query=query)

                context = "\n\n\n".join(
                    [
                        f"page content: {result.page_content}\n Page Number: {result.metadata.get('page_label', 'N/A')} \n File Location: {result.metadata.get('source', 'N/A')}"
                        for result in search_results
                    ]
                )

                SYSTEM_PROMPT = f"""
  You are a helpful AI assistant who answers user query based on available context 
  retrieved from a PDF file with page_contents along with page number.

  You should only answer the user based on the following context and navigate the user to open the right page number to know more.

    Context:
    {context}
  """

                response_stream = client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": query},   
                    ],
                    stream=True,
                )

                def stream_generator():
                    for chunk in response_stream:
                        delta = chunk.choices[0].delta.content
                        if delta is not None:
                            yield delta

                placeholder = st.empty()
                full_response = ""
                
                for chunk in stream_generator():
                    full_response += chunk
                    placeholder.markdown(
                        f"""
                        <div class="ai-bubble">
                            <div class="badge-ai">🤖 AI Assistant</div>
                            {full_response}▌
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                
                placeholder.markdown(
                    f"""
                    <div class="ai-bubble">
                        <div class="badge-ai">🤖 AI Assistant</div>
                        {full_response}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                answer = full_response
                
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                answer = f"⚠️ Error details: {e}"
                st.markdown(
                    f"""
                    <div class="ai-bubble" style="border-left-color: #ef4444;">
                        <div class="badge-ai" style="color: #ef4444;">⚠️ System Notice</div>
                        {answer}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.session_state["messages"].append(
                {"role": "assistant", "content": answer}
            )