import os
import re
import streamlit as st
from Class_loader import Doc_Loader
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState, StateGraph
from langgraph.graph import END, START
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

sep = "\n"
chunk_size = 1000
chunk_overlap = 100

model_name = "sentence-transformers/all-mpnet-base-v2"
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs
)

current_dir = os.path.dirname(os.path.abspath(__file__))
collection_name = os.getenv('CHROMA_COLLECTION_NAME', 'default_collection_name')
pdf_directory = os.getenv("PDF_DIRECTORY", "uploaded_pdfs")  # Default folder if env var is missing

book_path = os.path.join(current_dir, pdf_directory)
os.makedirs(book_path, exist_ok=True)  # Ensure the directory exists

# Initialize document loader
docloader = Doc_Loader(
    book_path,
    os.getenv("CLIENT_TYPE"),
    os.getenv("VECTORDB_DIRECTORY"),
    collection_name,
    sep,
    chunk_size,
    chunk_overlap,
    embeddings,
)

# Sidebar for API key
with st.sidebar:
    st.title('💬 RAG Chat')
    if 'GROQ_API_KEY' in st.secrets:
        st.success('GROQ API Key is already provided!', icon='✅')
        os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY']
    else:
        groq_api_key = st.text_input("GROQ API KEY", type="password")
        if groq_api_key:
            os.environ['GROQ_API_KEY'] = groq_api_key
            st.success('Proceed to entering your prompt message!', icon='👉')
        else:
            st.warning('Please enter your credentials!', icon='⚠️')

# File Upload Handling
st.header("📂 Upload PDF(s) for Analysis")
uploaded_files = st.file_uploader("Upload PDF(s):", accept_multiple_files=True, type="pdf")

if uploaded_files:
    if "uploaded_filenames" not in st.session_state:
        st.session_state["uploaded_filenames"] = set()

    new_files_uploaded = False  # Flag to check if new files are added

    for uploaded_file in uploaded_files:
        file_path = os.path.join(book_path, uploaded_file.name)
        if uploaded_file.name not in st.session_state["uploaded_filenames"]:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            st.session_state["uploaded_filenames"].add(uploaded_file.name)
            new_files_uploaded = True

    if new_files_uploaded:
        st.toast("New PDF(s) uploaded successfully!")
        docloader.create_update_vectorstore()
        st.session_state["vector_store"] = docloader.get_vector_store()
        st.toast("Vector store updated successfully! You can now ask questions.")
    else:
        st.info("No new files were uploaded.")

# Function to Generate Answers
def generate(state: MessagesState):
    """Generate an AI response based on retrieved documents."""
    llm = ChatGroq(model="deepseek-r1-distill-llama-70b")

    if "vector_store" not in st.session_state:
        return {"messages": [{"role": "assistant", "content": "No vector store available. Please upload documents first."}]}

    vector_store = st.session_state["vector_store"]
    
    recent_user_messages = [msg for msg in reversed(state["messages"]) if msg.type == "human"]
    
    if not recent_user_messages:
        return {"messages": [{"role": "assistant", "content": "Please provide a valid query."}]}

    query = recent_user_messages[-1].content
    retrieved_docs = vector_store.similarity_search(query, k=5)

    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\nPage Number: {doc.metadata.get('page', 'unknown')}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )

    system_message_content = (
        "You are an AI assistant that retrieves answers from the provided data. "
        "Always include the 'Source' and 'Page number' of the chunk from which the answer was derived. "
        "If the answer is not found in the given chunks, respond with 'I don't know.' "
        "Do not generate information beyond the provided data. "
        "Here are some relevant document chunks to help answer the question:\n\n"
        f"{serialized}"
    )

    conversation_messages = [
        msg for msg in state["messages"]
        if msg.type in ("human", "system") or (msg.type == "ai" and not msg.tool_calls)
    ]
    
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    response = llm.invoke(prompt)
    response.content = re.sub(r"<think>.*?</think>\s*", "", response.content, flags=re.DOTALL)

    return {"messages": [response]}

# Graph Configuration
def build_graph():
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("generate", generate)
    graph_builder.add_edge(START, "generate")
    graph_builder.add_edge("generate", END)
    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory)

def build_config():
    return {"configurable": {"thread_id": "abc123"}}
    
# Initialize Graph in Session State
if "graph" not in st.session_state:
    st.session_state["graph"] = build_graph()
    
if 'config' not in st.session_state:
    st.session_state['config'] = build_config()

graph = st.session_state["graph"]
config = st.session_state["config"]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display Chat History
for message_pair in st.session_state.chat_history:
    with st.chat_message(message_pair["role"]):
        st.markdown(message_pair["content"])

# Chat Input and Processing
user_input = st.chat_input(placeholder="Ask AI...", disabled=not ('GROQ_API_KEY' in os.environ))

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        response_message = None
        with st.spinner("Thinking..."):
            try:
                for step in graph.stream(
                    {"messages": [{"role": "user", "content": user_input}]},
                    stream_mode="values",
                    config=config,
                ):
                    response_message = step
                assistant_response = response_message["messages"][-1].content
                st.markdown(assistant_response)
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
            except Exception as e:
                st.error(f"Error generating response: {e}")
