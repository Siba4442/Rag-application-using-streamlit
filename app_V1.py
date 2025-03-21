import os
import streamlit as st 
from dotenv import load_dotenv
import re
import tempfile
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState, StateGraph
from langgraph.graph import END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq

def api_is_valid(api_key: str) -> bool:
    """Check if the API key follows the expected pattern."""
    if not isinstance(api_key, str):
        return False
    pattern = r"^gsk_[A-Za-z0-9]{52}$"
    return bool(re.match(pattern, api_key))

def pdf_reader(uploadedPDF: list) -> list:
    """Reads uploaded PDFs and extracts text."""
    documents = []
    
    for pdf in uploadedPDF:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf.read())  # Save the uploaded file locally
            temp_file_path = temp_file.name  # Get the temporary file path

        loader = PyMuPDFLoader(temp_file_path)
        docs = loader.load()

        documents.extend(docs)
        os.remove(temp_file_path)  # Clean up after processing

    return documents

def pdf_chunk(document: list):
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(document)


def create_update_vectorstore(chunked_documents, index_name):
    """Creates or updates a Pinecone vector store with document embeddings."""
    model_name = "sentence-transformers/all-mpnet-base-v2"
    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs=encode_kwargs
    )
    
    vector_store = PineconeVectorStore.from_documents(
        documents=chunked_documents,
        embedding=embeddings,
        index_name=index_name
    )

    return vector_store

def initialize_pinecone(api_key: str):
    """Initializes Pinecone and returns an index object."""
    pc = Pinecone(api_key=api_key)
    index_name = "rag-chat"

    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=768,  # Adjust the dimension based on your embedding model
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(index_name)

def main():
    """Main function for Streamlit app."""
    load_dotenv()

    st.session_state.setdefault('file_uploader_key', 0)
    st.session_state.setdefault("uploaded_pdfs", [])

    with st.sidebar:
        st.title('💬 RAG Chat')

        # Handle GROQ API key
        if 'GROQ_API_KEY' in st.secrets:
            st.success("GROQ API Key is already provided!", icon='✅')
            os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY']
        else:
            groq_api_key = st.text_input('GROQ API KEY', type='password')
            if not groq_api_key:
                st.warning("Please enter your credentials", icon='⚠️')
            elif not api_is_valid(groq_api_key):
                st.warning("Please enter a valid API KEY", icon='⚠️')
            else:
                os.environ['GROQ_API_KEY'] = groq_api_key
                st.success("Proceed to entering your prompt message!", icon='👉')

        # Handle Pinecone API key
        pc = None
        if 'PINECONE_API_KEY' in st.secrets:
            st.success("PINECONE API Key is already provided!", icon='✅')
            pine_api_key = st.secrets['PINECONE_API_KEY']
            pc = initialize_pinecone(pine_api_key)
        else:
            pine_api_key = st.text_input('PINECONE API KEY', type='password')
            if not pine_api_key:
                st.warning("Please enter your credentials", icon='⚠️')
            else:
                os.environ['PINECONE_API_KEY'] = pine_api_key
                pc = initialize_pinecone(pine_api_key)
                st.success("Pinecone initialized!", icon='👉')

    pdf_docs = st.file_uploader(
        "Upload your Data here in PDF format and click on 'Process'", 
        accept_multiple_files=True, 
        type=['pdf']
    )

    if st.button("Process"):
        if not pdf_docs:
            st.warning("Please upload at least one PDF file.")
        elif pc is None:
            st.warning("Pinecone is not initialized. Please enter a valid API key.")
        else:
            with st.spinner("Processing..."):
                raw_text = pdf_reader(pdf_docs)
                chunks = pdf_chunk(raw_text)
                st.success("Your data has been processed successfully!")

                index_name = "rag-chat"  # Pinecone index name
                st.session_state["vector_store"] = create_update_vectorstore(chunks, index_name)
                st.success("Pinecone database has been created successfully!")
                
    
    def generate(state: MessagesState):
        """Generate answer."""
        llm = ChatGroq(model="deepseek-r1-distill-llama-70b")
        
        if "vector_store" not in st.session_state:
           return {"messages": [{"role": "assistant", "content": "No vector store available. Please upload documents first."}]}

        vector_store = st.session_state["vector_store"]
        
        recent_user_messages = [msg for msg in reversed(state["messages"]) if msg.type == "human"]
        
        if not recent_user_messages:
            return {"messages": [{"role": "assistant", "content": "Please provide a valid query."}]}

        query = recent_user_messages[-1].content
        retrieved_docs = vector_store.similarity_search(query, k=5)

        # Format retrieved documents
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


    def build_graph():
        
        graph_builder = StateGraph(MessagesState)

        graph_builder.add_node("generate", generate)

        graph_builder.add_edge(START, "generate")
        graph_builder.add_edge("generate", END)

        memory = MemorySaver()
        graph = graph_builder.compile(checkpointer=memory)

        return graph

    def build_config():
        config = {"configurable": {"thread_id": "abc123"}}
        return config
            
    if "graph" not in st.session_state:
        st.session_state["graph"] = build_graph()
        
    if 'config' not in st.session_state:
        st.session_state['config'] = build_config()

    graph = st.session_state["graph"]
    config = st.session_state["config"]

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message_pair in st.session_state.chat_history:
        with st.chat_message(message_pair["role"]):
            st.markdown(message_pair["content"])

    user_input = st.chat_input(placeholder="Ask AI...", disabled= not(groq_api_key and pine_api_key))

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

if __name__ == '__main__':
    main()
