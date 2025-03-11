import os
from typing import List
from chromadb.utils.embedding_functions import create_langchain_embedding
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


class Doc_Loader:
    
    def __init__(self,
                  doc_path : str,
                  client_type : str,
                  verctordb_path : str,
                  collection_name : str,
                  sep : str,
                  chunk_size : int,
                  chunk_overlap : int,
                  embeddings) -> None:
        
        
        self.doc_path =  doc_path
        self.client_type =  client_type
        self.verctordb_path =  verctordb_path
        self.collection_name =  collection_name
        self.separator =  sep
        self.chunk_size = chunk_size 
        self.chunk_overlap = chunk_overlap
        self.embedding_function = embeddings
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        
        
    def get_text_splitter(self,
                          sep: str,
                          chunk_size: int,
                          chunk_overlap: int) -> RecursiveCharacterTextSplitter:
        
        txt_splitter = RecursiveCharacterTextSplitter(
                            chunk_size = self.chunk_size,
                            chunk_overlap = self.chunk_overlap,
                            length_function = len,
                            is_separator_regex = False
                        )
        return txt_splitter
    
    
    def get_langchain_embedding(self) -> List[List[float]]:
        return create_langchain_embedding(self.embedding_function)
    
    def doc_reader(self) -> list:
        
        book_path = os.path.join(self.current_dir, "documents")
        book_files = [f for f in os.listdir(book_path) if f.endswith(".pdf")]
        
        documents = []
        for book_file in book_files:
            file_path = os.path.join(book_path, book_file)
            loader = PyMuPDFLoader(file_path)
            book_docs = loader.load()
            for doc in book_docs:
                documents.append(doc)
                
        return documents
        
        
    def create_update_vectorstore(self, file = "") -> None:
            
        text_splitter = self.get_text_splitter(self.separator,
                                               self.chunk_size,
                                               self.chunk_overlap)
        
        chunked_documents = text_splitter.split_documents(self.doc_reader())
        
        presistent_directory = os.path.join(self.current_dir, "Vector_database", self.verctordb_path)
        
        Chroma.from_documents(
            documents=chunked_documents,
            embedding=self.get_langchain_embedding(),
            collection_name=self.collection_name,
            persist_directory=presistent_directory
        )
        
    def get_vector_store(self):
        
        presistent_directory = os.path.join(self.current_dir, "Vector_database", self.verctordb_path)
        
        vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=presistent_directory
        )
        
        return vector_store
        
        
    def del_source_file(self, file_name: str) -> None:
        
        vector_store = self.get_vector_store()
        
        file_path = os.path.join(self.current_dir, "documents", file_name)
        vector_store.delete(where={'source': file_path})
        
        print(f"Deleted all chunks related to: {file_path}")