import os
import logging
from pathlib import Path
from typing import List, Dict, Any

from git import Repo
import tree_sitter_rust as tsrust
import tree_sitter_go as tsgo
from tree_sitter import Language, Parser

# LangChain Imports
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodebaseIngestor:
    def __init__(self, repo_url: str, local_repo_dir: str, db_path: str = "./chroma_db", collection_name: str = "codebase_rag"):
        self.repo_url = repo_url
        self.local_repo_dir = local_repo_dir
        
        # 1. Initialize LangChain's HuggingFace local embedding wrapper
        logger.info("Loading local embedding model via LangChain...")
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 2. Initialize LangChain's Chroma wrapper
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_model,
            persist_directory=db_path
        )
        
        self.parsers = {
            ".rs": self._get_parser(tsrust.language()),
            ".go": self._get_parser(tsgo.language())
        }

    def _get_parser(self, pycapsule) -> Parser:
        """Initializes a Tree-sitter parser by wrapping the PyCapsule."""
        # 1. Wrap the raw C pointer (PyCapsule) into a Language object
        lang = Language(pycapsule)
        
        # 2. Initialize the parser and assign the language
        parser = Parser()
        parser.language = lang
        
        return parser

    def clone_repository(self):
        if not os.path.exists(self.local_repo_dir):
            logger.info(f"Cloning {self.repo_url}...")
            Repo.clone_from(self.repo_url, self.local_repo_dir)
        else:
            logger.info("Repository already cloned.")

    def chunk_file_ast(self, file_path: Path) -> List[Document]:
        """Parses AST blocks and returns LangChain Document objects directly."""
        ext = file_path.suffix
        if ext not in self.parsers:
            return []

        parser = self.parsers[ext]
        try:
            with open(file_path, "rb") as f:
                code_bytes = f.read()
        except Exception as e:
            logger.error(f"Read error {file_path}: {e}")
            return []

        tree = parser.parse(code_bytes)
        root_node = tree.root_node
        
        documents = []
        for child in root_node.children:
            node_type_str = str(child.type)
            if "function" in node_type_str or "method" in node_type_str or "declaration" in node_type_str or "type_spec" in node_type_str:
                chunk_text = code_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
                
                # Create a LangChain Document mapping text and metadata
                doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "file_path": str(file_path),
                        "node_type": node_type_str,
                        "start_line": int(child.start_point[0] + 1),
                        "end_line": int(child.end_point[0] + 1)
                    }
                )
                documents.append(doc)
        return documents

    def run_pipeline(self):
        self.clone_repository()
        total_documents = []
        
        for root, _, files in os.walk(self.local_repo_dir):
            if '.git' in root: continue
                
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix in self.parsers.keys():
                    docs = self.chunk_file_ast(file_path)
                    total_documents.extend(docs)
                    
                    if len(total_documents) >= 100:
                        # LangChain handles batch embeddings and IDs automatically!
                        self.vector_store.add_documents(total_documents)
                        total_documents = []
                        
        if total_documents:
            self.vector_store.add_documents(total_documents)
            
        logger.info("Ingestion complete via LangChain!")

if __name__ == "__main__":
    ingestor = CodebaseIngestor(
        repo_url="https://github.com/pulkit-999/task-processing-platform.git",
        local_repo_dir="./cloned_repos/task-processing-platform"
    )
    ingestor.run_pipeline()