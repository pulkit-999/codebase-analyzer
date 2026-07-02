import logging
import os
from dotenv import load_dotenv

# LangChain Vector & Embedding
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Core Tools & LLM
from langchain_core.tools import create_retriever_tool
from langchain_google_genai import ChatGoogleGenerativeAI

# Modern LangChain Agent
from langchain.agents import create_agent

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(message)s')

class ModernAgenticAnalyzer:
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "codebase_rag"):
        logging.info("Booting up local vector store...")
        
        # 1. Setup Local Retriever
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=db_path
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        
        # 2. Create the Search Tool
        self.retriever_tool = create_retriever_tool(
            self.retriever,
            name="search_codebase",
            description="Searches the codebase. Use this tool to look up function definitions, structs, or logic if you see a function call but don't have the context for what it does."
        )
        self.tools = [self.retriever_tool]
        
        # 3. Initialize LLM model
        logging.info("Connecting to Gemini...")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.0 
        )
        
        # 4. Build the Agent
        self.agent = self._build_agent()
        logging.info("Agent Ready!\n")

    def _build_agent(self):
        """Assembles the agent using the newest LangChain create_agent API."""
        system_prompt = (
            "You are a Senior Golang Engineer explaining architecture to a new developer. "
            "You have access to a tool called 'search_codebase'. "
            "If the user asks about a function, and that function relies on other custom structs or functions, "
            "YOU MUST use the 'search_codebase' tool to look them up before answering. "
            "Always cite the exact 'file_path' and line numbers when explaining code."
        )

        # The brand new wrapper replaces both create_tool_calling_agent AND create_react_agent
        return create_agent(
            model=self.llm, 
            tools=self.tools, 
            system_prompt=system_prompt
        )

    def chat_loop(self):
        print("="*60)
        print("🚀 Codebase RAG Active (Multi-Hop Enabled)")
        print("Type 'q' to exit.")
        print("="*60)
        
        while True:
            try:
                question = input("\nAsk about the codebase: ")
                if question.lower() in ['quit', 'q', 'exit']:
                    print("Shutting down...")
                    break
                if not question.strip():
                    continue

                print("\n🧠 Agent is analyzing and searching...")
                
                # LangGraph expects messages in this specific format
                response = self.agent.invoke(
                    {"messages": [("human", question)]}
                )
                
                # The final answer is always the content of the last message in the state
                raw_content = response["messages"][-1].content

                if isinstance(raw_content, list):
                    final_answer = "".join([block["text"] for block in raw_content if "text" in block])
                else:
                    final_answer = raw_content
                
                print("\n🤖 FINAL ANSWER:")
                print("-" * 50)
                print(final_answer)
                print("-" * 50)
                    
            except KeyboardInterrupt:
                print("\nShutting down...")
                break
            except Exception as e:
                print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    app = ModernAgenticAnalyzer()
    app.chat_loop()