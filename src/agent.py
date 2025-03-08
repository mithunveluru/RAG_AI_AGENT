import os
from typing import List, Dict, Any, Optional
import json
import math
from operator import add, sub, mul, truediv

# LangChain imports
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Get Groq API key from environment
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq LLM
llm = ChatGroq(
    model="llama3-70b",
    api_key=groq_api_key
)

# Define function calling tools
@tool
def add_numbers(a: float, b: float) -> float:
    return add(a, b)

@tool
def subtract_numbers(a: float, b: float) -> float:
    return sub(a, b)

@tool
def multiply_numbers(a: float, b: float) -> float:
    return mul(a, b)

@tool
def divide_numbers(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return truediv(a, b)

@tool
def generate_summary(text: str) -> str:
    summarization_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert summarizer. Create a concise summary of the following text:"),
        ("human", "{text}")
    ])
    summarization_chain = summarization_prompt | llm | StrOutputParser()
    return summarization_chain.invoke({"text": text})

@tool
def generate_notes(text: str) -> List[str]:
    notes_prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract 5-7 key points from the following text as a list:"),
        ("human", "{text}")
    ])
    notes_chain = notes_prompt | llm | StrOutputParser()
    notes = notes_chain.invoke({"text": text})
    return notes.split("\n")

class AIAgent:
    def __init__(self):
        self.llm = llm
        self.tools = [
            add_numbers, subtract_numbers, multiply_numbers, divide_numbers,
            generate_summary, generate_notes
        ]
        self.embeddings = OpenAIEmbeddings()
        self.vector_db = None
        self.documents = []
        self.chat_history = []
    
    def ingest_file(self, file_path: str) -> None:
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        elif file_path.endswith('.csv'):
            loader = CSVLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_documents = text_splitter.split_documents(documents)
        self.documents.extend(split_documents)
        
        if self.vector_db is None:
            self.vector_db = Chroma.from_documents(documents=split_documents, embedding=self.embeddings)
        else:
            self.vector_db.add_documents(split_documents)
        
        print(f"Successfully ingested {file_path} ({len(split_documents)} chunks)")
    
    def _retrieve_relevant_context(self, query: str, k: int = 4) -> List[str]:
        if self.vector_db is None:
            return []
        docs = self.vector_db.similarity_search(query, k=k)
        return docs
    
    def _create_rag_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant with reasoning capabilities.
            Use the following context to answer the user's question.
            Context: {context}"""),
            ("human", "{question}")
        ])
        rag_chain = (
            {"context": lambda x: self._retrieve_relevant_context(x["question"]), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain
    
    def process_query(self, query: str) -> str:
        self.chat_history.append(HumanMessage(content=query))
        import re
        math_pattern = r'(\d+)\s*([\+\-\*/])\s*(\d+)'
        math_match = re.search(math_pattern, query)
        
        if math_match:
            num1, operation, num2 = float(math_match.group(1)), math_match.group(2), float(math_match.group(3))
            if operation == '+':
                result = num1 + num2
            elif operation == '-':
                result = num1 - num2
            elif operation == '*':
                result = num1 * num2
            elif operation == '/' and num2 != 0:
                result = num1 / num2
            else:
                return "Cannot divide by zero."
            return f"The result of {num1} {operation} {num2} is {result}"
        else:
            rag_chain = self._create_rag_chain()
            answer = rag_chain.invoke({"question": query})
        
        self.chat_history.append(AIMessage(content=answer))
        return answer

# Example usage
def main():
    agent = AIAgent()
    agent.ingest_file("sample_data.pdf")
    agent.ingest_file("financial_data.csv")
    print("AI Agent ready! Type 'quit' to exit.")
    while True:
        user_query = input("Your question: ")
        if user_query.lower() == 'quit':
            break
        response = agent.process_query(user_query)
        print(f"\nAI: {response}\n")

if __name__ == "__main__":
    main()

