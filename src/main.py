import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import groq
import numpy as np

load_dotenv()

class EnhancedAIAgent:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("Error: GROQ_API_KEY not found in environment variables.")
        
        self.client = groq.Client(api_key=self.groq_api_key)
        self.model = "llama3-70b-8192"
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.ingested_content = []
        self.ingested_embeddings = []

    def process_query(self, query: str) -> str:
        if self.ingested_content:
            return self.query_document(query)
        return self.simple_query(query)

    def simple_query(self, query: str) -> str:
        chat_completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ],
            max_tokens=512
        )
        return chat_completion.choices[0].message.content

    def query_document(self, query: str) -> str:
        query_embedding = self.get_embedding(query)
        best_chunk = self.find_most_relevant_chunk(query_embedding)
        
        if not best_chunk:
            return "I couldn't find relevant information in the document."

        prompt = f"Based on the following document excerpt, answer the question:\n\n{best_chunk}\n\nQuestion: {query}"
        return self.simple_query(prompt)

    def get_embedding(self, text: str):
        return self.embedding_model.encode(text, convert_to_numpy=True)

    def find_most_relevant_chunk(self, query_embedding):
        if not self.ingested_embeddings:
            return None
        
        similarities = np.dot(self.ingested_embeddings, query_embedding)
        best_match_idx = np.argmax(similarities)
        return self.ingested_content[best_match_idx]

    def ingest_file(self, file_path: str):
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == '.pdf':
            content = self._extract_text_from_pdf(file_path)
        else:
            content = self._extract_text_from_txt(file_path)

        if content:
            chunks = self._split_text(content)
            self.ingested_content.extend(chunks)
            self.ingested_embeddings.extend([self.get_embedding(chunk) for chunk in chunks])
            print(f"File '{file_path}' ingested successfully!")

    def _extract_text_from_pdf(self, file_path):
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return None

    def _extract_text_from_txt(self, file_path):
        encodings = ['utf-8', 'latin-1', 'cp1252']
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError:
                continue
        print(f"Could not decode file {file_path} with any encoding.")
        return None

    def _split_text(self, text, chunk_size=400):
        words = text.split()
        return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def main():
    print("Initializing AI Agent...")
    
    try:
        agent = EnhancedAIAgent()
    except ValueError as e:
        print(e)
        return
    
    print("\nWould you like to ingest files? (y/n)")
    if input().lower() == 'y':
        while True:
            print("\nEnter file path (or 'q' to quit file ingestion):")
            file_path = input()
            if file_path.lower() == 'q':
                break
            try:
                agent.ingest_file(file_path)
            except Exception as e:
                print(f"Error ingesting file: {e}")
    
    print("\nAI Agent ready! Enter your queries (type 'exit' to quit):")
    while True:
        query = input("> ")
        if query.lower() == 'exit':
            break
        try:
            response = agent.process_query(query)
            print("\nResponse:", response)
        except Exception as e:
            print(f"Error processing query: {e}")

if __name__ == "__main__":
    main()

