# RAG AI Agent

## Description
This repository contains an implementation of a Retrieval-Augmented Generation (RAG) AI Agent. The system combines information retrieval with a generative AI model to provide more accurate and context-aware responses. It integrates with **Groq AI** for enhanced AI-driven capabilities.

## Features
- Retrieves relevant documents from a knowledge base
- Integrates with **Groq AI** for enhanced response generation
- Uses vector databases for efficient search

## Project Structure
```
RAG_AI_AGENT/
│── data/        # Indexed knowledge base (ignored in Git)
│── src/         # Source code for retrieval and generation
│── venv/        # Virtual environment (ignored in Git)
│── .env         # Environment variables (ignored in Git)
│── .gitignore   # Git ignore file
│── README.md    # Project documentation
│── requirements.txt # Required dependencies
```

## Setup Instructions

### 1. Clone the Repository
```sh
git clone https://github.com/mithunveluru/RAG_AI_AGENT.git
cd RAG_AI_AGENT
```

### 2. Create and Activate a Virtual Environment
```sh
python -m venv venv
source venv/bin/activate  # For Mac/Linux
venv\Scripts\activate     # For Windows
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file and add your API keys:
```
GROQ_AI_API_KEY=your_groq_ai_api_key
VECTOR_DB_PATH=./data/vector_db
```

### 5. Run the AI Agent
```sh
python src/main.py
```

## Contributing
Feel free to fork the repository and submit pull requests with improvements.

## License
This project is licensed under the MIT License.

