import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

def load_and_chunk_documents(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks):
    embeddings = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=EMBEDDING_MODEL)
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store

def create_rag_chain(vector_store):
    llm = OllamaLLM(base_url=OLLAMA_BASE_URL, model=LLM_MODEL)
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa_chain

def create_conversational_rag_chain(vector_store):
    llm = OllamaLLM(base_url=OLLAMA_BASE_URL, model=LLM_MODEL)
    retriever = vector_store.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory)
    return qa_chain

app = Flask(__name__)
CORS(app)

# Global variables
rag_chain_sop = None
conversational_rag_chain_sop = None
current_question_sop = None

@app.route('/sop/query', methods=['POST'])
def handle_query():
    """Handles conversational queries."""
    global conversational_rag_chain_sop  # Use the conversational chain
    data = request.get_json()
    question = data.get('question')
    history = data.get('history', [])  # Get the conversation history from the request

    if not question:
        return jsonify({"error": "Missing question"}), 400

    try:
        # Pass the question and history to the conversational RAG chain
        result = conversational_rag_chain_sop.run({"question": question, "chat_history": history})
        return jsonify({"answer": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Global variables
rag_chain_faq = None
conversational_rag_chain_faq = None
current_question_faq = None    

@app.route('/faq/query', methods=['POST'])
def handle_query():
    """Handles conversational queries."""
    global conversational_rag_chain_faq  # Use the conversational chain
    data = request.get_json()
    question = data.get('question')
    history = data.get('history', [])  # Get the conversation history from the request

    if not question:
        return jsonify({"error": "Missing question"}), 400

    try:
        # Pass the question and history to the conversational RAG chain
        result = conversational_rag_chain_faq.run({"question": question, "chat_history": history})
        return jsonify({"answer": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    document_directory_sop = "./documents_sop"
    chunks_sop = load_and_chunk_documents(document_directory_sop)
    vector_store_sop = create_vector_store(chunks_sop)
    rag_chain_sop = create_rag_chain(vector_store_sop)
    conversational_rag_chain_sop = create_conversational_rag_chain(vector_store_sop)
    
    document_directory_faq = "./documents_faq"
    chunks_faq = load_and_chunk_documents(document_directory_faq)
    vector_store_faq = create_vector_store(chunks_faq)
    rag_chain_faq = create_rag_chain(vector_store_faq)
    conversational_rag_chain_faq = create_conversational_rag_chain(vector_store_faq)
    
    app.run(debug=True, host="0.0.0.0", port=6050)