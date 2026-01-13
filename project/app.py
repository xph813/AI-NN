import os
import json
import requests
from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import tiktoken
import re

app = Flask(__name__)

LOCAL_LLM_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3-vl:2b"

documents_data = {}
embeddings = None
index = None
model = None
chunk_to_doc_map = []

def extract_text_from_pdf(pdf_path):
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
        return text
    except ImportError:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

def chunk_text(text, max_tokens=512):
    paragraphs = re.split(r'\n\s*\n+', text)
    chunks = []
    encoding = tiktoken.get_encoding("cl100k_base")
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        paragraph_token_count = len(encoding.encode(paragraph))
        
        if paragraph_token_count <= max_tokens:
            chunks.append(paragraph)
        else:
            sentences = re.split(r'[.!?]+\s+', paragraph)
            current_chunk = ""
            token_count = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                sentence_token_count = len(encoding.encode(sentence))
                
                if token_count + sentence_token_count > max_tokens and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                    token_count = sentence_token_count
                else:
                    current_chunk += " " + sentence
                    token_count += sentence_token_count
            
            if current_chunk:
                chunks.append(current_chunk.strip())
    
    chunks = [chunk for chunk in chunks if chunk.strip()]
    
    final_chunks = []
    for chunk in chunks:
        chunk_token_count = len(encoding.encode(chunk))
        if chunk_token_count > max_tokens * 1.5:
            sub_chunks = [chunk[i:i+max_tokens*4] for i in range(0, len(chunk), max_tokens*4)]
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)
    
    return final_chunks

def load_documents():
    global documents_data, embeddings, index, model, chunk_to_doc_map
    
    print("Initializing sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    doc_folder = "doc/"
    if not os.path.exists(doc_folder):
        os.makedirs(doc_folder)
    pdf_files = [f for f in os.listdir(doc_folder) if f.lower().endswith('.pdf')]
    
    all_chunks = []
    chunk_to_doc_map = []
    
    for filename in pdf_files:
        filepath = os.path.join(doc_folder, filename)
        print(f"Processing {filename}...")
        
        text = extract_text_from_pdf(filepath)
        chunks = chunk_text(text)
        
        documents_data[filename] = {
            'text': text,
            'chunks': chunks,
            'title': os.path.splitext(filename)[0].replace('_', ' ')
        }
        
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_to_doc_map.append(filename)
    
    if all_chunks:
        print("Generating embeddings for all document chunks...")
        embeddings = model.encode(all_chunks)
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype('float32'))
        
        print(f"Successfully loaded {len(documents_data)} documents with {len(all_chunks)} chunks")

def search_similar_chunks(query, k=5):
    global model, index, chunk_to_doc_map
    
    if not model or not index:
        return []

    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)
    
    scores, indices = index.search(query_embedding.astype('float32'), k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(chunk_to_doc_map):
            doc_name = chunk_to_doc_map[idx]
            doc_data = documents_data.get(doc_name, {})
            doc_chunks = doc_data.get('chunks', [])
            chunk_idx = idx - chunk_to_doc_map.index(doc_name) if doc_name in chunk_to_doc_map else 0
            
            if 0 <= chunk_idx < len(doc_chunks):
                results.append({
                    'score': float(score),
                    'chunk': doc_chunks[chunk_idx],
                    'document': doc_name,
                    'text': doc_chunks[chunk_idx]
                })
    
    return results

def analyze_document_similarity():
    global model
    similarities = {}
    
    for doc1 in documents_data:
        similarities[doc1] = {}
        for doc2 in documents_data:
            if doc1 != doc2:
                doc1_embeddings = model.encode(documents_data[doc1]['chunks'])
                doc2_embeddings = model.encode(documents_data[doc2]['chunks'])
                
                avg_doc1_emb = np.mean(doc1_embeddings, axis=0).reshape(1, -1)
                avg_doc2_emb = np.mean(doc2_embeddings, axis=0).reshape(1, -1)
                
                sim_score = cosine_similarity(avg_doc1_emb, avg_doc2_emb)[0][0]
                similarities[doc1][doc2] = float(sim_score)
    
    return similarities

def generate_response(prompt):
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7
            }
        }
        response = requests.post(LOCAL_LLM_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get('response', "No response generated.")
    except Exception as e:
        return f"Error communicating with local LLM: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/documents')
def get_documents():
    docs = []
    for filename, data in documents_data.items():
        docs.append({
            'filename': filename,
            'title': data['title'],
            'chunk_count': len(data['chunks'])
        })
    return jsonify(docs)

@app.route('/api/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    results = search_similar_chunks(query)
    return jsonify(results)

@app.route('/api/ask', methods=['POST'])
def ask():
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'Question is required'}), 400
    
    relevant_chunks = search_similar_chunks(query, k=3)
    
    if not relevant_chunks:
        return jsonify({'response': 'No relevant information found in documents.'}), 404
    
    context = "\n\n".join([f"Document: {chunk['document']}\nText: {chunk['text'][:500]}..." 
                          for chunk in relevant_chunks])
    
    prompt = f"""
    Based on the following context from research documents, please answer the question.
    If the answer cannot be found in the context, please say so.
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:
    """
    
    response = generate_response(prompt)
    
    return jsonify({
        'response': response,
        'sources': [chunk['document'] for chunk in relevant_chunks]
    })

@app.route('/api/similarity-analysis', methods=['GET'])
def similarity_analysis():
    similarities = analyze_document_similarity()
    return jsonify(similarities)

@app.route('/api/research-suggestions', methods=['GET'])
def research_suggestions():
    if not documents_data:
        return jsonify({'suggestions': []}), 404
    
    all_titles = [data['title'] for data in documents_data.values()]
    
    prompt = f"""
    You are a research expert analyzing the following research papers: {', '.join(all_titles)}.
    
    Based on these research areas, please suggest:
    1. Potential research directions that combine multiple papers
    2. Open problems or gaps that could be addressed
    3. Methods or techniques that could be applied across domains
    
    Keep the suggestions relevant to the research topics covered in these papers.
    """
    
    suggestions = generate_response(prompt)
    
    return jsonify({
        'suggestions': suggestions,
        'analyzed_documents': [data['title'] for data in documents_data.values()]
    })

if __name__ == '__main__':
    print("Starting document loading process...")
    load_documents()
    print("Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=5000)
