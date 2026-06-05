from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
# Ensure the root package 'src' can be imported
import sys
import os
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.chunking import (
    FixedSizeChunker,
    SentenceChunker,
    RecursiveChunker,
    compute_similarity
)
from src.embeddings import _mock_embed, LocalEmbedder, OpenAIEmbedder
from src.store import EmbeddingStore
from src.agent import KnowledgeBaseAgent
from src.models import Document

app = Flask(__name__, template_folder="templates")
CORS(app)

# Global store initialized with local embedding fn by default
# We will dynamically re-create it if the provider changes
current_provider = "local"
embedder_instances = {
    "mock": _mock_embed
}

def get_embedder(provider):
    global embedder_instances
    provider = provider.lower().strip()
    if provider not in embedder_instances:
        if provider == "local":
            try:
                embedder_instances[provider] = LocalEmbedder()
            except Exception as e:
                print(f"Error loading LocalEmbedder: {e}")
                return _mock_embed
        elif provider == "openai":
            try:
                embedder_instances[provider] = OpenAIEmbedder()
            except Exception as e:
                print(f"Error loading OpenAIEmbedder: {e}")
                return _mock_embed
        else:
            return _mock_embed
    return embedder_instances[provider]

# Global store instance
global_store = EmbeddingStore(collection_name="web_ui_store", embedding_fn=get_embedder(current_provider))

def get_gemini_client():
    import os
    from google import genai
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your-gemini-api-key-here":
        return None
    return genai.Client(api_key=api_key)

def gemini_llm_fn(prompt):
    client = get_gemini_client()
    if not client:
        return "Lỗi: Chưa cấu hình GEMINI_API_KEY hợp lệ trong file .env"
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Lỗi gọi Gemini API: {str(e)}"

@app.route("/")
def index():
    # Đọc trực tiếp file index.html để tránh Jinja2 parse nhầm cú pháp React style={{...}}
    template_path = os.path.join(app.template_folder, "index.html")
    with open(template_path, encoding="utf-8") as f:
        content = f.read()
    from flask import Response
    return Response(content, mimetype="text/html")

@app.route("/api/chunk", methods=["POST"])
def api_chunk():
    data = request.json or {}
    text = data.get("text", "")
    chunk_size = int(data.get("chunk_size", 200))
    overlap = int(data.get("overlap", 20))
    max_sentences_per_chunk = int(data.get("max_sentences_per_chunk", 3))

    if not text:
        return jsonify({"error": "Văn bản rỗng"}), 400

    # Initialize chunkers
    fixed_chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
    sentence_chunker = SentenceChunker(max_sentences_per_chunk=max_sentences_per_chunk)
    recursive_chunker = RecursiveChunker(chunk_size=chunk_size)

    # Perform chunking
    fixed_chunks = fixed_chunker.chunk(text)
    sentence_chunks = sentence_chunker.chunk(text)
    recursive_chunks = recursive_chunker.chunk(text)

    # Compute statistics
    def get_stats(chunks):
        count = len(chunks)
        avg_len = sum(len(c) for c in chunks) / count if count > 0 else 0.0
        return {
            "count": count,
            "avg_length": round(avg_len, 2),
            "chunks": chunks
        }

    return jsonify({
        "fixed_size": get_stats(fixed_chunks),
        "by_sentences": get_stats(sentence_chunks),
        "recursive": get_stats(recursive_chunks)
    })

@app.route("/api/similarity", methods=["POST"])
def api_similarity():
    data = request.json or {}
    a = data.get("sentence_a", "")
    b = data.get("sentence_b", "")
    provider = data.get("provider", "mock")

    if not a or not b:
        return jsonify({"error": "Vui lòng nhập cả hai câu"}), 400

    try:
        embedder = get_embedder(provider)
        vec_a = embedder(a)
        vec_b = embedder(b)
        sim = compute_similarity(vec_a, vec_b)
        return jsonify({
            "similarity": sim,
            "vector_a_preview": vec_a[:5],
            "vector_b_preview": vec_b[:5],
            "dimensions": len(vec_a)
        })
    except Exception as e:
        return jsonify({"error": f"Lỗi tính similarity: {str(e)}"}), 500

@app.route("/api/store/add", methods=["POST"])
def api_store_add():
    global global_store, current_provider
    data = request.json or {}
    documents_data = data.get("documents", [])
    provider = data.get("provider", "mock")

    # If provider changed, reset/recreate store with new embedder
    if provider != current_provider:
        current_provider = provider
        global_store = EmbeddingStore(collection_name="web_ui_store", embedding_fn=get_embedder(provider))

    if not documents_data:
        return jsonify({"error": "Danh sách documents trống"}), 400

    try:
        docs = []
        for d in documents_data:
            doc_id = d.get("id", f"doc_{global_store.get_collection_size() + len(docs)}")
            content = d.get("content", "")
            metadata = d.get("metadata", {})
            if content.strip():
                docs.append(Document(id=doc_id, content=content, metadata=metadata))

        if docs:
            global_store.add_documents(docs)
        
        return jsonify({
            "success": True,
            "message": f"Đã thêm {len(docs)} documents thành công",
            "size": global_store.get_collection_size()
        })
    except Exception as e:
        return jsonify({"error": f"Lỗi lưu documents: {str(e)}"}), 500

@app.route("/api/store/search", methods=["POST"])
def api_store_search():
    data = request.json or {}
    query = data.get("query", "")
    top_k = int(data.get("top_k", 3))
    metadata_filter = data.get("metadata_filter", None)

    if not query:
        return jsonify({"error": "Query rỗng"}), 400

    try:
        if metadata_filter:
            # Clean empty filters
            clean_filter = {k: v for k, v in metadata_filter.items() if v}
            if clean_filter:
                results = global_store.search_with_filter(query, top_k=top_k, metadata_filter=clean_filter)
            else:
                results = global_store.search(query, top_k=top_k)
        else:
            results = global_store.search(query, top_k=top_k)

        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": f"Lỗi tìm kiếm: {str(e)}"}), 500

@app.route("/api/store/size", methods=["GET"])
def api_store_size():
    return jsonify({"size": global_store.get_collection_size()})

@app.route("/api/store/clear", methods=["POST"])
def api_store_clear():
    global global_store, current_provider
    # Reinitialize store to clear it
    global_store = EmbeddingStore(collection_name="web_ui_store", embedding_fn=get_embedder(current_provider))
    return jsonify({"success": True, "size": 0})

@app.route("/api/agent/query", methods=["POST"])
def api_agent_query():
    data = request.json or {}
    question = data.get("question", "")
    top_k = int(data.get("top_k", 3))

    if not question:
        return jsonify({"error": "Câu hỏi trống"}), 400

    try:
        agent = KnowledgeBaseAgent(store=global_store, llm_fn=gemini_llm_fn)
        answer = agent.answer(question, top_k=top_k)
        
        # Also retrieve source chunks so frontend can show them
        sources = global_store.search(question, top_k=top_k)

        return jsonify({
            "answer": answer,
            "sources": sources
        })
    except Exception as e:
        return jsonify({"error": f"Lỗi RAG Agent: {str(e)}"}), 500

@app.route("/api/store/upload", methods=["POST"])
def api_store_upload():
    global global_store
    if 'file' not in request.files:
        return jsonify({"error": "Không tìm thấy file"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Tên file rỗng"}), 400

    content = ""
    try:
        if file.filename.lower().endswith('.pdf'):
            import fitz
            doc = fitz.open(stream=file.read(), filetype="pdf")
            for page in doc:
                content += page.get_text() + "\n"
        else:
            content = file.read().decode('utf-8')
    except Exception as e:
        return jsonify({"error": f"Lỗi đọc file: {str(e)}"}), 500

    if not content.strip():
        return jsonify({"error": "File rỗng"}), 400

    try:
        chunk_size = int(request.form.get("chunk_size", 500))
        chunker = RecursiveChunker(chunk_size=chunk_size)
        chunks = chunker.chunk(content)
        docs = []
        for i, chunk in enumerate(chunks):
            doc_id = f"{file.filename}_{i}"
            docs.append(Document(id=doc_id, content=chunk, metadata={"source": file.filename}))
        
        global_store.add_documents(docs)
        return jsonify({
            "success": True,
            "message": f"Đã chia thành {len(chunks)} chunks và lưu vào Store.",
            "size": global_store.get_collection_size()
        })
    except Exception as e:
        return jsonify({"error": f"Lỗi xử lý chunking/lưu store: {str(e)}"}), 500

import time
@app.route("/api/benchmark", methods=["POST"])
def api_benchmark():
    data = request.json or {}
    questions = data.get("questions", [])
    top_k = int(data.get("top_k", 3))
    
    if not questions:
        return jsonify({"error": "Danh sách câu hỏi rỗng"}), 400
    
    results = []
    total_retrieval_time = 0
    total_gen_time = 0
    
    try:
        agent = KnowledgeBaseAgent(store=global_store, llm_fn=gemini_llm_fn)
        
        for q in questions:
            # Measure retrieval
            start_ret = time.time()
            chunks = global_store.search(q, top_k=top_k)
            ret_time = time.time() - start_ret
            total_retrieval_time += ret_time
            
            # Measure generation
            start_gen = time.time()
            answer = agent.answer(q, top_k=top_k)
            gen_time = time.time() - start_gen
            total_gen_time += gen_time
            
            results.append({
                "question": q,
                "retrieval_time_ms": round(ret_time * 1000, 2),
                "generation_time_ms": round(gen_time * 1000, 2),
                "retrieved_chunks": len(chunks)
            })
            
        avg_retrieval = total_retrieval_time / len(questions)
        avg_gen = total_gen_time / len(questions)
        
        return jsonify({
            "metrics": {
                "avg_retrieval_time_ms": round(avg_retrieval * 1000, 2),
                "avg_generation_time_ms": round(avg_gen * 1000, 2),
                "total_questions": len(questions)
            },
            "details": results
        })
    except Exception as e:
        return jsonify({"error": f"Lỗi Benchmark: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5000)
