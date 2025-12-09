import json
import os
import re
import requests
import uvicorn
import numpy as np
import time
import traceback
import asyncio
import httpx

# Suppress huggingface/tokenizers warning about forking
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, HTTPException, UploadFile, File, Body, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Generator
import uuid
import logging

# Global signals to abort active chat streams
abort_signals = {}

# Global Lock for Knowledge Base Access
KB_LOCK = asyncio.Lock()

# --- HELPER FUNCTIONS ---
# --- CONFIGURATION ---
# Centralize data storage to user's home directory
# This ensures both the App Bundle and the Executable share the same data
# and avoids "Read-only file system" errors in the App Bundle.
USER_DATA_DIR = os.path.join(os.path.expanduser("~"), ".cellami")
os.makedirs(USER_DATA_DIR, exist_ok=True)

SETTINGS_FILE = os.path.join(USER_DATA_DIR, "settings.json")
KB_FILE = os.path.join(USER_DATA_DIR, "knowledge_base.json")
MD_STORAGE_DIR = os.path.join(USER_DATA_DIR, "markdown_storage")
os.makedirs(MD_STORAGE_DIR, exist_ok=True)

OLLAMA_BASE_URL = "http://localhost:11434/api"

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "Cellami Backend is Running", "docs_url": "/docs"}

# Enable CORS for Office Add-in
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to the add-in's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware to prevent caching of index.html
@app.middleware("http")
async def add_no_cache_header(request, call_next):
    response = await call_next(request)
    if request.url.path in ["/", "/index.html"]:
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# Store background tasks
tasks: Dict[str, Dict[str, Any]] = {}
batch_tasks: Dict[str, Dict[str, Any]] = {}

# --- DATA MODELS ---

class Settings(BaseModel):
    config: Dict[str, Any]
    prompts: List[Dict[str, Any]]



class ChatMessage(BaseModel):
    role: str
    content: str
    images: Optional[List[str]] = None
    context_data: Optional[str] = None

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    temperature: Optional[float] = None
    use_rag: bool = False
    context_data: Optional[str] = None
    filtered_documents: Optional[List[str]] = None

class BatchRequest(BaseModel):
    inputs: List[str]
    prompt: str
    use_rag: bool = False
    source_only: bool = False
    search_query: Optional[str] = None # Legacy single query
    search_queries: Optional[List[str]] = None # List of queries for "provided" strategy
    refinement_strategy: str = "none" # "none", "auto", "provided"
    model: Optional[str] = None
    doc_filters: Optional[List[str]] = None # List of document names to INCLUDE

class RAGSource(BaseModel):
    source_id: int
    document_name: str
    chunk_text: str
    score: float

# --- SETTINGS MANAGEMENT ---

def load_settings() -> dict:
    if not os.path.exists(SETTINGS_FILE):
        return {"config": {}, "prompts": []}
    with open(SETTINGS_FILE, "r") as f:
        return json.load(f)

def save_settings(settings: dict):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)

def get_config_value(key: str, default: Any = None) -> Any:
    settings = load_settings()
    return settings.get("config", {}).get(key, default)

# --- KNOWLEDGE BASE MANAGEMENT ---

def load_kb() -> List[Dict]:
    if not os.path.exists(KB_FILE):
        return []
    with open(KB_FILE, "r") as f:
        return json.load(f)

def save_kb(kb_data: List[Dict]):
    with open(KB_FILE, "w") as f:
        json.dump(kb_data, f, indent=2)

def get_next_chunk_id(kb_data: List[Dict]) -> int:
    if not kb_data:
        return 1
    return max(item["id"] for item in kb_data) + 1

# --- OLLAMA API WRAPPERS ---

# Global embedding model
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from fastembed import TextEmbedding # Lazy import to speed up startup
        logger.info("Loading embedding model (nomic-ai/nomic-embed-text-v1.5)...")
        _embedding_model = TextEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5")
        logger.info("Embedding model loaded.")
    return _embedding_model

def get_local_embedding(text: str) -> List[float]:
    try:
        model = get_embedding_model()
        # embed returns a generator, we want the first (and only) item
        embeddings = list(model.embed([text]))
        return embeddings[0].tolist()
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return []

async def get_ollama_generate(prompt: str, model: str, system: str = "", temperature: float = 0.7, images: List[str] = None) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature}
    }
    if system:
        payload["system"] = system
    if images:
        payload["images"] = images

    try:
        async with httpx.AsyncClient(timeout=1200.0) as client:
            response = await client.post(f"{OLLAMA_BASE_URL}/generate", json=payload)
            response.raise_for_status()
            return response.json()["response"]
    except Exception as e:
        return f"Error: {str(e)}"

async def unload_model(model: str):
    # Signal active streams to stop
    abort_signals[model] = True
    
    try:
        # Use a short timeout to avoid blocking
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Try unloading via generate endpoint (with empty prompt as per docs)
            await client.post(f"{OLLAMA_BASE_URL}/generate", json={"model": model, "prompt": "", "keep_alive": 0})
            
            # Also try chat endpoint just in case
            await client.post(f"{OLLAMA_BASE_URL}/chat", json={"model": model, "messages": [], "keep_alive": 0})

            logger.info(f"Model {model} unload requested (forced).")
    except Exception as e:
        logger.error(f"Error unloading model: {e}")

async def get_ollama_chat_stream(messages: List[Dict], model: str, temperature: float = 0.7):
    logger.info(f"Starting chat stream for model {model}")
    # Reset abort signal for this model
    abort_signals[model] = False
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": temperature}
    }
    try:
        async with httpx.AsyncClient(timeout=1200.0) as client:
            async with client.stream("POST", f"{OLLAMA_BASE_URL}/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    # Check for explicit abort signal
                    if abort_signals.get(model, False):
                        logger.info(f"Abort signal received for {model}. Terminating stream.")
                        return # Exit generator, closing connection

                    if line:
                        try:
                            chunk = json.loads(line)
                            if "message" in chunk and "content" in chunk["message"]:
                                yield chunk["message"]["content"]
                            if chunk.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
    except asyncio.CancelledError:
        logger.info(f"Chat stream cancelled for model {model}")
        # Run unload in background to not block cancellation
        asyncio.create_task(unload_model(model))
        raise
    except Exception as e:
        logger.error(f"Error in chat stream: {e}")
        yield f"Error: {str(e)}"

# --- RAG LOGIC ---

def recursive_chunker(text: str, chunk_size: int = 6000) -> List[str]:
    # Standard "Split then Merge" strategy to avoid infinite loops and ensure greedy filling
    separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    
    def _split_text(text: str, separators: List[str]) -> List[str]:
        final_splits = []
        separator = ""
        next_seps = []
        
        # Find the first separator that exists in the text
        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                next_seps = separators[i+1:]
                break
        
        if not separator:
            # No separator found (or we are at char level)
            if len(text) > chunk_size:
                # Hard split by character
                return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            return [text]
            
        # Split and keep separator attached to the preceding segment
        parts = text.split(separator)
        
        for i, part in enumerate(parts):
            # Re-attach separator to all parts except the last one
            segment = part + separator if i < len(parts) - 1 else part
            if not segment:
                continue
            
            if len(segment) > chunk_size:
                # Segment is too big, recurse with lower-priority separators
                if next_seps:
                    final_splits.extend(_split_text(segment, next_seps))
                else:
                    # No more separators, hard split
                    final_splits.extend([segment[k:k+chunk_size] for k in range(0, len(segment), chunk_size)])
            else:
                final_splits.append(segment)
                
        return final_splits

    # Phase 1: Split into atomic blocks <= chunk_size
    atomic_splits = _split_text(text, separators)
    
    # Phase 2: Greedily merge blocks into chunks
    final_chunks = []
    current_chunk = ""
    
    for split in atomic_splits:
        if len(current_chunk) + len(split) <= chunk_size:
            current_chunk += split
        else:
            if current_chunk:
                final_chunks.append(current_chunk.strip())
            current_chunk = split
            
    if current_chunk:
        final_chunks.append(current_chunk.strip())
        
    return final_chunks



def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2:
        return 0.0
    a = np.array(v1)
    b = np.array(v2)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def find_relevant_context(query: str, top_k: int = 3, doc_filters: List[str] = None) -> Dict[str, Any]:
    kb_data = load_kb()
    if not kb_data:
        return {"context": "", "tag": ""}
    
    # Filter by document name if filters are provided
    if doc_filters is not None:
        kb_data = [item for item in kb_data if item["source"] in doc_filters]
        
    if not kb_data:
        return {"context": "", "tag": ""}
        
    # embedding_model = get_config_value("embedding_model_name", "nomic-embed-text:v1.5")
    query_vector = get_local_embedding(query)
    
    if not query_vector:
        return {"context": "Error: Could not embed query.", "tag": ""}
        
    scores = []
    for item in kb_data:
        score = cosine_similarity(query_vector, item["embedding"])
        scores.append({
            "id": item["id"],
            "text": item["text"],
            "source": item["source"],
            "score": score
        })
        
    scores.sort(key=lambda x: x["score"], reverse=True)
    top_results = scores[:top_k]
    
    context_parts = [f"- {res['text']}" for res in top_results]
    tag_parts = [f"{res['id']} ({int(res['score']*100)}%)" for res in top_results]
    
    context_str = "\n".join(context_parts)
    tag_str = f"(Source ID: {', '.join(tag_parts)})"
    
    return {"context": context_str, "tag": tag_str, "sources": top_results}

def clean_text(text: str) -> str:
    # Remove non-printable characters but keep newlines/tabs
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t\r")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- API ENDPOINTS ---

@app.get("/api/settings")
def get_settings():
    return load_settings()

@app.get("/api/list-models")
def list_models():
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/tags")
        if response.status_code == 200:
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            return {"models": models}
        else:
            return {"models": []}
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return {"models": []}

@app.post("/api/settings")
def update_settings(settings: Settings):
    save_settings(settings.dict())
    return {"status": "success"}




class UnloadRequest(BaseModel):
    model: str

@app.post("/api/unload-model")
async def unload_model_endpoint(request: UnloadRequest):
    logger.info(f"Received explicit unload request for {request.model}")
    # Run unload in background
    asyncio.create_task(unload_model(request.model))
    return {"status": "unloading"}

class BenchmarkRequest(BaseModel):
    model: str

@app.post("/api/benchmark-model")
async def benchmark_model(request: BenchmarkRequest):
    try:
        # Use a standardized prompt for consistent benchmarking
        prompt = "Write a short haiku about coding."
        
        payload = {
            "model": request.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7}
        }
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(f"{OLLAMA_BASE_URL}/generate", json=payload)
            response.raise_for_status()
            data = response.json()
        
        eval_count = data.get("eval_count", 0)
        eval_duration = data.get("eval_duration", 0) # in nanoseconds
        
        if eval_duration > 0:
            tps = eval_count / (eval_duration / 1e9)
            return {"model": request.model, "tps": round(tps, 2)}
        else:
            return {"model": request.model, "tps": 0}
            
    except Exception as e:
        logger.error(f"Benchmark failed for {request.model}: {e}")
        return {"model": request.model, "tps": 0, "error": str(e)}
    finally:
        # Ensure model is unloaded after benchmark to free up memory
        await unload_model(request.model)

@app.post("/api/chat")
async def chat(request: ChatRequest):
    settings = load_settings()
    config = settings.get("config", {})
    
    model = request.model or config.get("model_name")
    if not model:
        raise HTTPException(status_code=400, detail="No model selected. Please go to Settings and choose a model.")
    temp = request.temperature if request.temperature is not None else config.get("temperature", 0.7)
    
    # Convert Pydantic models to dicts for the wrapper
    messages_dicts = [m.dict(exclude_none=True) for m in request.messages]
    
    # Inject Context Data (Table Selection) if present
    # Iterate through all messages to merge context_data into content
    for msg in messages_dicts:
        if msg.get('context_data'):
            msg['content'] = (
                f"--- ATTACHED DATA ---\n{msg['context_data']}\n---------------------\n\n"
                f"{msg['content']}"
            )

    # Legacy: Handle top-level context_data (if sent separately, though we prefer it in the message now)
    if request.context_data:
        # If the last message is user, attach it there if not already attached
        if messages_dicts and messages_dicts[-1]['role'] == 'user':
            last_msg = messages_dicts[-1]
            # Check if we already attached it (avoid duplication if frontend sends it in both places)
            if not last_msg.get('context_data'):
                last_msg['content'] = (
                    f"--- ATTACHED DATA ---\n{request.context_data}\n---------------------\n\n"
                    f"{last_msg['content']}"
                )
    
    sources_metadata = []
    
    # Handle RAG
    # Handle RAG
    if request.use_rag and messages_dicts:
        # Find the last user message
        last_user_msg = next((m for m in reversed(messages_dicts) if m['role'] == 'user'), None)
        if last_user_msg:
            # 1. Construct Context for Query Generation
            # Get last few messages for context
            history_context = ""
            recent_msgs = messages_dicts[-3:] if len(messages_dicts) > 3 else messages_dicts
            for m in recent_msgs:
                role = m['role'].upper()
                content = m['content']
                # Truncate content if too long for the prompt
                if len(content) > 500:
                    content = content[:500] + "..."
                history_context += f"{role}: {content}\n"

            # Check for attached data in the last message
            attached_data = ""
            if request.context_data:
                 attached_data = request.context_data
            elif last_user_msg.get('context_data'):
                 attached_data = last_user_msg.get('context_data')
            
            # Truncate attached data for the query generation prompt to save tokens
            if attached_data and len(attached_data) > 1000:
                attached_data = attached_data[:1000] + "... [TRUNCATED]"

            # 2. Generate Search Query
            query_gen_prompt = (
                f"You are a research assistant. Your goal is to write a specific search query to find relevant background information "
                f"from the knowledge base that will help answer the user's latest question.\n\n"
                f"---CONVERSATION HISTORY---\n{history_context}\n\n"
                f"---ATTACHED DATA SUMMARY---\n{attached_data}\n\n"
                f"---INSTRUCTION---\n"
                f"Based on the conversation and data, write a specific search query to find external knowledge. "
                f"If the user asks to 'answer questions in the table', generate a query for the TOPICS mentioned in the table/questions. "
                f"Do not answer the question. Just output the search query."
            )

            try:
                # Use the same model (or a fast one) to generate the query
                # We use the sync function here as per existing patterns, though async would be better long-term
                generated_query = await get_ollama_generate(query_gen_prompt, model, temperature=0.3)
                generated_query = generated_query.strip().strip('"').strip("'")
                logger.info(f"RAG Query Expansion: '{last_user_msg['content'][:50]}...' -> '{generated_query}'")
                
                # 3. Execute Search with Generated Query
                rag_result = find_relevant_context(generated_query, doc_filters=request.filtered_documents)
                context = rag_result["context"]
                
                # Capture sources
                if "sources" in rag_result:
                    for s in rag_result["sources"]:
                        # Truncate text to avoid huge headers
                        snippet = s['text'][:100] + "..." if len(s['text']) > 100 else s['text']
                        sources_metadata.append({
                            "id": s['id'],
                            "source": s['source'],
                            "score": s['score'],
                            "text": snippet # snippet for preview
                        })

                if context:
                    # Inject RAG context
                    context_msg = {
                        "role": "system", 
                        "content": f"Use the following context from the knowledge base to help answer the user's question:\n\n{context}"
                    }
                    # Insert before the last message
                    messages_dicts.insert(-1, context_msg)
            
            except Exception as e:
                logger.error(f"RAG Query Expansion failed: {e}")
                # Fallback to original query
                pass
    

    headers = {}
    if sources_metadata:
        headers["X-Sources"] = json.dumps(sources_metadata)

    return StreamingResponse(get_ollama_chat_stream(messages_dicts, model, temp), media_type="text/plain", headers=headers)



async def generate_with_progress(prompt: str, model: str, system: str, task_id: str, timeout: int = 1200):
    logger.info(f"generate_with_progress called for task {task_id}")
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": True,
        "options": {"temperature": 0.7}
    }
    
    full_response = ""
    last_update = time.time()
    
    try:
        async with httpx.AsyncClient(timeout=float(timeout)) as client:
            async with client.stream("POST", f"{OLLAMA_BASE_URL}/generate", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    # Check for cancellation
                    if task_id in batch_tasks and batch_tasks[task_id].get("status") == "cancelled":
                        logger.info(f"Task {task_id} cancelled during generation.")
                        return full_response + " [CANCELLED]"
                    
                    # Check for explicit abort signal (Force Kill)
                    if abort_signals.get(model, False):
                        logger.info(f"Abort signal received for {model}. Terminating batch stream.")
                        return full_response + " [CANCELLED]"

                    if line:
                        try:
                            chunk = json.loads(line)
                            if "response" in chunk:
                                content = chunk["response"]
                                full_response += content
                                
                                # Update progress every 0.5 seconds
                                now = time.time()
                                if now - last_update > 0.5:
                                    batch_tasks[task_id]["message"] = f"Generating... ({len(full_response)} chars)"
                                    last_update = now
                                    
                            if chunk.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        logger.error(f"Error in generate_with_progress: {e}")
        raise e
        
    return full_response

async def run_batch_job(task_id: str, request: BatchRequest):
    start_time = time.time()
    settings = load_settings()
    config = settings.get("config", {})
    model = request.model or config.get("model_name")
    if not model:
        batch_tasks[task_id]["status"] = "failed"
        batch_tasks[task_id]["error"] = "No model selected. Please go to Settings and choose a model."
        return
    system = config.get("system_prompt", "")
    timeout_seconds = config.get("timeout_seconds", 1200) # Default 20 mins
    
    # Reset abort signal for this model to ensure we don't inherit a cancelled state
    if model in abort_signals:
        abort_signals[model] = False
    
    logger.info(f"Starting batch job {task_id} with {len(request.inputs)} inputs. Timeout: {timeout_seconds}s")
    results = []
    start_time = time.time()
    
    try:
        total_inputs = len(request.inputs)
        processed_count = 0
        
        # Update initial status
        batch_tasks[task_id].update({
            "total": total_inputs,
            "processed": 0
        })

        for i, text in enumerate(request.inputs):
            # Handle "provided" strategy here to ensure index alignment
            if request.use_rag and request.refinement_strategy == "provided" and request.search_queries:
                if i < len(request.search_queries):
                    # We'll use this in the RAG block
                    # But we need to pass it down. 
                    # Let's modify the loop variable 'text' to be a tuple or just handle it inside.
                    # Actually, let's just set a local variable 'current_query'
                    current_query = request.search_queries[i]
                else:
                    current_query = text # Fallback
            else:
                current_query = text # Default base

            # Check if text is valid
            if not text or not isinstance(text, str) or not text.strip():
                results.append({
                    "answer": "",
                    "sources": []
                })
                processed_count += 1
                batch_tasks[task_id]["processed"] = processed_count
                continue
                
            # Check for cancellation before processing next item
            # Check for cancellation before processing next item
            if batch_tasks[task_id].get("status") == "cancelled":
                logger.info(f"Batch job {task_id} cancelled.")
                await unload_model(model)
                break
            
            # Check for total job timeout
            if time.time() - start_time > timeout_seconds:
                logger.info(f"Batch job {task_id} timed out (exceeded {timeout_seconds}s).")
                batch_tasks[task_id]["status"] = "failed"
                batch_tasks[task_id]["error"] = f"Job timed out after {timeout_seconds} seconds."
                # The finally block will handle unloading
                break

            if request.use_rag:
                # Determine the query to use based on strategy
                # Default: Append cell text to prompt for context-aware search
                truncated_text = text[:500] if text else ""
                query_text = f"{request.prompt} {truncated_text}"
                logger.info(f"Batch RAG Query [{i}] (Default): {query_text}")
                
                if request.refinement_strategy == "none":
                    # Already set above
                    pass
                
                if request.refinement_strategy == "provided" and request.search_queries:
                    if i < len(request.search_queries):
                        query_text = request.search_queries[i]
                elif request.refinement_strategy == "auto":
                    # Generate refinement on the fly
                    refinement_prompt = (
                        f"You are a research assistant. Your goal is to write a specific search query to find relevant background information "
                        f"that will help answer the user's question about the provided data.\n\n"
                        f"---DATA---\n{text[:2000]}\n\n"
                        f"---USER QUESTION---\n{request.prompt}\n\n"
                        f"---INSTRUCTION---\n"
                        f"Based on the data and the question, enhance the user's question to allow for a more targeted search query to external knowledge. "
                        f"Do not answer the question. Just output the enhanced question. You will be penalized for every word that is not query related."
                    )
                    query_text = (await get_ollama_generate(refinement_prompt, model)).strip().strip('"').strip("'")
                elif request.search_query:
                    # Legacy fallback
                    query_text = request.search_query

                # Run RAG search in a separate thread to avoid blocking the event loop
                rag_result = await asyncio.to_thread(find_relevant_context, query_text, doc_filters=request.doc_filters)
                context = rag_result["context"]
                sources = rag_result.get("sources", [])
                
                if request.source_only:
                    # Return only the source IDs and scores
                    if sources:
                        source_tags = [f"{s['id']} ({round(s['score'] * 100)}%)" for s in sources]
                        source_text = f"(Source ID: {', '.join(source_tags)})"
                        results.append({
                            "answer": source_text,
                            "sources": []
                        })
                    else:
                        results.append({
                            "answer": "No relevant sources found.",
                            "sources": []
                        })
                else:
                    # Normal RAG with AI processing
                    if context:
                        full_prompt = (
                            f"You are a helpful assistant. Use the following context to answer the user's question. "
                            f"If the context doesn't contain the answer, say you don't know.\n"
                            f"---CONTEXT---\n{context}\n"
                            f"---QUESTION---\n{request.prompt} {text}"
                        )
                        raw_response = await generate_with_progress(full_prompt, model, system, task_id, timeout=timeout_seconds)
                        results.append({
                            "answer": raw_response,
                            "sources": sources
                        })
                    else:
                        results.append({
                            "answer": "No relevant context found.",
                            "sources": []
                        })
            else:
                # No RAG - direct AI processing
                full_prompt = f"{request.prompt}\n---\nTEXT: {text}\n\nQUESTION: {request.prompt}"
                response = await generate_with_progress(full_prompt, model, system, task_id, timeout=timeout_seconds)
                results.append({
                    "answer": response,
                    "sources": []
                })
            
            # Update progress
            processed_count += 1
            batch_tasks[task_id]["processed"] = processed_count
        
        logger.info(f"Batch job {task_id} loop finished normally.")
        duration = time.time() - start_time
        
        # Final status check
        final_status = "completed"
        if batch_tasks[task_id].get("status") == "cancelled":
            final_status = "cancelled"
            await unload_model(model)
            
        batch_tasks[task_id] = {
            "status": final_status,
            "results": results,
            "duration": duration,
            "total": total_inputs,
            "processed": processed_count
        }
        
    except asyncio.CancelledError:
        logger.info(f"Batch job {task_id} caught CancelledError!")
        await unload_model(model)
        batch_tasks[task_id] = {
            "status": "cancelled",
            "error": "Operation cancelled by user"
        }
        # Re-raise to ensure proper task cleanup if needed, though we handled it
        raise
    except Exception as e:
        logger.error(f"Error in batch job {task_id}: {str(e)}")
        traceback.print_exc()
        batch_tasks[task_id] = {
            "status": "failed",
            "error": str(e)
        }
    finally:
        # Ensure model is unloaded after batch job (success, failure, or timeout)
        await unload_model(model)

async def run_refinement_job(task_id: str, request: BatchRequest):
    logger.info(f"Starting refinement job {task_id}")
    try:
        settings = load_settings()
        config = settings.get("config", {})
        model = request.model or config.get("model_name")
        if not model:
            batch_tasks[task_id]["status"] = "failed"
            batch_tasks[task_id]["error"] = "No model selected. Please go to Settings and choose a model."
            return
        
        # Update initial status
        batch_tasks[task_id].update({
            "total": len(request.inputs),
            "processed": 0,
            "message": "Starting refinement..."
        })
        
        refined_queries = []
        # Pre-fill with None to maintain order
        refined_queries = [None] * len(request.inputs)
        
        # Process sequentially to ensure stability and avoid memory pressure
        semaphore = asyncio.Semaphore(1)

        async def process_item(index, item):
            async with semaphore:
                # Check cancellation
                if batch_tasks[task_id].get("status") == "cancelled":
                    return

                # Truncate item data if it's too huge
                item_data = item
                if len(item_data) > 2000:
                    item_data = item_data[:2000] + "... [TRUNCATED]"
                    
                refinement_prompt = (
                    f"You are a research assistant. Your goal is to write a specific search query to find relevant background information "
                    f"that will help answer the user's question about the provided data.\n\n"
                    f"---DATA---\n{item_data}\n\n"
                    f"---USER QUESTION---\n{request.prompt}\n\n"
                    f"---INSTRUCTION---\n"
                    f"Based on the data and the question, enhance the user's question to allow for a more targeted search query to external knowledge. "
                    f"Do not answer the question. Just output the enhanced question. You will be penalized for every word that is not query related."
                )
                
                try:
                    query = await get_ollama_generate(refinement_prompt, model)
                    result = query.strip().strip('"').strip("'")
                except Exception as e:
                    logger.error(f"Error refining item {index}: {e}")
                    result = request.prompt # Fallback
                
                refined_queries[index] = result
                
                # Update progress
                batch_tasks[task_id]["processed"] += 1
                batch_tasks[task_id]["message"] = f"Refining... ({batch_tasks[task_id]['processed']}/{len(request.inputs)})"

        await asyncio.gather(*(process_item(i, item) for i, item in enumerate(request.inputs)))
        
        if batch_tasks[task_id].get("status") == "cancelled":
            logger.info(f"Refinement job {task_id} cancelled.")
            return

        batch_tasks[task_id]["status"] = "completed"
        batch_tasks[task_id]["results"] = refined_queries
        logger.info(f"Refinement job {task_id} completed.")

    except Exception as e:
        logger.error(f"Refinement job failed: {e}")
        batch_tasks[task_id]["status"] = "failed"
        batch_tasks[task_id]["error"] = str(e)

@app.post("/api/refine-query")
async def refine_query(request: BatchRequest):
    logger.info(f"Received refine-query request. Inputs: {len(request.inputs)}")
    task_id = str(uuid.uuid4())
    
    batch_tasks[task_id] = {
        "status": "processing",
        "start_time": time.time(),
        "type": "refinement"
    }
    
    task = asyncio.create_task(run_refinement_job(task_id, request))
    batch_tasks[task_id]["async_task"] = task
    
    return {"task_id": task_id}

@app.post("/api/batch-process")
async def start_batch_process(request: BatchRequest):
    logger.info(f"Received batch process request. Inputs: {len(request.inputs)}")
    task_id = str(uuid.uuid4())
    
    batch_tasks[task_id] = {
        "status": "processing",
        "start_time": time.time()
    }
    
    # Use asyncio.create_task instead of BackgroundTasks so we can cancel it
    logger.info(f"Creating asyncio task for {task_id}")
    task = asyncio.create_task(run_batch_job(task_id, request))
    batch_tasks[task_id]["async_task"] = task
    
    return {"task_id": task_id}

@app.get("/api/batch-status/{task_id}")
async def get_batch_status(task_id: str):
    if task_id not in batch_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Create a clean copy without the async_task object which is not serializable
    status_copy = batch_tasks[task_id].copy()
    if "async_task" in status_copy:
        del status_copy["async_task"]
        
    return status_copy

@app.post("/api/cancel-batch/{task_id}")
async def cancel_batch(task_id: str):
    logger.info(f"Received cancellation request for {task_id}")
    if task_id not in batch_tasks:
        logger.warning(f"Task {task_id} not found in batch_tasks")
        raise HTTPException(status_code=404, detail="Task not found")
        
    batch_tasks[task_id]["status"] = "cancelled"
    logger.info(f"Set status to cancelled for {task_id}")
    
    # Cancel the asyncio task if it exists
    if "async_task" in batch_tasks[task_id]:
        logger.info(f"Found async_task for {task_id}, calling cancel()")
        task = batch_tasks[task_id]["async_task"]
        task.cancel()
    else:
        logger.warning(f"No async_task found for {task_id}")
        
    return {"status": "cancelled"}

def extract_text_from_file(file_path: str, filename: str) -> str:
    try:
        if filename.lower().endswith('.txt'):
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()

        from docling.document_converter import DocumentConverter # Lazy import
        
        try:
            converter = DocumentConverter()
            result = converter.convert(file_path)
            return result.document.export_to_markdown()
        except Exception as e:
            logger.error(f"Docling conversion failed: {e}")
            logger.error(traceback.format_exc())
            raise e
    except Exception as e:
        print(f"Error extracting text from {filename}: {e}")
        return ""


async def process_document_task(task_id: str, temp_path: str, filename: str):
    try:
        # Step 1: Text Extraction (10-20%)
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["message"] = "Extracting text from document..."
        tasks[task_id]["progress"] = 10
        await asyncio.sleep(0.5)  # Allow frontend to poll and see this update
        
        print(f"Task {task_id}: Extracting text from {filename}...")
        content = extract_text_from_file(temp_path, filename)
        
        if not content:
             raise ValueError("Could not extract text or unsupported file type")
        
        # Step 2: Converting to Markdown (20-25%)
        tasks[task_id]["message"] = "Converting to markdown format..."
        tasks[task_id]["progress"] = 20
        await asyncio.sleep(0.5)  # Allow frontend to poll and see this update
        print(f"Task {task_id}: Saving markdown...")
        
        # Save full markdown for source viewer
        # os.makedirs(MD_STORAGE_DIR, exist_ok=True) # Already created at startup
        md_path = os.path.join(MD_STORAGE_DIR, f"{filename}.md")
        with open(md_path, "w") as f:
            f.write(content)
        
        # Step 3: Chunking (25-35%)
        tasks[task_id]["message"] = "Analyzing document structure..."
        tasks[task_id]["progress"] = 25
        await asyncio.sleep(0.5)  # Allow frontend to poll and see this update
        print(f"Task {task_id}: Chunking content...")
            
        chunks = recursive_chunker(content)
        
        total_chunks = len(chunks)
        tasks[task_id]["message"] = f"Generated {total_chunks} text chunks"
        tasks[task_id]["progress"] = 35
        await asyncio.sleep(0.5)  # Allow frontend to poll and see this update
        
        print(f"Task {task_id}: Generated {total_chunks} chunks")
        print("--- Chunk Size Distribution ---")
        for i, c in enumerate(chunks):
            print(f"Chunk {i}: {len(c)} chars")
        print("-------------------------------")
        
        # Step 4: Embedding Generation (35-100%)
        kb_data = load_kb()
        next_id = get_next_chunk_id(kb_data)

        
        new_entries = []
        failed_chunks = 0
        failed_indices = []
        
        for i, chunk in enumerate(chunks):
            # Update progress (35% to 100% distributed across all chunks)
            progress = 35 + int((i / total_chunks) * 65)
            tasks[task_id]["progress"] = progress
            tasks[task_id]["message"] = f"Generating embeddings ({i+1}/{total_chunks})..."
            
            # Rate limiting
            await asyncio.sleep(0.1) 
            
            try:
                embedding = get_local_embedding(chunk)
                if embedding:
                    new_entries.append({
                        "id": next_id + i,
                        "source": filename,
                        "text": chunk,
                        "embedding": embedding
                    })
                else:
                    print(f"Task {task_id}: Failed to get embedding for chunk {i+1}")
                    failed_chunks += 1
                    failed_indices.append(i)
            except Exception as e:
                print(f"Task {task_id}: Error embedding chunk {i+1}: {e}")
                # Log problematic chunk content (truncated)
                print(f"Task {task_id}: Failed chunk content: {chunk[:100]}...")
                failed_chunks += 1
                failed_indices.append(i)

        # Critical Section: Atomic Update of Knowledge Base
        # We re-load the KB here to get the latest state (in case other tasks finished while we were embedding)
        async with KB_LOCK:
            current_kb_data = load_kb()
            # Recalculate IDs based on the *current* latest ID to avoid collisions
            start_id = get_next_chunk_id(current_kb_data)
            
            # Update IDs of our new entries to match the current sequence
            for i, entry in enumerate(new_entries):
                entry["id"] = start_id + i
                
            current_kb_data.extend(new_entries)
            save_kb(current_kb_data)
        
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 100
        
        if failed_chunks > 0:
            tasks[task_id]["message"] = f"Completed with warnings. Added {len(new_entries)} chunks. {failed_chunks} chunks failed."
            tasks[task_id]["result"] = {
                "chunks_added": len(new_entries), 
                "chunks_failed": failed_chunks,
                "failed_indices": failed_indices,
                "total_chunks": total_chunks
            }
        else:
            tasks[task_id]["message"] = f"Success! Added {len(new_entries)} chunks."
            tasks[task_id]["result"] = {
                "chunks_added": len(new_entries), 
                "chunks_failed": 0,
                "failed_indices": [],
                "total_chunks": total_chunks
            }
        
    except Exception as e:
        print(f"Task {task_id}: Failed with error: {e}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["message"] = str(e)
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/api/document-content")
def get_document_content(filename: str):
    md_path = os.path.join(MD_STORAGE_DIR, f"{filename}.md")
    if not os.path.exists(md_path):
        # Fallback: If file exists in KB but no markdown stored (old file), 
        # we could try to reconstruct or just return error.
        # For now, return 404 but maybe with a nice message?
        raise HTTPException(status_code=404, detail="Document content not found. Please re-upload this document.")
        
    with open(md_path, "r") as f:
        content = f.read()
        
    return {"content": content}

@app.post("/api/add-document")
async def add_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # Sync KB before adding new files to ensure clean state
    sync_knowledge_base()

    filename = file.filename
    task_id = str(uuid.uuid4())
    
    # Save file temporarily in system temp dir to avoid Read-only FS errors
    import tempfile
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"temp_{task_id}_{filename}")
    
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    # Initialize task
    tasks[task_id] = {
        "id": task_id,
        "status": "pending",
        "progress": 0,
        "message": "Starting...",
        "filename": filename
    }
    
    # Start background task
    background_tasks.add_task(process_document_task, task_id, temp_path, filename)
    
    return {"task_id": task_id}

@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]

def sync_knowledge_base():
    """
    Ensures that every document in the KB has a corresponding markdown file.
    If the markdown file is missing, remove the document from the KB.
    """
    kb_data = load_kb()
    if not kb_data:
        return

    # Identify unique sources in KB
    unique_sources = set(item["source"] for item in kb_data)
    
    # 1. Clean Ghost Entries (KB entry exists, but file missing)
    sources_to_remove = []
    for source in unique_sources:
        md_path = os.path.join(MD_STORAGE_DIR, f"{source}.md")
        if not os.path.exists(md_path):
            sources_to_remove.append(source)
            
    if sources_to_remove:
        print(f"Syncing KB: Removing {len(sources_to_remove)} ghost documents from DB: {sources_to_remove}")
        # Filter out chunks belonging to removed sources
        kb_data = [item for item in kb_data if item["source"] not in sources_to_remove]
        save_kb(kb_data)
        # Update unique sources after cleanup
        unique_sources = set(item["source"] for item in kb_data)

    # 2. Clean Orphaned Files (File exists, but KB entry missing)
    if os.path.exists(MD_STORAGE_DIR):
        for filename in os.listdir(MD_STORAGE_DIR):
            if filename.endswith(".md"):
                source_name = filename[:-3] # Remove .md extension
                if source_name not in unique_sources:
                    file_path = os.path.join(MD_STORAGE_DIR, filename)
                    try:
                        os.remove(file_path)
                        print(f"Syncing KB: Deleted orphaned file: {filename}")
                    except Exception as e:
                        print(f"Syncing KB: Failed to delete {filename}: {e}")

@app.get("/api/list-documents")
def list_documents():
    kb_data = load_kb()
    unique_docs = list(set(item["source"] for item in kb_data))
    return {"documents": unique_docs}

@app.delete("/api/remove-document")
def remove_document(filename: str = Body(..., embed=True)):
    kb_data = load_kb()
    initial_len = len(kb_data)
    kb_data = [item for item in kb_data if item["source"] != filename]
    
    if len(kb_data) == initial_len:
        raise HTTPException(status_code=404, detail="Document not found")
        
    save_kb(kb_data)
    
    # Also remove markdown file if exists
    md_path = os.path.join(MD_STORAGE_DIR, f"{filename}.md")
    if os.path.exists(md_path):
        os.remove(md_path)
        
    # Sync KB after removal to ensure consistency
    sync_knowledge_base()
        
    return {"status": "success"}

@app.post("/api/get-rag-sources")
def get_rag_sources(source_ids: List[int]):
    kb_data = load_kb()
    results = []
    
    # Create a lookup dict for faster access
    kb_dict = {item["id"]: item for item in kb_data}
    
    for sid in source_ids:
        if sid in kb_dict:
            item = kb_dict[sid]
            results.append({
                "source_id": item["id"],
                "document_name": item["source"],
                "chunk_text": item["text"]
            })
            
    return {"sources": results}

@app.get("/api/chunk/{chunk_id}")
def get_chunk(chunk_id: str):
    kb_data = load_kb()
    for item in kb_data:
        if str(item.get("id")) == chunk_id:
            return item
    raise HTTPException(status_code=404, detail="Chunk not found")

@app.get("/api/document-chunks")
def get_document_chunks(filename: str):
    kb_data = load_kb()
    chunks = [item for item in kb_data if item.get("source") == filename]
    
    if not chunks:
        return {"chunks": []}
        
    chunks.sort(key=lambda x: x["id"])
    
    return {"chunks": chunks}

# --- STATIC FILES SERVING ---
from fastapi.staticfiles import StaticFiles
import sys

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Mount assets folder (for icons in manifest)
assets_path = resource_path("assets")
if os.path.exists(assets_path):
    app.mount("/icons", StaticFiles(directory=assets_path), name="icons")
else:
    print(f"Warning: Assets folder not found at {assets_path}")

# Mount frontend dist folder
frontend_path = resource_path("frontend/dist")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")
else:
    print(f"Warning: Frontend dist folder not found at {frontend_path}. API mode only.")

# Configure logging
# Store logs in the centralized user data directory
LOG_FILE = os.path.join(USER_DATA_DIR, "cellami.log")

# Configure root logger for our app's messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode='w')
    ]
)
logger = logging.getLogger(__name__)

# --- ENVIRONMENT FIX FOR MACOS APP ---
# When running as a .app, PATH is restricted. We need to add common paths
# so that libraries wrapping external tools (like tesseract/poppler) can find them.
if sys.platform == "darwin":
    common_paths = [
        "/opt/homebrew/bin",
        "/usr/local/bin",
        "/usr/bin",
        "/bin",
        "/usr/sbin",
        "/sbin"
    ]
    current_path = os.environ.get("PATH", "")
    new_path = ":".join(common_paths) + ":" + current_path
    os.environ["PATH"] = new_path
    logger.info(f"Updated PATH for macOS App: {os.environ['PATH']}")
# -------------------------------------

# ... (rest of imports)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    
    import threading
    import pystray
    import subprocess
    from PIL import Image
    import copy
    
    # Configure Uvicorn logging to write to our file
    # We copy the default config and add our file handler
    log_config = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)
    log_config["handlers"]["file"] = {
        "class": "logging.FileHandler",
        "filename": LOG_FILE,
        "mode": "w",
        "formatter": "default",
    }
    # Add file handler to uvicorn loggers and disable propagation to avoid duplicates
    for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        if logger_name in log_config["loggers"]:
            if "handlers" in log_config["loggers"][logger_name]:
                log_config["loggers"][logger_name]["handlers"].append("file")
            else:
                log_config["loggers"][logger_name]["handlers"] = ["file"]
            
            # Prevent double logging (once by uvicorn, once by root logger)
            log_config["loggers"][logger_name]["propagate"] = False

    # Define server configuration with custom log_config
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info", log_config=log_config)
    server = uvicorn.Server(config)
    
    def resource_path(relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

    def run_server():
        server.run()
        
    def on_quit(icon, item):
        icon.stop()
        server.should_exit = True

    def on_show_logs(icon, item):
        """ Open a terminal window tailing the hidden log file """
        if sys.platform == "darwin":
            # macOS: Open in Console.app (native log viewer)
            # This avoids AppleScript permission issues and provides a better UI
            try:
                subprocess.run(["open", "-a", "Console", LOG_FILE])
            except Exception as e:
                print(f"Failed to open logs in Console: {e}")
                # Fallback to default handler
                subprocess.run(["open", LOG_FILE])
        elif sys.platform == "win32":
            # Windows: Use 'start' to open a new CMD window and PowerShell to tail
            # 'start' is a shell command, so shell=True is needed.
            # We use PowerShell's Get-Content -Wait which is equivalent to tail -f
            try:
                subprocess.run(f'start powershell -NoExit -Command "Get-Content -Wait \'{LOG_FILE}\'"', shell=True)
            except Exception as e:
                print(f"Failed to open logs: {e}")
        else:
            # Linux/Other: Try generic x-terminal-emulator or similar (fallback)
            try:
                subprocess.Popen(["x-terminal-emulator", "-e", f"tail -f {LOG_FILE}"])
            except Exception:
                print("Platform not supported for automatic log viewing.")
        
    def setup_tray():
        try:
            # Use resource_path to find the icon
            icon_path = resource_path(os.path.join("assets", "Cellami_Template.png"))
            if not os.path.exists(icon_path):
                print(f"Warning: Icon not found at {icon_path}. Tray icon might be blank.")
                # Create a simple colored square as fallback
                image = Image.new('RGB', (64, 64), color = (79, 70, 229)) # Indigo color
            else:
                image = Image.open(icon_path)
                # Resize for better quality if it's huge
                image.thumbnail((64, 64), Image.Resampling.LANCZOS)
                
            # Create menu with "Show Logs" and "Quit"
            menu = pystray.Menu(
                pystray.MenuItem("Show Logs", on_show_logs),
                pystray.MenuItem("Quit Cellami", on_quit)
            )
            icon = pystray.Icon("Cellami", image, "Cellami", menu)
            icon.run()
        except Exception as e:
            print(f"Failed to setup tray icon: {e}")
            # Fallback to just running server if tray fails
            if not server.started:
                server.run()

    # Start server in a separate thread
    server_thread = threading.Thread(target=run_server)
    server_thread.start()
    
    # Run tray icon in main thread (blocking)
    setup_tray()
