import glob
import json
import logging
import os
import re
import time
import uuid
from typing import List, Optional

import nest_asyncio
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from llama_index.core import (
    QueryBundle,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike
from pydantic import BaseModel

app = FastAPI()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")  # noqa: E501
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

nest_asyncio.apply()


def get_sorted_chunks(folder_path):
    chunk_files = glob.glob(os.path.join(folder_path, "chunk_*.txt"))

    def get_chunk_number(filename):
        match = re.search(r"chunk_(\d+)", filename)
        return int(match.group(1)) if match else 0

    sorted_files = sorted(chunk_files, key=get_chunk_number)
    return sorted_files


def process_text_file(input_file, output_folder, chunk_size=256):
    os.makedirs(output_folder, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    while "\n\n" in content:
        content = content.replace("\n\n", "\n")

    words = content.split()
    chunks = [
        words[i : i + chunk_size] for i in range(0, len(words), chunk_size)  # noqa: E203  # noqa: E203
    ]  # noqa: E203

    for i, chunk in enumerate(chunks):
        output_file = os.path.join(output_folder, f"chunk_{i+1}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(" ".join(chunk))


Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
embed_model = OpenAIEmbedding(model="text-embedding-3-large",)

URL = os.getenv("BACKEND_URL", "http://localhost:8003")

Settings.llm = OpenAILike(
    model="mistral-tensorrtllm",
    api_base=f"{URL}/v1",
    api_key="fake",
    is_chat_model=True,
    context_window=2048,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)





try:
    logger.info("Loading index...")
    storage_context = StorageContext.from_defaults(persist_dir="./storage/aiact")
    aia_index = load_index_from_storage(storage_context)
    logger.info("Index loaded successfully.")
except Exception as e:
    logger.info("Failed to load index: " + str(e))
    logger.info("Creating index...")
    process_text_file("constant/aiact.txt", "data/source", 128)
    aia_act = SimpleDirectoryReader(input_files=get_sorted_chunks("data/source"), filename_as_id=True).load_data()
    logger.info(f"Loaded {len(aia_act)} documents")
    node_parser = SemanticSplitterNodeParser(
        embed_model=embed_model,
        buffer_size=5,
        breakpoint_percentile_threshold=95
    )
    nodes = node_parser.build_semantic_nodes_from_documents(aia_act, show_progress=True)
    logger.info(f"Built {len(nodes)} nodes")
    for node in nodes:
        node.set_content(node.get_content()[:8192])
    logger.info(f"Set content for {len(nodes)} nodes")
    vector_index = VectorStoreIndex(nodes)
    logger.info("Vector index created")
    aia_index = VectorStoreIndex.from_documents(aia_act, show_progress=True)
    aia_index.storage_context.persist(persist_dir="./storage/aiact")
    logger.info("Index created successfully.")


engine = aia_index.as_chat_engine(
    similarity_top_k=1,
    retriever_mode="embedding",
    chat_mode="context",
    llm=Settings.llm,
    streaming=True,
    verbose=True,
)


def warmup_engine():
    url = f"{URL}/v1/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "mistral-tensorrtllm",  # change to mistral-instruct
        "prompt": "Where is New York?",
        "max_tokens": 1,
        "temperature": 0,
    }
    trials = 10
    for i in range(trials):
        try:
            _ = requests.post(url, headers=headers, json=payload)
            logger.info(f"Warmup trial {i} successful")
            break
        except Exception as e:
            logger.error(f"Warmup trial {i} failed: {e}; Waiting 10 seconds before retrying...")
            time.sleep(10)


class QueryRequest(BaseModel):
    query: str


warmup_engine()


class Prompt(BaseModel):
    newMessage: str
    messages: list[Optional[str]]


def message_generator(stream):
    if stream:
        for chunk in stream:
            unique_id = str(uuid.uuid4())
            yield f"event: data\ndata: {json.dumps({'content': chunk, 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': f'run-{unique_id}', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []})}\n\n"  # noqa: E501
        yield "event: end"


@app.post("/api2/openai/stream/")
@app.options("/api2/openai/stream/")
async def query_endpoint(request: Request):
    try:
        data = json.loads(await request.body())
        data = json.loads(data["input"])
        prompt = Prompt(**data)
        response = engine.stream_chat(prompt.newMessage)
        return StreamingResponse(message_generator(response.response_gen), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
