#!/usr/bin/env python3
"""
api.py — FastAPI backend: retrieve from Qdrant, answer with Qwen3 via Ollama.

Reuses the retrieval logic from ingest.py so search behaves identically in the
API and on the CLI.

    uv pip install fastapi uvicorn qdrant-client
    uvicorn api:app --host 127.0.0.1 --port 8088   # run from backend/

Config via environment:
    OLLAMA_URL     default http://localhost:11434
    OLLAMA_MODEL   default qwen3:8b
    EMBED_MODEL    default bge-m3
    QDRANT_URL     default http://localhost:6333
    COLLECTION     default docs
    NUM_CTX        default 16384   <- must be large enough for TOP_K chunks
    TOP_K          default 5
    THINK          default 0       <- Qwen3 thinking mode
    WEB_DIR        default ../frontend/dist
"""

import json
import os
import sys
import types
import urllib.error
import urllib.request

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import ingest

OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen3:8b')
NUM_CTX = int(os.getenv('NUM_CTX', '16384'))
TOP_K = int(os.getenv('TOP_K', '5'))
THINK = os.getenv('THINK', '0') == '1'

SYSTEM_PROMPT = (
    'Responde preguntas sobre una colección de documentos. '
    'Escribe siempre en español.\n'
    'Reglas:\n'
    '1. Usa ÚNICAMENTE los fragmentos numerados de CONTEXTO. Nunca uses '
    'conocimiento externo.\n'
    '2. Cita los números de los fragmentos que utilices, así: [1] o [2][3].\n'
    '3. Si el CONTEXTO no contiene la respuesta, responde exactamente: '
    '"No se encuentra en los documentos." No inventes nada.\n'
    '4. Las tablas se dan en HTML. Léelas con cuidado y cita valores exactos.\n'
    '5. Sé conciso.'
)

# Namespace matching what ingest.search() reads off its args object.
RETRIEVAL = types.SimpleNamespace(
    collection=os.getenv('COLLECTION', ingest.COLLECTION),
    ollama=OLLAMA_URL,
    embed_model=os.getenv('EMBED_MODEL', ingest.EMBED_MODEL),
    qdrant=os.getenv('QDRANT_URL', ingest.QDRANT_URL),
    batch_size=16,
    timeout=300,
    limit=TOP_K,
    no_keyword=os.getenv('NO_KEYWORD', '0') == '1',
)

app = FastAPI(title='doc-rag')
router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None
    doc_id: str | None = None


def retrieve(question, top_k):
    args = types.SimpleNamespace(**vars(RETRIEVAL))
    args.limit = top_k or TOP_K
    hits = ingest.search(question, args)
    passages = []
    for i, (hit, score) in enumerate(hits, 1):
        p = hit.payload
        passages.append({
            'n': i,
            'score': round(score, 5),
            'doc_id': p.get('doc_id'),
            'page': p.get('page'),
            'heading_path': p.get('heading_path') or [],
            'has_table': p.get('has_table', False),
            'text': p.get('text', ''),
        })
    return passages


def build_messages(question, passages):
    blocks = []
    for p in passages:
        path = ' > '.join(p['heading_path'])
        header = f'[{p["n"]}] {p["doc_id"]}'
        if p['page']:
            header += f' página {p["page"]}'
        if path:
            header += f' — {path}'
        blocks.append(f'{header}\n{p["text"]}')
    context = '\n\n---\n\n'.join(blocks) if blocks else '(sin fragmentos)'
    return [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': f'CONTEXTO\n\n{context}\n\nPREGUNTA: {question}'},
    ]


def ollama_chat(messages, stream):
    body = json.dumps({
        'model': OLLAMA_MODEL,
        'messages': messages,
        'stream': stream,
        'think': THINK,
        'options': {'num_ctx': NUM_CTX, 'temperature': 0},
    }).encode()
    req = urllib.request.Request(f'{OLLAMA_URL}/api/chat', data=body,
                                 headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=600)


@router.get('/health')
def health():
    status = {'qdrant': 'down', 'ollama': 'down', 'collection': None}
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=RETRIEVAL.qdrant)
        info = client.get_collection(RETRIEVAL.collection)
        status['qdrant'] = 'ok'
        status['collection'] = {'name': RETRIEVAL.collection,
                                'points': info.points_count}
    except Exception as exc:
        status['qdrant_error'] = str(exc)[:200]

    try:
        with urllib.request.urlopen(f'{OLLAMA_URL}/api/tags', timeout=10) as resp:
            tags = [m['name'] for m in json.load(resp).get('models', [])]
        status['ollama'] = 'ok'
        status['models_present'] = {
            OLLAMA_MODEL: OLLAMA_MODEL in tags,
            RETRIEVAL.embed_model: any(t.startswith(RETRIEVAL.embed_model)
                                       for t in tags),
        }
    except Exception as exc:
        status['ollama_error'] = str(exc)[:200]

    status['config'] = {'model': OLLAMA_MODEL, 'num_ctx': NUM_CTX,
                        'top_k': TOP_K, 'think': THINK}
    if status['qdrant'] != 'ok' or status['ollama'] != 'ok':
        raise HTTPException(status_code=503, detail=status)
    return status


@router.get('/documents')
def documents():
    """Distinct doc_ids in the collection, with chunk counts."""
    from qdrant_client import QdrantClient
    client = QdrantClient(url=RETRIEVAL.qdrant)
    counts = {}
    offset = None
    while True:
        points, offset = client.scroll(
            RETRIEVAL.collection, limit=512, offset=offset,
            with_payload=['doc_id'], with_vectors=False)
        for point in points:
            doc = point.payload.get('doc_id')
            counts[doc] = counts.get(doc, 0) + 1
        if offset is None:
            break
    return {'documents': [{'doc_id': k, 'chunks': v}
                          for k, v in sorted(counts.items())]}


@router.post('/search')
def search_only(req: QueryRequest):
    """Retrieval with no generation — for debugging answer quality."""
    return {'question': req.question,
            'passages': retrieve(req.question, req.top_k)}


@router.post('/query')
def query(req: QueryRequest):
    passages = retrieve(req.question, req.top_k)
    try:
        with ollama_chat(build_messages(req.question, passages), stream=False) as r:
            payload = json.load(r)
    except urllib.error.URLError as exc:
        raise HTTPException(502, f'Ollama unreachable: {exc.reason}')
    return {
        'question': req.question,
        'answer': payload.get('message', {}).get('content', ''),
        'citations': [{k: p[k] for k in ('n', 'doc_id', 'page', 'score')}
                      for p in passages],
        'passages': passages,
    }


@router.post('/query/stream')
def query_stream(req: QueryRequest):
    """SSE: citations first, then answer tokens as they arrive."""
    passages = retrieve(req.question, req.top_k)

    def events():
        yield ('event: citations\ndata: '
               + json.dumps([{k: p[k] for k in
                              ('n', 'doc_id', 'page', 'score', 'heading_path',
                               'has_table', 'text')}
                             for p in passages]) + '\n\n')
        try:
            with ollama_chat(build_messages(req.question, passages),
                             stream=True) as resp:
                for raw in resp:
                    line = raw.decode('utf-8').strip()
                    if not line:
                        continue
                    chunk = json.loads(line)
                    delta = chunk.get('message', {}).get('content', '')
                    if delta:
                        yield f'event: token\ndata: {json.dumps(delta)}\n\n'
                    if chunk.get('done'):
                        break
        except Exception as exc:
            yield f'event: error\ndata: {json.dumps(str(exc)[:300])}\n\n'
        yield 'event: done\ndata: {}\n\n'

    return StreamingResponse(events(), media_type='text/event-stream',
                             headers={'Cache-Control': 'no-cache',
                                      'X-Accel-Buffering': 'no'})


# API lives under /api so the dev proxy and the production build use identical
# URLs. Mounted before the SPA so /api/* never falls through to StaticFiles.
app.include_router(router, prefix='/api')

# Serve the built frontend if present (npm run build -> frontend/dist).
# html=True gives SPA fallback: unknown paths return index.html.
WEB_DIR = os.getenv('WEB_DIR', '../frontend/dist')
_resolved = os.path.abspath(WEB_DIR)
if os.path.isfile(os.path.join(_resolved, 'index.html')):
    app.mount('/', StaticFiles(directory=_resolved, html=True), name='web')
    print(f'serving frontend from {_resolved}', flush=True)
else:
    # Loud on purpose: a missing or mistyped WEB_DIR otherwise shows up only as
    # a bare 404 at / with nothing in the log to explain it.
    print(f'WARNING: no frontend at {_resolved} '
          f'(WEB_DIR={WEB_DIR!r}, cwd={os.getcwd()!r}, '
          f'isdir={os.path.isdir(_resolved)}); serving API only',
          file=sys.stderr, flush=True)
