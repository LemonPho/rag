#!/usr/bin/env python3
"""
ingest.py — validate chunks, embed via Ollama's bge-m3, upsert into Qdrant, query.

Embeddings come from Ollama (`/api/embed`), so nothing here holds VRAM and there
is no torch dependency. Ollama loads and unloads bge-m3 on demand.

Tradeoff: Ollama exposes only bge-m3's DENSE output, not its sparse/lexical head.
To recover literal-token matching we keep a Qdrant full-text index on the chunk
text and fuse a keyword-filtered pass with the dense pass (client-side RRF).

    python3 ingest.py chunks.jsonl --check      # structural checks; no services
    python3 ingest.py chunks.jsonl --recreate   # embed + upsert
    python3 ingest.py --query "..."             # hybrid-ish search
"""

import argparse
import collections
import json
import re
import statistics
import sys
import urllib.error
import urllib.request

OLLAMA_URL = 'http://localhost:11434'
EMBED_MODEL = 'bge-m3'
QDRANT_URL = 'http://localhost:6333'
COLLECTION = 'docs'
DENSE_DIM = 1024

TABLE_OPEN = re.compile(r'<table[\s>]', re.IGNORECASE)
TABLE_CLOSE = re.compile(r'</table\s*>', re.IGNORECASE)
WORD_RE = re.compile(r'[\w./-]{3,}')

# ---------------------------------------------------------------- embeddings


def embed(texts, args):
    """Embed a list of strings via Ollama. Returns list of 1024-dim vectors."""
    out = []
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start:start + args.batch_size]
        body = json.dumps({'model': args.embed_model, 'input': batch}).encode()
        req = urllib.request.Request(f'{args.ollama}/api/embed', data=body,
                                     headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                payload = json.load(resp)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode('utf-8', 'replace')[:300]
            sys.exit(f'Ollama embed failed ({exc.code}): {detail}\n'
                     f'is {args.embed_model!r} present? try: ollama list')
        except urllib.error.URLError as exc:
            sys.exit(f'cannot reach Ollama at {args.ollama}: {exc.reason}')

        vectors = payload.get('embeddings')
        if not vectors:
            sys.exit(f'unexpected Ollama response: {str(payload)[:300]}')
        out.extend(vectors)
        if len(texts) > args.batch_size:
            print(f'  embedded {min(start + len(batch), len(texts))}/{len(texts)}',
                  end='\r', flush=True)

    if len(texts) > args.batch_size:
        print()
    if out and len(out[0]) != DENSE_DIM:
        sys.exit(f'expected {DENSE_DIM}-dim vectors, got {len(out[0])} — '
                 f'is {args.embed_model!r} really bge-m3?')
    return out


# ---------------------------------------------------------------- validation


def load(path):
    records = []
    with open(path, encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                sys.exit(f'{path}:{lineno}: bad JSON: {exc}')
    if not records:
        sys.exit(f'{path}: no records')
    return records


def check(records, min_tokens, max_tokens):
    problems = 0

    def fail(msg):
        nonlocal problems
        problems += 1
        print(f'  FAIL {msg}')

    print(f'{len(records)} chunks from '
          f'{len(set(r["doc_id"] for r in records))} document(s)')

    if [r for r in records if not r['text'].strip()]:
        fail('chunk(s) with empty text')

    ids = collections.Counter(r['id'] for r in records)
    dupes = {k: v for k, v in ids.items() if v > 1}
    if dupes:
        fail(f'{len(dupes)} duplicate id(s) — chunks would overwrite each other')

    tokens = sorted(r['tokens'] for r in records)
    print(f'  tokens: min {tokens[0]}  median {int(statistics.median(tokens))}  '
          f'max {tokens[-1]}')

    tiny = [r for r in records if r['tokens'] < min_tokens and not r['has_table']]
    if tiny:
        print(f'  WARN {len(tiny)} chunk(s) under {min_tokens} tokens '
              f'(too small to retrieve well)')

    huge = [r for r in records if r['tokens'] > max_tokens and not r['has_table']]
    if huge:
        fail(f'{len(huge)} non-table chunk(s) over {max_tokens} tokens')

    over = [r for r in records if r['tokens'] > 8192]
    if over:
        fail(f'{len(over)} chunk(s) over 8192 tokens — bge-m3 will TRUNCATE these')

    split_tables = [r for r in records
                    if len(TABLE_OPEN.findall(r['text']))
                    != len(TABLE_CLOSE.findall(r['text']))]
    if split_tables:
        fail(f'{len(split_tables)} chunk(s) with unbalanced <table> — '
             f'a table got split')

    print(f'  {sum(1 for r in records if r["has_table"])} chunk(s) contain tables')

    if [r for r in records if r.get('heading_path') and r['embed_text'] == r['text']]:
        fail('heading_path missing from embed_text')

    print('  OK' if not problems else f'  {problems} problem(s)')
    return problems


# ---------------------------------------------------------------- qdrant


def upsert(records, args):
    from qdrant_client import QdrantClient, models

    print(f'embedding {len(records)} chunks via Ollama ({args.embed_model}) ...')
    vectors = embed([r['embed_text'] for r in records], args)

    client = QdrantClient(url=args.qdrant)
    exists = client.collection_exists(args.collection)
    if exists and args.recreate:
        print(f'deleting existing collection {args.collection!r}')
        client.delete_collection(args.collection)
        exists = False
    if not exists:
        client.create_collection(
            args.collection,
            vectors_config=models.VectorParams(size=DENSE_DIM,
                                               distance=models.Distance.COSINE),
        )
        client.create_payload_index(
            args.collection, field_name='doc_id',
            field_schema=models.PayloadSchemaType.KEYWORD)
        # Full-text index recovers literal-token matching without sparse vectors.
        client.create_payload_index(
            args.collection, field_name='text',
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                min_token_len=2, max_token_len=25, lowercase=True))
        print(f'created collection {args.collection!r} (dense + full-text index)')

    client.upsert(args.collection, points=[
        models.PointStruct(
            id=r['id'],
            vector=vectors[i],
            payload={
                'text': r['text'],
                'doc_id': r['doc_id'],
                'page': r.get('page'),
                'heading_path': r.get('heading_path'),
                'chunk_index': r.get('chunk_index'),
                'has_table': r.get('has_table', False),
            },
        )
        for i, r in enumerate(records)
    ])
    info = client.get_collection(args.collection)
    print(f'upserted {len(records)} points; collection holds {info.points_count}')


def search(text, args):
    """Dense search fused with a keyword-filtered pass (client-side RRF)."""
    from qdrant_client import QdrantClient, models

    vector = embed([text], args)[0]
    client = QdrantClient(url=args.qdrant)
    depth = args.limit * 4

    dense = client.query_points(args.collection, query=vector, limit=depth,
                                with_payload=True).points

    keyword = []
    if not args.no_keyword:
        terms = WORD_RE.findall(text)
        if terms:
            flt = models.Filter(should=[
                models.FieldCondition(key='text', match=models.MatchText(text=t))
                for t in terms
            ])
            keyword = client.query_points(args.collection, query=vector,
                                         query_filter=flt, limit=depth,
                                         with_payload=True).points

    # Reciprocal rank fusion.
    scores, seen = collections.defaultdict(float), {}
    for ranking in (dense, keyword):
        for rank, hit in enumerate(ranking, 1):
            scores[hit.id] += 1.0 / (60 + rank)
            seen[hit.id] = hit

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(seen[pid], score) for pid, score in ordered[:args.limit]]


def query(text, args):
    hits = search(text, args)
    print(f'Q: {text}\n')
    if not hits:
        print('no results — is the collection populated?')
        return
    for i, (hit, score) in enumerate(hits, 1):
        p = hit.payload
        path = ' > '.join(p.get('heading_path') or []) or '(no heading)'
        flag = ' [TABLE]' if p.get('has_table') else ''
        print(f'{i}. rrf={score:.4f}  {p["doc_id"]} p{p.get("page")}{flag}')
        print(f'   {path}')
        print(f'   {" ".join(p["text"].split())[:220]}...\n')


# ---------------------------------------------------------------- cli


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('chunks', nargs='?', help='chunks.jsonl from chunker.py')
    ap.add_argument('--check', action='store_true',
                    help='structural validation only; no services needed')
    ap.add_argument('--query', help='search instead of ingesting')
    ap.add_argument('--recreate', action='store_true',
                    help='drop and rebuild the collection first')
    ap.add_argument('--no-keyword', action='store_true',
                    help='dense only; skip the full-text pass')
    ap.add_argument('--collection', default=COLLECTION)
    ap.add_argument('--ollama', default=OLLAMA_URL)
    ap.add_argument('--embed-model', default=EMBED_MODEL)
    ap.add_argument('--qdrant', default=QDRANT_URL)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--timeout', type=int, default=300)
    ap.add_argument('--limit', type=int, default=5)
    ap.add_argument('--min-tokens', type=int, default=50)
    ap.add_argument('--max-tokens', type=int, default=1200)
    args = ap.parse_args()

    if args.query:
        query(args.query, args)
        return

    if not args.chunks:
        ap.error('chunks.jsonl required unless --query is used')

    records = load(args.chunks)
    problems = check(records, args.min_tokens, args.max_tokens)

    if args.check:
        sys.exit(1 if problems else 0)
    if problems:
        sys.exit('\nrefusing to ingest with structural problems')

    print()
    upsert(records, args)


if __name__ == '__main__':
    main()
