#!/usr/bin/env python3
"""
chunker.py — split cleaned OCR markdown into chunks for embedding.

Input is the output of `clean_det.py --combine`: one .md per document with
`<!-- page N -->` markers. Emits JSONL, one record per chunk:

    {"id", "doc_id", "page", "heading_path", "chunk_index", "text", "embed_text",
     "tokens", "has_table"}

Design notes for this corpus:
  * Heading depth comes from section numbering ("3.4.2." -> depth 3), because
    the OCR emits '#' for every level.
  * HTML tables stay in one chunk even if oversized, with the preceding
    "Table N | ..." caption attached so the chunk is retrievable at all.
  * Table-of-contents pages are detected and skipped.

    python3 chunker.py outputs/*.md > chunks.jsonl
"""

import argparse
import hashlib
import json
import pathlib
import re
import sys
import uuid

PAGE_RE = re.compile(r'^<!--\s*page\s+(\d+)\s*-->$')
HEADING_RE = re.compile(r'^(#+)\s+(.*)$')
NUMBERED_RE = re.compile(r'^((?:\d+\.)*\d+)\.?\s+(.+)$')
TABLE_RE = re.compile(r'<table[\s>]', re.IGNORECASE)
CAPTION_RE = re.compile(r'^(Table|Figure)\s+\d+\s*\|', re.IGNORECASE)
SENT_RE = re.compile(r'(?<=[.!?])\s+')

NAMESPACE = uuid.UUID('6f9619ff-8b86-d011-b42d-00c04fc964ff')

_tokenizer = None


def token_count(text, model_path):
    """Count tokens with BGE-M3's own tokenizer; fall back to a rough estimate."""
    global _tokenizer
    if _tokenizer is None and model_path:
        try:
            from transformers import AutoTokenizer
            _tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as exc:
            print(f'note: no tokenizer ({exc}); estimating tokens as chars/4',
                  file=sys.stderr)
            _tokenizer = False
    if _tokenizer:
        return len(_tokenizer.encode(text, add_special_tokens=False))
    return max(1, len(text) // 4)


def heading_depth(title):
    """Depth from section numbering, else 1."""
    m = NUMBERED_RE.match(title)
    if m:
        return len(m.group(1).split('.'))
    return 1


def parse_blocks(text):
    """Yield (page, kind, body) blocks. kind in {heading, table, text}."""
    page = 1
    buf = []

    def flush():
        if not buf:
            return None
        body = '\n'.join(buf).strip()
        buf.clear()
        if not body:
            return None
        return ('table' if TABLE_RE.search(body) else 'text'), body

    for line in text.splitlines():
        stripped = line.strip()

        m = PAGE_RE.match(stripped)
        if m:
            out = flush()
            if out:
                yield (page, *out)
            page = int(m.group(1))
            continue

        if not stripped:
            out = flush()
            if out:
                yield (page, *out)
            continue

        m = HEADING_RE.match(stripped)
        if m:
            out = flush()
            if out:
                yield (page, *out)
            yield page, 'heading', m.group(2).strip()
            continue

        buf.append(stripped)

    out = flush()
    if out:
        yield (page, *out)


def looks_like_toc(blocks):
    """A page whose headings nearly all end in a page number is a contents page."""
    headings = [b for _, kind, b in blocks if kind == 'heading']
    if len(headings) < 6:
        return False
    numbered_tail = sum(1 for h in headings if re.search(r'\s\d+$', h))
    return numbered_tail / len(headings) >= 0.6


def split_long(body, target, model_path):
    """Break an oversized text block at sentence boundaries."""
    sentences = SENT_RE.split(body)
    out, cur, cur_tokens = [], [], 0
    for sentence in sentences:
        n = token_count(sentence, model_path)
        if cur and cur_tokens + n > target:
            out.append(' '.join(cur))
            cur, cur_tokens = [], 0
        cur.append(sentence)
        cur_tokens += n
    if cur:
        out.append(' '.join(cur))
    return out


def chunk_document(path, args):
    raw = path.read_text(encoding='utf-8')
    doc_id = path.stem
    all_blocks = list(parse_blocks(raw))

    # Drop contents pages.
    if args.skip_toc:
        by_page = {}
        for block in all_blocks:
            by_page.setdefault(block[0], []).append(block)
        toc_pages = {p for p, blocks in by_page.items() if looks_like_toc(blocks)}
        if toc_pages:
            print(f'{doc_id}: skipping TOC page(s) {sorted(toc_pages)}',
                  file=sys.stderr)
        all_blocks = [b for b in all_blocks if b[0] not in toc_pages]

    heading_stack, chunks = [], []
    pending, pending_tokens, pending_page = [], 0, 1
    last_caption = None

    def flush(force_table=False):
        nonlocal pending, pending_tokens
        if not pending:
            return
        body = '\n\n'.join(pending)
        chunks.append({
            'page': pending_page,
            'heading_path': list(heading_stack),
            'text': body,
            'tokens': pending_tokens,
            'has_table': force_table or bool(TABLE_RE.search(body)),
        })
        pending, pending_tokens = [], 0

    for page, kind, body in all_blocks:
        if kind == 'heading':
            flush()
            depth = heading_depth(body)
            del heading_stack[depth - 1:]
            heading_stack.append(body)
            pending_page = page
            last_caption = None
            continue

        if kind == 'table':
            flush()
            # A bare HTML table is unretrievable; glue the caption on.
            parts = [last_caption, body] if last_caption else [body]
            text = '\n\n'.join(parts)
            chunks.append({
                'page': page,
                'heading_path': list(heading_stack),
                'text': text,
                'tokens': token_count(text, args.model),
                'has_table': True,
            })
            last_caption = None
            continue

        if CAPTION_RE.match(body):
            last_caption = body

        n = token_count(body, args.model)
        if n > args.max_tokens:
            flush()
            for piece in split_long(body, args.target_tokens, args.model):
                pending, pending_tokens, pending_page = [piece], token_count(
                    piece, args.model), page
                flush()
            continue

        if pending and pending_tokens + n > args.target_tokens:
            flush()
        if not pending:
            pending_page = page
        pending.append(body)
        pending_tokens += n

    flush()

    records = []
    for i, chunk in enumerate(chunks):
        prefix = ' > '.join(chunk['heading_path'])
        embed_text = f'{prefix}\n\n{chunk["text"]}' if prefix else chunk['text']
        digest = hashlib.sha256(
            f'{doc_id}:{chunk["text"]}'.encode('utf-8')).hexdigest()
        records.append({
            'id': str(uuid.uuid5(NAMESPACE, digest)),
            'doc_id': doc_id,
            'page': chunk['page'],
            'heading_path': chunk['heading_path'],
            'chunk_index': i,
            'text': chunk['text'],
            'embed_text': embed_text,
            'tokens': chunk['tokens'],
            'has_table': chunk['has_table'],
        })
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paths', nargs='+', help='combined .md files (or a directory)')
    ap.add_argument('--target-tokens', type=int, default=600)
    ap.add_argument('--max-tokens', type=int, default=1200,
                    help='text blocks larger than this get sentence-split; '
                         'tables are never split')
    ap.add_argument('--model', default='/home/ai/models/embeddings/bge-m3',
                    help='tokenizer source for accurate counts')
    ap.add_argument('--no-skip-toc', dest='skip_toc', action='store_false')
    ap.add_argument('--out', help='write JSONL here instead of stdout')
    args = ap.parse_args()

    files = []
    for p in args.paths:
        p = pathlib.Path(p)
        files += sorted(p.glob('*.md')) if p.is_dir() else [p]
    files = [f for f in files if not f.name.endswith('.clean.md')]

    sink = open(args.out, 'w', encoding='utf-8') if args.out else sys.stdout
    total = 0
    for f in files:
        records = chunk_document(f, args)
        for record in records:
            sink.write(json.dumps(record, ensure_ascii=False) + '\n')
        tables = sum(1 for r in records if r['has_table'])
        biggest = max((r['tokens'] for r in records), default=0)
        print(f'{f.name}: {len(records)} chunks ({tables} with tables), '
              f'largest {biggest} tokens', file=sys.stderr)
        total += len(records)
    if args.out:
        sink.close()
    print(f'total {total} chunks', file=sys.stderr)


if __name__ == '__main__':
    main()
