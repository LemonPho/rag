#!/usr/bin/env python3
"""
clean_det.py — turn Unlimited-OCR output into structured markdown.

The model emits blocks tagged with a layout category and bounding box:

    <|det|>header [x1,y1,x2,y2]<|/det|>Introduction

The README's `remove_det` throws the category away and yields flat text.
This keeps it and maps it to markdown structure instead.

Usage:
    python3 clean_det.py ./outputs --stats            # discover categories first
    python3 clean_det.py ./outputs                    # write *.clean.md
    python3 clean_det.py ./outputs --combine          # one .md per document
    python3 clean_det.py ./outputs --keep-bbox        # retain boxes as comments
"""

import argparse
import collections
import pathlib
import re
import sys

DET_RE = re.compile(r'^<\|det\|>([^<\s]+)(?:\s*\[([^\]]*)\])?\s*<\|/det\|>(.*)$')

# Page filename produced by infer.py: <doc>_page_0001.md
PAGE_RE = re.compile(r'^(?P<doc>.+)_page_(?P<page>\d+)$')

# Category -> markdown prefix. Extend from `--stats` output; these are the
# common names, not a verified list for this model.
PREFIX = {
    'title': '# ',
    'doc_title': '# ',
    'header': '## ',
    'section': '## ',
    'section_header': '## ',
    'sec_header': '## ',
    'subheader': '### ',
    'sub_header': '### ',
    'subsection': '### ',
    'subsection_header': '### ',
    'footnote': '> ',
}

# Wrapped in emphasis rather than prefixed.
ITALIC = {'caption', 'figure_caption', 'table_caption', 'image_caption'}

# Passed through byte-for-byte: already-structured content we must not reflow.
VERBATIM = {'table', 'code', 'code_block'}

# Dropped entirely.
SKIP = {'image', 'figure', 'page_number', 'page_header', 'page_footer',
        'header_footer', 'watermark'}

# Wrapped as display math if not already delimited.
MATH = {'formula', 'equation', 'isolate_formula', 'display_formula'}


def parse_blocks(raw):
    """Yield (category, bbox, [lines]) for each detected block.

    Text appearing before any marker is yielded with category None so nothing
    is silently lost.
    """
    category, bbox, lines = None, None, []
    started = False

    for line in raw.splitlines():
        line = line.rstrip()
        if not line:
            continue
        m = DET_RE.match(line)
        if m:
            if started or lines:
                yield category, bbox, lines
            category, bbox = m.group(1).strip(), m.group(2)
            first = m.group(3).strip()
            lines = [first] if first else []
            started = True
            continue
        lines.append(line)

    if started or lines:
        yield category, bbox, lines


def render(category, bbox, lines, keep_bbox=False, unknown=None):
    if category in SKIP:
        return None

    body = '\n'.join(lines).strip()
    if not body:
        return None

    if category in VERBATIM:
        out = body
    elif category in MATH:
        out = body if body.startswith('$') else f'$$\n{body}\n$$'
    elif category in ITALIC:
        out = f'*{body}*'
    elif category in PREFIX:
        # Don't double up if the model already emitted markdown headings.
        out = body if body.lstrip().startswith('#') else PREFIX[category] + body
    else:
        if category is not None and unknown is not None:
            unknown[category] += 1
        out = body

    if keep_bbox and bbox:
        out = f'<!-- {category} [{bbox}] -->\n{out}'
    return out


def convert(raw, keep_bbox=False, unknown=None):
    parts = []
    for category, bbox, lines in parse_blocks(raw):
        rendered = render(category, bbox, lines, keep_bbox, unknown)
        if rendered:
            parts.append(rendered)
    return '\n\n'.join(parts).strip() + '\n'


def gather(paths):
    """Expand directories to their .md files, skipping our own output."""
    files = []
    for p in paths:
        p = pathlib.Path(p)
        if p.is_dir():
            files += [f for f in sorted(p.glob('*.md'))
                      if not f.name.endswith('.clean.md')]
        elif p.is_file():
            files.append(p)
        else:
            print(f'warning: {p} not found', file=sys.stderr)
    return files


def do_stats(files):
    counts = collections.Counter()
    for f in files:
        for category, _, _ in parse_blocks(f.read_text(encoding='utf-8')):
            counts[category if category is not None else '(untagged)'] += 1

    known = set(PREFIX) | ITALIC | VERBATIM | SKIP | MATH
    width = max((len(str(c)) for c in counts), default=10)
    print(f'{"category".ljust(width)}  count  handling')
    for category, n in counts.most_common():
        if category == '(untagged)':
            handling = 'passed through'
        elif category in SKIP:
            handling = 'dropped'
        elif category in VERBATIM:
            handling = 'verbatim'
        elif category in PREFIX:
            handling = f'prefix {PREFIX[category]!r}'
        elif category in ITALIC:
            handling = 'italic'
        elif category in MATH:
            handling = 'display math'
        else:
            handling = '** UNMAPPED — add to PREFIX/SKIP **'
        print(f'{str(category).ljust(width)}  {n:5d}  {handling}')

    unmapped = [c for c in counts if c not in known and c != '(untagged)']
    if unmapped:
        print(f'\n{len(unmapped)} unmapped category(ies): {", ".join(unmapped)}',
              file=sys.stderr)


def do_combine(files, out_dir, keep_bbox, unknown):
    """Merge per-page files into one markdown file per document."""
    docs = collections.defaultdict(list)
    loose = []
    for f in files:
        m = PAGE_RE.match(f.stem)
        if m:
            docs[m.group('doc')].append((int(m.group('page')), f))
        else:
            loose.append(f)

    for doc, pages in sorted(docs.items()):
        chunks = []
        for page_no, f in sorted(pages):
            text = convert(f.read_text(encoding='utf-8'), keep_bbox, unknown)
            chunks.append(f'<!-- page {page_no} -->\n\n{text.strip()}')
        target = out_dir / f'{doc}.md'
        target.write_text('\n\n'.join(chunks) + '\n', encoding='utf-8')
        print(f'{target}  ({len(pages)} pages)')

    for f in loose:
        target = out_dir / f'{f.stem}.clean.md'
        target.write_text(convert(f.read_text(encoding='utf-8'), keep_bbox, unknown),
                          encoding='utf-8')
        print(target)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paths', nargs='+', help='.md files or directories of them')
    ap.add_argument('--stats', action='store_true',
                    help='report category counts and handling; write nothing')
    ap.add_argument('--combine', action='store_true',
                    help='merge <doc>_page_NNNN.md into one <doc>.md with page markers')
    ap.add_argument('--keep-bbox', action='store_true',
                    help='retain bounding boxes as HTML comments')
    ap.add_argument('--out-dir', help='write here instead of alongside the inputs')
    args = ap.parse_args()

    files = gather(args.paths)
    if not files:
        sys.exit('no input files found')

    if args.stats:
        do_stats(files)
        return

    out_dir = pathlib.Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    unknown = collections.Counter()

    if args.combine:
        do_combine(files, out_dir or files[0].parent, args.keep_bbox, unknown)
    else:
        for f in files:
            text = convert(f.read_text(encoding='utf-8'), args.keep_bbox, unknown)
            target = (out_dir / f'{f.stem}.clean.md') if out_dir \
                else f.with_suffix('.clean.md')
            target.write_text(text, encoding='utf-8')
            print(target)

    if unknown:
        print('\nunmapped categories passed through as plain text: '
              + ', '.join(f'{c} ({n})' for c, n in unknown.most_common()),
              file=sys.stderr)


if __name__ == '__main__':
    main()
