import { useEffect, useRef, useState } from 'react'

/** Parse an SSE byte stream into {event, data} objects. */
async function* sseEvents(response) {
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let split
    while ((split = buffer.indexOf('\n\n')) !== -1) {
      const block = buffer.slice(0, split)
      buffer = buffer.slice(split + 2)
      let event = 'message'
      const dataLines = []
      for (const line of block.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim()
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
      }
      if (!dataLines.length) continue
      let data
      try {
        data = JSON.parse(dataLines.join('\n'))
      } catch {
        data = dataLines.join('\n')
      }
      yield { event, data }
    }
  }
}

/** OCR tables are HTML, so they get rendered as markup; prose stays plain text. */
function Passage({ citation: c }) {
  if (!c) return null
  const path = (c.heading_path || []).join(' > ')
  return (
    <div className="passage">
      <div className="passage-meta">
        {c.doc_id}
        {c.page ? ` — page ${c.page}` : ''}
        {path ? ` — ${path}` : ''}
        {` — rrf ${c.score}`}
      </div>
      {c.has_table ? (
        <div className="passage-body" dangerouslySetInnerHTML={{ __html: c.text }} />
      ) : (
        <div className="passage-body">{c.text}</div>
      )}
    </div>
  )
}

function Sources({ citations }) {
  const [open, setOpen] = useState(null)
  if (!citations?.length) return null
  return (
    <div className="sources">
      <div className="chips">
        {citations.map((c) => (
          <button
            key={c.n}
            className={'chip' + (open === c.n ? ' active' : '')}
            onClick={() => setOpen(open === c.n ? null : c.n)}
            title={(c.heading_path || []).join(' > ')}
          >
            [{c.n}] {c.doc_id}
            {c.page ? ` p${c.page}` : ''}
            {c.has_table ? ' ▤' : ''}
          </button>
        ))}
      </div>
      {open !== null && <Passage citation={citations.find((c) => c.n === open)} />}
    </div>
  )
}

export default function App() {
  const [question, setQuestion] = useState('')
  const [turns, setTurns] = useState([])
  const [busy, setBusy] = useState(false)
  const [docs, setDocs] = useState([])
  const [health, setHealth] = useState(null)
  const bottom = useRef(null)

  useEffect(() => {
    fetch('/api/documents')
      .then((r) => r.json())
      .then((d) => setDocs(d.documents || []))
      .catch(() => {})
    fetch('/api/health')
      .then((r) => r.json().then((d) => ({ ok: r.ok, ...d })))
      .then(setHealth)
      .catch(() => setHealth({ ok: false }))
  }, [])

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth' })
  }, [turns, busy])

  async function ask(e) {
    e?.preventDefault()
    const q = question.trim()
    if (!q || busy) return
    setQuestion('')
    setBusy(true)
    const index = turns.length
    setTurns((t) => [...t, { question: q, answer: '', citations: [], error: null }])

    try {
      const response = await fetch('/api/query/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q }),
      })
      if (!response.ok) throw new Error(`backend returned ${response.status}`)

      for await (const { event, data } of sseEvents(response)) {
        if (event === 'citations') {
          setTurns((t) => t.map((x, i) => (i === index ? { ...x, citations: data } : x)))
        } else if (event === 'token') {
          setTurns((t) =>
            t.map((x, i) => (i === index ? { ...x, answer: x.answer + data } : x)),
          )
        } else if (event === 'error') {
          setTurns((t) => t.map((x, i) => (i === index ? { ...x, error: data } : x)))
        }
      }
    } catch (err) {
      setTurns((t) =>
        t.map((x, i) => (i === index ? { ...x, error: String(err.message || err) } : x)),
      )
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="app">
      <aside>
        <h1>doc-rag</h1>
        <div className={'status ' + (health?.ok ? 'up' : 'down')}>
          {health === null
            ? 'checking…'
            : health.ok
              ? `${health.collection?.points ?? '?'} chunks · ${health.config?.model}`
              : 'backend unavailable'}
        </div>
        <h2>Documents ({docs.length})</h2>
        <ul className="docs">
          {docs.map((d) => (
            <li key={d.doc_id}>
              <span>{d.doc_id}</span>
              <em>{d.chunks}</em>
            </li>
          ))}
          {!docs.length && <li className="muted">none indexed</li>}
        </ul>
      </aside>

      <main>
        <div className="thread">
          {!turns.length && (
            <p className="muted intro">
              Ask a question about the indexed documents. Answers cite the passages
              they used — click a citation to see the source text.
            </p>
          )}
          {turns.map((t, i) => (
            <div className="turn" key={i}>
              <div className="q">{t.question}</div>
              <div className="a">
                {t.answer || (busy && i === turns.length - 1 ? <em>thinking…</em> : null)}
                {t.error && <div className="error">{t.error}</div>}
              </div>
              <Sources citations={t.citations} />
            </div>
          ))}
          <div ref={bottom} />
        </div>

        <form onSubmit={ask}>
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask about your documents…"
            autoFocus
          />
          <button type="submit" disabled={busy || !question.trim()}>
            {busy ? '…' : 'Ask'}
          </button>
        </form>
      </main>
    </div>
  )
}
