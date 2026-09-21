import { useEffect, useRef, useState } from 'react'

const EXAMPLES = [
  'Resume los puntos principales de los documentos.',
  '¿Qué valores aparecen en las tablas?',
  '¿Qué documento menciona la cifra más alta?',
]

/** Parse an SSE byte stream into {event, data} objects. */
async function* sseEvents(response) {
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
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

/** Render [1] markers in the answer as clickable badges. */
function Answer({ text, onCite }) {
  const notFound = text.trim().startsWith('No se encuentra en los documentos')
  const parts = text.split(/(\[\d+\])/g)
  return (
    <div className={'a' + (notFound ? ' notfound' : '')}>
      {parts.map((part, i) => {
        const m = /^\[(\d+)\]$/.exec(part)
        if (!m) return part
        return (
          <button
            key={i}
            className="cite"
            onClick={() => onCite(Number(m[1]))}
            title={`Ver fuente ${m[1]}`}
          >
            {m[1]}
          </button>
        )
      })}
    </div>
  )
}

function Passage({ citation: c }) {
  if (!c) return null
  const path = (c.heading_path || []).join(' › ')
  return (
    <div className="passage">
      <div className="passage-meta">
        <strong>{c.doc_id}</strong>
        {c.page ? <span>página {c.page}</span> : null}
        {path ? <span>{path}</span> : null}
        {c.has_table ? <span>tabla</span> : null}
        <span className="score">rrf {c.score}</span>
      </div>
      {c.has_table ? (
        <div className="passage-body" dangerouslySetInnerHTML={{ __html: c.text }} />
      ) : (
        <div className="passage-body">{c.text}</div>
      )}
    </div>
  )
}

function Sources({ citations, open, setOpen }) {
  if (!citations?.length) return null
  return (
    <div className="sources">
      <div className="chips">
        <span className="label">Fuentes</span>
        {citations.map((c) => (
          <button
            key={c.n}
            className={'chip' + (open === c.n ? ' active' : '')}
            onClick={() => setOpen(open === c.n ? null : c.n)}
            title={(c.heading_path || []).join(' › ')}
          >
            <span className="n">{c.n}</span>
            {c.doc_id}
            <span className="tag">
              {c.page ? `p${c.page}` : ''}
              {c.has_table ? ' ▤' : ''}
            </span>
          </button>
        ))}
      </div>
      {open !== null && <Passage citation={citations.find((c) => c.n === open)} />}
    </div>
  )
}

function Turn({ turn, busy, isLast }) {
  const [open, setOpen] = useState(null)
  return (
    <div className="turn">
      <div className="q">{turn.question}</div>
      {turn.answer ? (
        <Answer text={turn.answer} onCite={setOpen} />
      ) : busy && isLast ? (
        <div className="typing"><i /><i /><i /></div>
      ) : null}
      {turn.error && <div className="error">{turn.error}</div>}
      <Sources citations={turn.citations} open={open} setOpen={setOpen} />
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
    bottom.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [turns, busy])

  async function ask(text) {
    const q = (text ?? question).trim()
    if (!q || busy) return
    setQuestion('')
    setBusy(true)
    const index = turns.length
    setTurns((t) => [...t, { question: q, answer: '', citations: [], error: null }])

    const patch = (fn) => setTurns((t) => t.map((x, i) => (i === index ? fn(x) : x)))

    try {
      const response = await fetch('/api/query/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q }),
      })
      if (!response.ok) throw new Error(`backend returned ${response.status}`)

      for await (const { event, data } of sseEvents(response)) {
        if (event === 'citations') patch((x) => ({ ...x, citations: data }))
        else if (event === 'token') patch((x) => ({ ...x, answer: x.answer + data }))
        else if (event === 'error') patch((x) => ({ ...x, error: data }))
      }
    } catch (err) {
      patch((x) => ({ ...x, error: String(err.message || err) }))
    } finally {
      setBusy(false)
    }
  }

  const totalChunks = docs.reduce((n, d) => n + d.chunks, 0)

  return (
    <div className="app">
      <aside>
        <div>
          <h1 className="brand">
            <span className="glyph">R</span> doc-rag
          </h1>
          <div className="status">
            <span
              className={'dot ' + (health === null ? '' : health.ok ? 'up' : 'down')}
            />
            {health === null
              ? 'comprobando…'
              : health.ok
                ? health.config?.model
                : 'servidor no disponible'}
          </div>
        </div>

        <div>
          <p className="section-label">Documentos · {docs.length}</p>
          <ul className="docs">
            {docs.map((d) => (
              <li key={d.doc_id}>
                <span className="name" title={d.doc_id}>{d.doc_id}</span>
                <span className="count">{d.chunks}</span>
              </li>
            ))}
            {!docs.length && <li className="name">ninguno indexado</li>}
          </ul>
        </div>

        <div className="meta-foot">
          {totalChunks ? `${totalChunks} fragmentos indexados` : null}
          {health?.config ? ` · ctx ${health.config.num_ctx} · top-k ${health.config.top_k}` : null}
        </div>
      </aside>

      <main>
        <div className="thread">
          <div className="column">
            {!turns.length ? (
              <div className="empty">
                <h2>Consulta tus documentos</h2>
                <p>
                  Las respuestas se basan únicamente en el texto indexado y citan los
                  fragmentos de origen. Pulsa una cita para ver la fuente.
                </p>
                <div className="examples">
                  {EXAMPLES.map((e) => (
                    <button key={e} className="example" onClick={() => ask(e)}>
                      {e}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              turns.map((t, i) => (
                <Turn key={i} turn={t} busy={busy} isLast={i === turns.length - 1} />
              ))
            )}
            <div ref={bottom} />
          </div>
        </div>

        <div className="composer">
          <form
            onSubmit={(e) => {
              e.preventDefault()
              ask()
            }}
          >
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Pregunta sobre tus documentos…"
              autoFocus
            />
            <button className="send" type="submit" disabled={busy || !question.trim()}>
              {busy ? 'Pensando…' : 'Preguntar'}
            </button>
          </form>
          <p className="hint">
            Respuestas basadas solo en los documentos: el modelo responde «No se
            encuentra en los documentos» cuando el corpus no cubre la pregunta.
          </p>
        </div>
      </main>
    </div>
  )
}
