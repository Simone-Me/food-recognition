// Analyse your plate: take or pick a photo -> the ingredients the model sees (+ top-3 dishes in part D).
// "Evaluate" lets the user correct the list; "Send" stores the corrections (ids only, never the photo, 0111).
// Source of truth: food-recognition/web/src/pages/FoodScan.tsx. Inference runs entirely on the phone.
import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link } from 'react-router-dom'
import { WipChip } from '../components/WipChip'
import { useAuth } from '../lib/auth'
import { loadFoodModel, predictFile, type ModelMeta, type Prediction } from '../lib/foodModel'
import { submitScanFeedback, type ScanScores } from '../lib/rpc'
import './FoodScan.css'

const MODEL_URL = '/models/a_convnext__model_int8_minmax_200_pc_only-conv_rr.onnx' // the variant chosen in step A13
const META_URL = '/models/a_convnext__meta.json'
// What the feedback records as the model: the file name, unique per variant (meta.version is shared by fp32 and int8).
const MODEL_ID = MODEL_URL.split('/').pop()!.replace(/.onnx$/, '')

// Where the analysis is. 'download' only happens on first use (afterwards the model is cached).
type Stage = { kind: 'idle' } | { kind: 'download'; fraction: number | null } | { kind: 'analyse' } | { kind: 'error'; message: string }

// view: the model's answer, read-only. edit: the user corrects it. sent: the scores are shown.
type Mode = 'view' | 'edit' | 'sent'

interface Named { id: number; name: string }

/** Same formulas as submit_scan_feedback (0111): shown to a signed-out user, who cannot send. */
function localScores(proposed: number, removed: number, added: number): ScanScores {
  const correct = proposed - removed
  return {
    precision: proposed > 0 ? correct / proposed : null,
    recall: correct + added > 0 ? correct / (correct + added) : null,
    jaccard: proposed + added > 0 ? correct / (proposed + added) : null,
  }
}

const pct = (x: number | null) => (x === null ? '—' : String(Math.round(x * 100)))

export default function FoodScan() {
  const { t } = useTranslation()
  const { session } = useAuth()
  const [photoUrl, setPhotoUrl] = useState<string | null>(null)
  const [pred, setPred] = useState<Prediction | null>(null)
  const [meta, setMeta] = useState<ModelMeta | null>(null)
  const [stage, setStage] = useState<Stage>({ kind: 'idle' })
  const [mode, setMode] = useState<Mode>('view')
  const [removed, setRemoved] = useState<Set<number>>(new Set())
  const [added, setAdded] = useState<Named[]>([])
  const [query, setQuery] = useState('')
  const [sending, setSending] = useState(false)
  const [sendError, setSendError] = useState('')
  const [scores, setScores] = useState<{ value: ScanScores; saved: boolean } | null>(null)
  const busy = stage.kind === 'download' || stage.kind === 'analyse'

  // The preview is an object URL: release it when replaced or when leaving the page.
  useEffect(() => () => { if (photoUrl) URL.revokeObjectURL(photoUrl) }, [photoUrl])

  async function onFile(file: File | undefined) {
    if (!file || busy) return
    setPhotoUrl(URL.createObjectURL(file))
    setPred(null)
    setMode('view')
    setRemoved(new Set())
    setAdded([])
    setQuery('')
    setScores(null)
    setSendError('')
    setStage({ kind: 'analyse' })
    try {
      const p = await predictFile(file, MODEL_URL, META_URL, (fraction) =>
        setStage(fraction !== null && fraction >= 1 ? { kind: 'analyse' } : { kind: 'download', fraction }))
      setMeta((await loadFoodModel(MODEL_URL, META_URL)).meta) // already loaded: cached
      setPred(p)
      setStage({ kind: 'idle' })
    } catch (e) {
      console.error(e)
      setStage({ kind: 'error', message: e instanceof Error ? e.message : String(e) })
    }
  }

  // Every ingredient the model knows (English names for now), minus those already on the list.
  const suggestions = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!meta || q.length === 0) return []
    const taken = new Set([...(pred?.ingredients ?? []).map((i) => i.id), ...added.map((i) => i.id)])
    return meta.ingredient_ids
      .map((id, k) => ({ id, name: meta.ingredient_names[k] }))
      .filter((i) => !taken.has(i.id) && i.name.toLowerCase().includes(q))
      .sort((a, b) => Number(!a.name.toLowerCase().startsWith(q)) - Number(!b.name.toLowerCase().startsWith(q)))
      .slice(0, 8)
  }, [query, meta, pred, added])

  function toggleRemoved(id: number) {
    const next = new Set(removed)
    if (next.has(id)) next.delete(id)
    else next.add(id)
    setRemoved(next)
  }

  async function send() {
    if (!pred || !meta) return
    const proposed = pred.ingredients
    const fallback = localScores(proposed.length, removed.size, added.length)
    if (!session) {
      setScores({ value: fallback, saved: false })
      setMode('sent')
      return
    }
    setSending(true)
    setSendError('')
    try {
      const value = await submitScanFeedback({
        modelVersion: MODEL_ID,
        proposed: proposed.map((i) => i.id),
        proposedProbs: proposed.map((i) => i.prob),
        removed: [...removed],
        added: added.map((i) => i.id),
      })
      setScores({ value, saved: true })
      setMode('sent')
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e)
      // 30 evaluations an hour per person (0111): say it in words, not with the server's code.
      setSendError(message.includes('RATE_LIMIT') ? t('scan.rateLimit') : t('scan.sendError', { message }))
    } finally {
      setSending(false)
    }
  }

  const proposed = pred?.ingredients ?? []

  return (
    <div className="stack sheet scan">
      <Link to="/profile" className="scan__back">← {t('scan.toProfile')}</Link>
      <h1>
        {t('scan.title')} <WipChip />
      </h1>
      <p className="muted">{t('scan.lead')} <Link to="/scan/model">{t('scan.howItWorks')}</Link></p>

      <div className="scan__actions">
        {/* Two inputs, not one: `capture` opens the camera directly, without it the phone offers the gallery. */}
        <label className={`scan__pick${busy ? ' is-disabled' : ''}`}>
          {t('scan.takePhoto')}
          <input type="file" accept="image/*" capture="environment" disabled={busy}
            onChange={(e) => { onFile(e.target.files?.[0]); e.target.value = '' }} />
        </label>
        <label className={`scan__pick${busy ? ' is-disabled' : ''}`}>
          {t('scan.gallery')}
          <input type="file" accept="image/*" disabled={busy}
            onChange={(e) => { onFile(e.target.files?.[0]); e.target.value = '' }} />
        </label>
      </div>

      {photoUrl && <img className="scan__photo" src={photoUrl} alt={t('scan.photoAlt')} />}

      {busy && (
        <div className="scan__progress" role="status" aria-live="polite">
          {stage.kind === 'download' && stage.fraction !== null
            ? <progress max={1} value={stage.fraction} />
            : <progress />}
          <span className="muted">
            {stage.kind === 'download'
              ? t('scan.downloading', { pct: stage.fraction === null ? '…' : Math.round(stage.fraction * 100) })
              : t('scan.analysing')}
          </span>
        </div>
      )}

      {stage.kind === 'error' && <div className="error">{t('scan.error')} <small>({stage.message})</small></div>}

      {pred?.unknown && <p className="card">{t('scan.unknown')}</p>}
      {pred && pred.dishes.length > 0 && (
        <ol>{pred.dishes.map((d) => <li key={d.name}>{d.name}</li>)}</ol>
      )}

      {/* The model's answer, read-only: correcting it is a separate, deliberate step. */}
      {pred && mode === 'view' && (
        <>
          {proposed.length > 0 && (
            <div className="scan__chips">
              {proposed.map((c) => <span key={c.id} className="scan__chip">{c.name}</span>)}
            </div>
          )}
          <button type="button" className="secondary" onClick={() => setMode('edit')}>{t('scan.evaluate')}</button>
        </>
      )}

      {pred && mode === 'edit' && (
        <div className="card stack">
          <p className="muted">{t('scan.evaluateHint')}</p>
          {proposed.length > 0 && (
            <div className="scan__chips">
              {proposed.map((c) => (
                <button key={c.id} type="button" aria-pressed={removed.has(c.id)}
                  className={`scan__chip scan__chip--toggle${removed.has(c.id) ? ' is-removed' : ''}`}
                  onClick={() => toggleRemoved(c.id)}>
                  {c.name} <span aria-hidden="true">{removed.has(c.id) ? '↺' : '✕'}</span>
                </button>
              ))}
            </div>
          )}
          {added.length > 0 && (
            <div className="scan__chips">
              {added.map((c) => (
                <button key={c.id} type="button" className="scan__chip scan__chip--added"
                  onClick={() => setAdded(added.filter((x) => x.id !== c.id))}>
                  + {c.name} <span aria-hidden="true">✕</span>
                </button>
              ))}
            </div>
          )}
          <input type="search" value={query} placeholder={t('scan.addPlaceholder')}
            aria-label={t('scan.addPlaceholder')} onChange={(e) => setQuery(e.target.value)} />
          {suggestions.length > 0 && (
            <div className="scan__suggestions">
              {suggestions.map((s) => (
                <button key={s.id} type="button" className="secondary"
                  onClick={() => { setAdded([...added, s]); setQuery('') }}>
                  {s.name}
                </button>
              ))}
            </div>
          )}
          {sendError && <div className="error">{sendError}</div>}
          <div className="scan__actions">
            <button type="button" className="secondary" disabled={sending}
              onClick={() => { setMode('view'); setRemoved(new Set()); setAdded([]); setQuery('') }}>
              {t('scan.cancel')}
            </button>
            <button type="button" disabled={sending} onClick={send}>
              {sending ? t('scan.sending') : t('scan.send')}
            </button>
          </div>
        </div>
      )}

      {mode === 'sent' && scores && (
        <div className="card stack" role="status">
          <strong>{t('scan.score', { pct: pct(scores.value.jaccard) })}</strong>
          <span className="muted">
            {t('scan.scoreDetail', { precision: pct(scores.value.precision), recall: pct(scores.value.recall) })}
          </span>
          {scores.saved
            ? <span className="muted">{t('scan.sent')}</span>
            : <span className="muted">{t('scan.signInToSend')} <Link to="/signin">{t('scan.signIn')}</Link></span>}
        </div>
      )}

      {pred && (
        <small className="muted">
          {t('scan.timings', { pre: pred.timings.preMs.toFixed(0), run: pred.timings.runMs.toFixed(0) })}
        </small>
      )}
    </div>
  )
}
