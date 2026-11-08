// "How the model works" (/scan/model): the model behind /scan, its training, its int8 quantization, the
// fp32 / int8 comparison and the phone measures. Every number comes from model_report.json, written by
// food-recognition/scripts/export_model_report.py from the committed results: nothing here is typed by hand.
import { useEffect, useState, type ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { Link } from 'react-router-dom'
import { BarList, DumbbellChart, LineChart, SERIES, type Series } from './modelReportCharts'
import './ModelReport.css'

interface Epoch { epoch: number; train_loss: number; val_loss: number; val_map: number; select: number; sec: number }
interface TrainedModel {
  run: string; name: string; backbone: string; resolution: number; lr: number; augment: string
  best_epoch: number; best_val_map: number; epochs_run: number; sec_per_epoch: number; history: Epoch[]
  test: { map: number; f1_micro: number; map_frequent: number; map_rare: number } | null
}
interface QuantVariant {
  run: string; variant: string; method: string; calib_size: number; per_channel: boolean; sampling: string
  op_types: string; reduce_range: boolean; size_mb: number; size_fp32_mb: number
  loss_val: number; loss_val_lo: number; loss_val_hi: number; chosen: boolean
}
interface Comparison {
  run: string; name: string; variant: string; deployed: boolean
  fp32: { map: number; f1_micro: number; size_mb: number }; int8: { map: number; f1_micro: number; size_mb: number }
  loss: number; loss_lo: number; loss_hi: number
}
interface PhoneRow { variant: string; backend: string; threads: number; run_ms: number; pre_ms: number; total_ms: number; measures: number }
interface Report {
  generated: string
  dataset: { name: string; classes: number; train: number; val: number; test: number }
  targets: { latency_ms: number; size_mb: number; min_map: number; max_map_drop: number }
  deployed: { run: string; variant: string }
  screening: { model: string; val_map: number; seeds: number; params_m: number; mult_adds_g: number }[]
  models: TrainedModel[]
  quantization: QuantVariant[]
  comparison: Comparison[]
  phone: { device: string; rows: PhoneRow[] } | null
}

const f3 = (v: number) => v.toFixed(3)
const pts = (v: number) => (Math.abs(v) < 0.0005 ? 0 : v * 100).toFixed(1) // no "-0.0"
const ms = (v: number) => Math.round(v).toString()
const NAMES: Record<string, string> = { a_convnext: 'ConvNeXt', a_mnv3: 'MobileNetV3', a_lite0: 'Lite0' }
const BACKBONES: Record<string, string> = {
  convnext_atto: 'ConvNeXt-Atto', mobilenetv3_large_100: 'MobileNetV3-Large', efficientnet_b0: 'EfficientNet-B0',
  mobilenetv3_small_100: 'MobileNetV3-Small', mobilevit_xs: 'MobileViT-XS', tf_efficientnet_lite0: 'EfficientNet-Lite0',
}
// Glossary order: the metrics first, then training, then quantization, then the phone and the model names.
const GLOSSARY = ['map', 'ap', 'select', 'test', 'pt', 'loss', 'epoch', 'es', 'lr', 'sec', 'fp', 'ptq', 'calib', 'method',
  'pc', 'cov', 'rr', 'conv', 'ci', 'thread', 'convnext', 'mnv3', 'lite0']

function variantLabel(q: QuantVariant) {
  return [
    NAMES[q.run] ?? q.run,
    `${q.method} ${q.calib_size}`,
    q.per_channel ? null : 'per-tensor', // per-channel is the default: only the exception is named
    q.sampling === 'coverage' ? 'coverage' : null,
    q.op_types !== 'all' ? `only ${q.op_types}` : null,
    q.reduce_range ? 'reduce_range' : null,
  ].filter(Boolean).join(' · ')
}

/** A card that opens on tap: the page stays short and the speaker opens the part being discussed. */
function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <details className="card mr-sec">
      <summary><h2>{title}</h2></summary>
      <div className="stack">{children}</div>
    </details>
  )
}

export default function ModelReport() {
  const { t } = useTranslation()
  const [report, setReport] = useState<Report | null>(null)
  const [error, setError] = useState('')
  const [lossRun, setLossRun] = useState('a_convnext')

  useEffect(() => {
    fetch('/models/model_report.json')
      .then((r) => {
        if (!r.ok || (r.headers.get('content-type') ?? '').includes('text/html')) throw new Error('model_report.json')
        return r.json()
      })
      .then(setReport)
      .catch((e) => setError(String(e)))
  }, [])

  if (error) return <div className="stack sheet"><div className="error">{t('modelPage.loadError')} ({error})</div></div>
  if (!report) return <div className="stack sheet"><p className="muted">{t('modelPage.loading')}</p></div>

  const deployed = report.models.find((m) => m.run === report.deployed.run)!
  const cmp = report.comparison.find((c) => c.deployed)!
  const lossModel = report.models.find((m) => m.run === lossRun) ?? deployed
  const mapSeries: Series[] = report.models.map((m, i) => ({
    id: m.run, label: m.name, color: SERIES[i], mark: m.best_epoch,
    points: m.history.map((h) => ({ x: h.epoch, y: h.val_map })),
  }))
  const lossSeries: Series[] = [
    { id: 'train', label: t('modelPage.trainLoss'), color: SERIES[0], dashed: true, points: lossModel.history.map((h) => ({ x: h.epoch, y: h.train_loss })) },
    { id: 'val', label: t('modelPage.valLoss'), color: SERIES[1], mark: lossModel.best_epoch, points: lossModel.history.map((h) => ({ x: h.epoch, y: h.val_loss })) },
  ]
  const finalists = new Set(report.models.map((m) => m.backbone))
  const chosenQuant = report.quantization.find((q) => q.chosen && q.run === cmp.run)
  const quant = [...report.quantization].sort((a, b) => b.loss_val - a.loss_val)
  const phoneName = (v: string) => {
    const [run, variant] = v.split('__')
    return `${NAMES[run] ?? run} ${variant === 'model_fp32' ? 'fp32' : 'int8'}`
  }
  const phoneRows = report.phone
    ? [...new Set(report.phone.rows.map((r) => r.variant))].map((v) => {
        const at = (b: string, th: number) => report.phone!.rows.find((r) => r.variant === v && r.backend === b && r.threads === th)
        return { v, w1: at('wasm', 1), w4: at('wasm', 4), gpu: at('webgpu', 1) }
      })
    : []
  const deployedPhone = phoneRows.find((r) => r.v === `${cmp.run}__${cmp.variant}`)

  return (
    <div className="stack sheet mr">
      <Link to="/scan" className="scan__back">← {t('scan.title')}</Link>
      <h1>{t('modelPage.title')}</h1>
      <p className="muted">{t('modelPage.lead')}</p>

      {/* 1. The model in one look */}
      <Section title={t('modelPage.s1')}>
        <ol className="mr-steps">
          <li>{t('modelPage.step1')}</li>
          <li>{t('modelPage.step2', { res: deployed.resolution })}</li>
          <li>{t('modelPage.step3', { name: deployed.name, size: cmp.int8.size_mb })}</li>
          <li>{t('modelPage.step4', { n: report.dataset.classes })}</li>
          <li>{t('modelPage.step5')}</li>
        </ol>
        <div className="mr-tiles">
          <div><span className="muted">{t('modelPage.tileMap')}</span><strong>{f3(cmp.int8.map)}</strong></div>
          <div><span className="muted">{t('modelPage.tileSize')}</span><strong>{cmp.int8.size_mb} MB</strong></div>
          <div><span className="muted">{t('modelPage.tileLatency')}</span><strong>{deployedPhone?.w4 ? `${ms(deployedPhone.w4.run_ms)} ms` : '—'}</strong></div>
        </div>
        <p className="muted">{t('modelPage.dataset', { ...report.dataset })}</p>
      </Section>

      {/* 2. Training */}
      <Section title={t('modelPage.s2')}>
        <p className="muted">{t('modelPage.s2Lead')}</p>
        <LineChart title={t('modelPage.mapChart')} series={mapSeries} yLabel={t('modelPage.epoch')} format={f3} />
        <div className="mr-switch" role="group" aria-label={t('modelPage.lossChart')}>
          {report.models.map((m) => (
            <button key={m.run} type="button" className={m.run === lossRun ? '' : 'secondary'} onClick={() => setLossRun(m.run)}>{m.name}</button>
          ))}
        </div>
        <LineChart title={`${t('modelPage.lossChart')} — ${lossModel.name}`} series={lossSeries} yLabel={t('modelPage.epoch')} format={(v) => v.toFixed(3)} />
        <p className="muted">{t('modelPage.overfit')}</p>
        <div className="mr-table">
          <table>
            <thead><tr><th>{t('modelPage.model')}</th><th>{t('modelPage.bestEpoch')}</th><th>{t('modelPage.valMap')}</th><th>{t('modelPage.testMap')}</th><th>{t('modelPage.secEpoch')}</th><th>LR</th></tr></thead>
            <tbody>
              {report.models.map((m) => (
                <tr key={m.run}>
                  <td>{m.name}</td><td>{m.best_epoch} / {m.epochs_run}</td><td>{f3(m.best_val_map)}</td>
                  <td>{m.test ? f3(m.test.map) : '—'}</td><td>{m.sec_per_epoch}</td><td>{m.lr}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <h3 className="mr-h3">{t('modelPage.screenTitle')}</h3>
        <p className="muted">{t('modelPage.screenLead')}</p>
        <div className="mr-table">
          <table>
            <thead><tr><th>{t('modelPage.model')}</th><th>{t('modelPage.valMap')}</th><th>{t('modelPage.params')}</th><th>{t('modelPage.gmacs')}</th></tr></thead>
            <tbody>
              {report.screening.map((s) => (
                <tr key={s.model}>
                  <td>{BACKBONES[s.model] ?? s.model}{finalists.has(s.model) ? ' ✓' : ''}</td>
                  <td>{f3(s.val_map)}</td><td>{s.params_m}</td><td>{s.mult_adds_g}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      {/* 3. Quantization: what was done */}
      <Section title={t('modelPage.s3')}>
        <p className="muted">{t('modelPage.s3Lead')}</p>
        <BarList title={t('modelPage.quantChart')} max={Math.max(...quant.map((q) => q.loss_val)) * 1.15}
          refValue={report.targets.max_map_drop} refLabel={t('modelPage.target', { v: pts(report.targets.max_map_drop) })}
          format={(v) => `${pts(v)} pt`}
          rows={quant.map((q) => ({ id: `${q.run}${q.variant}`, label: variantLabel(q), value: q.loss_val, note: q.chosen ? `✓ ${t('modelPage.chosen')}` : '' }))} />
        {chosenQuant && (
          <p className="muted">{t('modelPage.quantNote', { t: pts(cmp.loss) })}</p>
        )}
        <h3 className="mr-h3">{t('modelPage.fixesTitle')}</h3>
        <ul className="mr-fixes">
          <li>{t('modelPage.fix1')}</li>
          <li>{t('modelPage.fix2')}</li>
          <li>{t('modelPage.fix3')}</li>
        </ul>
      </Section>

      {/* 4. fp32 against int8 */}
      <Section title={t('modelPage.s4')}>
        <DumbbellChart title={t('modelPage.cmpChart')} labels={['fp32', 'int8']} domain={[0.3, 0.4]} format={f3}
          rows={report.comparison.map((c) => ({ id: c.run, label: c.name, a: c.fp32.map, b: c.int8.map }))} />
        <div className="mr-table">
          <table>
            <thead><tr><th>{t('modelPage.model')}</th><th>mAP fp32</th><th>mAP int8</th><th>{t('modelPage.loss')}</th><th>{t('modelPage.size')}</th></tr></thead>
            <tbody>
              {report.comparison.map((c) => (
                <tr key={c.run}>
                  <td>{c.name}{c.deployed ? ` ✓` : ''}</td><td>{f3(c.fp32.map)}</td><td>{f3(c.int8.map)}</td>
                  <td>{pts(c.loss)} [{pts(c.loss_lo)} ; {pts(c.loss_hi)}]</td>
                  <td>{c.fp32.size_mb} → {c.int8.size_mb} MB</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted">{t('modelPage.why', { test: report.dataset.test.toLocaleString() })}</p>
      </Section>

      {/* 5. On the phone */}
      {report.phone && (
        <Section title={t('modelPage.s5')}>
          <p className="muted">{t('modelPage.s5Lead', { device: report.phone.device })}</p>
          <div className="mr-table">
            <table>
              <thead><tr><th>{t('modelPage.variant')}</th><th>CPU 1</th><th>CPU 4</th><th>GPU</th><th>{t('modelPage.total')}</th></tr></thead>
              <tbody>
                {phoneRows.map((r) => (
                  <tr key={r.v}>
                    <td>{phoneName(r.v)}{r.v === `${cmp.run}__${cmp.variant}` ? ' ✓' : ''}</td>
                    <td>{r.w1 ? ms(r.w1.run_ms) : '—'}</td><td>{r.w4 ? ms(r.w4.run_ms) : '—'}</td>
                    <td>{r.gpu ? ms(r.gpu.run_ms) : '—'}</td><td>{r.w4 ? ms(r.w4.total_ms) : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}

      {/* 6. What is deployed, and why */}
      <Section title={t('modelPage.s6')}>
        <p>{t('modelPage.s6Body', { name: cmp.name, map: f3(cmp.int8.map), size: cmp.int8.size_mb })}</p>
        <p className="muted">{t('modelPage.limits', { target: report.targets.min_map })}</p>
      </Section>

      {/* 7. Every acronym on the page */}
      <Section title={t('modelPage.glTitle')}>
        <dl className="mr-glossary">
          {GLOSSARY.map((k) => (
            <div key={k}><dt>{t(`modelPage.gl.${k}.t`)}</dt><dd>{t(`modelPage.gl.${k}.d`)}</dd></div>
          ))}
        </dl>
      </Section>

    </div>
  )
}
