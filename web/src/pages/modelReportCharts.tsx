// Small inline-SVG charts for the model page (/scan/model). No chart library: three forms only.
// Specs: 2px lines, >= 8px markers with a surface ring, 1px recessive grid, text in text tokens (never the
// series colour), a legend for >= 2 series plus direct labels, a hover/touch tooltip on every chart.
import { useLayoutEffect, useRef, useState, type PointerEvent, type ReactNode } from 'react'

// Categorical slots 1-3 of the reference palette, validated on the app's paper (#fff9ee): blue, orange, aqua.
// Aqua is below 3:1 against the paper: every chart also gets direct labels and a table, never colour alone.
export const SERIES = ['#2a78d6', '#eb6834', '#1baf7a']

export interface Series {
  id: string
  label: string
  color: string
  dashed?: boolean
  points: { x: number; y: number | null }[]
  mark?: number // x of a point to ring (e.g. the best epoch)
}

const H = 240

/** The plot's real width in CSS pixels: the SVG is drawn at that size instead of being scaled down. */
function useWidth() {
  const ref = useRef<HTMLDivElement>(null)
  const [w, setW] = useState(600)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setW(Math.max(280, Math.round(e.contentRect.width))))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return [ref, w] as const
}

function niceTicks(min: number, max: number, n = 4) {
  const span = max - min || 1
  const step0 = span / n
  const mag = 10 ** Math.floor(Math.log10(step0))
  const step = [1, 2, 2.5, 5, 10].map((k) => k * mag).find((s) => span / s <= n) ?? mag * 10
  const lo = Math.floor(min / step) * step
  const ticks: number[] = []
  for (let v = lo; ; v += step) {
    ticks.push(Number(v.toFixed(10)))
    if (v >= max - step * 0.001) break // the last tick is at or above the largest value: nothing is drawn off the plot
  }
  return ticks
}

export function Legend({ items }: { items: { label: string; color: string; dashed?: boolean }[] }) {
  return (
    <div className="mr-legend">
      {items.map((i) => (
        <span key={i.label} className="mr-legend__item">
          <svg width="18" height="10" aria-hidden="true">
            <line x1="1" y1="5" x2="17" y2="5" stroke={i.color} strokeWidth="2" strokeLinecap="round"
              strokeDasharray={i.dashed ? '4 3' : undefined} />
          </svg>
          {i.label}
        </span>
      ))}
    </div>
  )
}

/** Lines over epochs, with a crosshair tooltip and labels at the line ends. */
export function LineChart({ series, yLabel, format, title }: {
  series: Series[]; yLabel: string; format: (v: number) => string; title: string
}) {
  const [hoverX, setHoverX] = useState<number | null>(null)
  const [ref, W] = useWidth()
  const endLabels = W >= 480
  const M = { l: 44, r: endLabels ? 132 : 12, t: 14, b: 34 }
  const xs = series.flatMap((s) => s.points.map((p) => p.x))
  const ys = series.flatMap((s) => s.points.map((p) => p.y).filter((y): y is number => y !== null))
  const xMax = Math.max(...xs, 1)
  const ticksY = niceTicks(Math.min(...ys), Math.max(...ys))
  const y0 = ticksY[0]
  const y1 = ticksY[ticksY.length - 1]
  const sx = (x: number) => M.l + (x / xMax) * (W - M.l - M.r)
  const sy = (y: number) => H - M.b - ((y - y0) / (y1 - y0 || 1)) * (H - M.t - M.b)
  const ticksX = niceTicks(0, xMax, W < 480 ? 4 : 6).filter((t) => t <= xMax)

  // End labels: nudged apart by at least 14px; a leader line keeps each label attached to its line.
  const ends = series
    .map((s) => {
      const last = [...s.points].reverse().find((p) => p.y !== null)
      return last ? { s, x: sx(last.x), y: sy(last.y as number), ly: sy(last.y as number) } : null
    })
    .filter((e): e is NonNullable<typeof e> => e !== null)
    .sort((a, b) => a.y - b.y)
  for (let i = 1; i < ends.length; i++) ends[i].ly = Math.max(ends[i].ly, ends[i - 1].ly + 14)

  function onMove(e: PointerEvent<SVGRectElement>) {
    const box = e.currentTarget.ownerSVGElement!.getBoundingClientRect()
    const px = ((e.clientX - box.left) / box.width) * W
    const x = Math.round(((px - M.l) / (W - M.l - M.r)) * xMax)
    setHoverX(Math.max(0, Math.min(xMax, x)))
  }

  const tip = hoverX === null ? null : series.map((s) => ({ s, p: s.points.find((p) => p.x === hoverX) }))

  return (
    <figure className="mr-chart">
      <figcaption className="mr-chart__title">{title}</figcaption>
      <Legend items={series.map((s) => ({ label: s.label, color: s.color, dashed: s.dashed }))} />
      <div className="mr-chart__plot" ref={ref}>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title}>
          {ticksY.map((t) => (
            <g key={t}>
              <line x1={M.l} x2={W - M.r} y1={sy(t)} y2={sy(t)} className="mr-grid" />
              <text x={M.l - 6} y={sy(t)} className="mr-tick" textAnchor="end" dominantBaseline="middle">{format(t)}</text>
            </g>
          ))}
          {ticksX.map((t) => (
            <text key={t} x={sx(t)} y={H - M.b + 16} className="mr-tick" textAnchor="middle">{t}</text>
          ))}
          <text x={(M.l + W - M.r) / 2} y={H - 4} className="mr-axis" textAnchor="middle">{yLabel}</text>
          {series.map((s) => (
            <polyline key={s.id} fill="none" stroke={s.color} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round"
              strokeDasharray={s.dashed ? '5 4' : undefined}
              points={s.points.filter((p) => p.y !== null).map((p) => `${sx(p.x)},${sy(p.y as number)}`).join(' ')} />
          ))}
          {series.map((s) => {
            const p = s.mark === undefined ? undefined : s.points.find((q) => q.x === s.mark)
            return p && p.y !== null
              ? <circle key={s.id} cx={sx(p.x)} cy={sy(p.y)} r="5" fill={s.color} className="mr-ring" />
              : null
          })}
          {endLabels && ends.map((e) => (
            <g key={e.s.id}>
              {Math.abs(e.ly - e.y) > 2 && <line x1={e.x + 3} y1={e.y} x2={W - M.r + 6} y2={e.ly} className="mr-leader" />}
              <text x={W - M.r + 8} y={e.ly} className="mr-label" dominantBaseline="middle">{e.s.label}</text>
            </g>
          ))}
          {hoverX !== null && <line x1={sx(hoverX)} x2={sx(hoverX)} y1={M.t} y2={H - M.b} className="mr-crosshair" />}
          {tip?.map(({ s, p }) => p && p.y !== null
            ? <circle key={s.id} cx={sx(p.x)} cy={sy(p.y)} r="4" fill={s.color} className="mr-ring" />
            : null)}
          <rect x={M.l} y={M.t} width={W - M.l - M.r} height={H - M.t - M.b} fill="transparent"
            onPointerMove={onMove} onPointerDown={onMove} onPointerLeave={() => setHoverX(null)} />
        </svg>
        {tip && hoverX !== null && (
          // Below the plot, so it never hides the lines; shifted by its own width so it stays inside the card.
          <div className="mr-tip" style={{ left: `${(sx(hoverX) / W) * 100}%`, transform: `translateX(-${(sx(hoverX) / W) * 100}%)` }}>
            <strong>{yLabel} {hoverX}</strong>
            {tip.map(({ s, p }) => (
              <span key={s.id}><i style={{ background: s.color }} />{s.label}: {p && p.y !== null ? format(p.y) : '—'}</span>
            ))}
          </div>
        )}
      </div>
    </figure>
  )
}

/** Horizontal bars, one series, value at the tip, an optional reference line (e.g. the 1.5-point target). */
export function BarList({ rows, max, refValue, refLabel, format, title }: {
  rows: { id: string; label: ReactNode; value: number; note?: string }[]
  max: number; refValue?: number; refLabel?: string; format: (v: number) => string; title: string
}) {
  const [hover, setHover] = useState<string | null>(null)
  const [ref, W] = useWidth()
  const rowH = 46
  const L = 0
  const R = 70
  const h = rows.length * rowH + 22
  const sx = (v: number) => L + (Math.max(0, v) / max) * (W - L - R)
  return (
    <figure className="mr-chart">
      <figcaption className="mr-chart__title">{title}</figcaption>
      <div className="mr-chart__plot" ref={ref}>
        <svg viewBox={`0 0 ${W} ${h}`} role="img" aria-label={title}>
          {refValue !== undefined && (
            <g>
              {rows.map((r, i) => (
                <line key={r.id} x1={sx(refValue)} x2={sx(refValue)} y1={i * rowH + 16} y2={i * rowH + 38} className="mr-ref" />
              ))}
              <text x={sx(refValue)} y={h - 4} className="mr-tick">↑ {refLabel}</text>
            </g>
          )}
          {rows.map((r, i) => {
            const y = i * rowH + 6
            const x1 = sx(r.value)
            return (
              <g key={r.id} onPointerEnter={() => setHover(r.id)} onPointerLeave={() => setHover(null)}>
                <rect x={0} y={y - 4} width={W} height={rowH} fill="transparent" />
                <text x={L} y={y + 4} className="mr-tick">{r.label}</text>
                {/* square at the baseline, 4px rounded data-end */}
                <path d={`M${L},${y + 12} H${Math.max(L, x1 - 4)} a4,4 0 0 1 4,4 V${y + 26} a4,4 0 0 1 -4,4 H${L} Z`}
                  fill={SERIES[0]} opacity={hover && hover !== r.id ? 0.45 : 1} />
                <text x={x1 + 6} y={y + 21} className="mr-label" dominantBaseline="middle">
                  {format(r.value)}{r.note ? `  ${r.note}` : ''}
                </text>
              </g>
            )
          })}
        </svg>
      </div>
    </figure>
  )
}

/** Before / after on one line per model (fp32 -> int8), the gap drawn between the two dots. */
export function DumbbellChart({ rows, domain, labels, format, title }: {
  rows: { id: string; label: string; a: number; b: number }[]
  domain: [number, number]; labels: [string, string]; format: (v: number) => string; title: string
}) {
  const [hover, setHover] = useState<string | null>(null)
  const [ref, W] = useWidth()
  const rowH = 64
  const L = 28
  const R = 40
  const h = rows.length * rowH + 24
  const sx = (v: number) => L + ((v - domain[0]) / (domain[1] - domain[0])) * (W - L - R)
  const ticks = niceTicks(domain[0], domain[1], 4).filter((t) => t >= domain[0] && t <= domain[1])
  return (
    <figure className="mr-chart">
      <figcaption className="mr-chart__title">{title}</figcaption>
      <Legend items={[{ label: labels[0], color: SERIES[0] }, { label: labels[1], color: SERIES[1] }]} />
      <div className="mr-chart__plot" ref={ref}>
        <svg viewBox={`0 0 ${W} ${h}`} role="img" aria-label={title}>
          {ticks.map((t) => (
            <g key={t}>
              <line x1={sx(t)} x2={sx(t)} y1={4} y2={h - 26} className="mr-grid" />
              <text x={sx(t)} y={h - 10} className="mr-tick" textAnchor="middle">{format(t)}</text>
            </g>
          ))}
          {rows.map((r, i) => {
            const y = i * rowH + 40
            const hi = r.a >= r.b
            const same = format(r.a) === format(r.b)
            return (
              <g key={r.id} onPointerEnter={() => setHover(r.id)} onPointerLeave={() => setHover(null)}>
                <rect x={0} y={y - 20} width={W} height={rowH} fill="transparent" />
                <text x={L} y={y - 24} className="mr-label">{r.label}</text>
                <line x1={sx(r.a)} x2={sx(r.b)} y1={y} y2={y} className="mr-gap" />
                <circle cx={sx(r.a)} cy={y} r="6" fill={SERIES[0]} className="mr-ring" />
                <circle cx={sx(r.b)} cy={y} r="6" fill={SERIES[1]} className="mr-ring" />
                <text x={sx(r.a)} y={y + (same ? 18 : hi ? -14 : 18)} className="mr-tick" textAnchor="middle">{format(r.a)}</text>
                {!same && <text x={sx(r.b)} y={y + (hi ? 18 : -14)} className="mr-tick" textAnchor="middle">{format(r.b)}</text>}
                {hover === r.id && (
                  <text x={W - R} y={y - 14} className="mr-label" textAnchor="end">
                    {labels[0]} {format(r.a)} → {labels[1]} {format(r.b)}
                  </text>
                )}
              </g>
            )
          })}
        </svg>
      </div>
    </figure>
  )
}
