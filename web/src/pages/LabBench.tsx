// Hidden benchmark page (route /lab/bench, not in the menu). Copy to CovertCook/src/pages/LabBench.tsx
// and add the route in the router. Results are copied as JSON -> results/bench_raw/<date>.json in the ML repo.
import { useState } from "react";
import { bench, type BenchResult } from "../lib/bench";
import type { Backend, ModelMeta } from "../lib/foodModel";

// Edit this list. Trained runs: files copied by scripts.copy_to_web into public/models/ (<run>__<variant>.onnx,
// one <run>__meta.json). Screening candidates: models/candidates/* copied to public/models/candidates/.
const VARIANTS = ["/models/a_convnext__model_fp32.onnx", "/models/a_convnext__model_int8_minmax_200_pc_only-conv_rr.onnx", "/models/a_lite0__model_fp32.onnx", "/models/a_lite0__model_int8_minmax_200_pc_rr.onnx"];

const metaUrlOf = (url: string) =>
  url.includes("__") ? url.replace(/__[^/]*$/, "__meta.json") : url.replace(/\.onnx$/, ".meta.json");

export default function LabBench() {
  const [device, setDevice] = useState("");
  const [threads, setThreads] = useState(1);
  const [campaigns, setCampaigns] = useState(1);
  const [photo, setPhoto] = useState<File | null>(null);
  const [log, setLog] = useState<BenchResult[]>([]);
  const [running, setRunning] = useState(false);
  const [done, setDone] = useState(0);
  const [copyState, setCopyState] = useState("");
  const [withGpu, setWithGpu] = useState(true);
  const [status, setStatus] = useState("");
  const hasGpu = "gpu" in navigator;
  const backends: Backend[] = hasGpu && withGpu ? ["wasm", "webgpu"] : ["wasm"];
  const total = campaigns * VARIANTS.length * backends.length;

  async function runAll() {
    setRunning(true);
    setDone(0);
    setStatus("");
    // A phone turns its screen off after ~30 s and Chrome then suspends the page: the run froze at 1 result.
    // The wake lock keeps the screen on while measuring (released automatically if the tab is hidden).
    let lock: { release: () => Promise<void> } | null = null;
    try {
      lock = await (navigator as Navigator & { wakeLock?: { request: (t: "screen") => Promise<{ release: () => Promise<void> }> } })
        .wakeLock?.request("screen") ?? null;
    } catch {
      setStatus("screen wake lock refused: keep touching the screen");
    }
    const out: BenchResult[] = [];
    for (let c = 0; c < campaigns; c++) {
      for (const url of VARIANTS) {
        const meta: ModelMeta = await (await fetch(metaUrlOf(url))).json();
        for (const b of backends) {
          const name = url.split("/").pop()!.replace(".onnx", "");
          setStatus(`campaign ${c + 1}/${campaigns}: ${name} on ${b}${b === "wasm" ? `/${threads}` : ""}…`);
          try {
            out.push(await bench(url, meta, b, b === "wasm" ? threads : 1, photo, device));
          } catch (e) {
            console.error(url, b, e);
            setStatus(`failed: ${name} on ${b}: ${e instanceof Error ? e.message : String(e)}`);
          }
          setLog([...out]);
          setDone((d) => d + 1);
        }
      }
    }
    await lock?.release().catch(() => undefined);
    setStatus(`finished: ${out.length} / ${total} measures`);
    setRunning(false);
  }

  const json = () => JSON.stringify(log, null, 2);

  // The clipboard is refused on some phones (no recognised gesture, permissions): say so, and keep the download.
  async function copyJson() {
    try {
      await navigator.clipboard.writeText(json());
      setCopyState("copied");
    } catch {
      setCopyState("copy refused: use download");
    }
  }

  function downloadJson() {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([json()], { type: "application/json" }));
    a.download = `bench_${(device || "phone").replace(/\W+/g, "-")}_t${threads}_c${campaigns}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  return (
    <div style={{ padding: 16, fontFamily: "monospace" }}>
      <h2>Model benchmark</h2>
      <p>crossOriginIsolated: {String(globalThis.crossOriginIsolated)} | cores: {navigator.hardwareConcurrency}</p>
      <input placeholder="phone model" value={device} onChange={(e) => setDevice(e.target.value)} />
      <label> wasm threads <input type="number" min={1} max={8} value={threads}
        onChange={(e) => setThreads(+e.target.value)} /></label>
      <label> campaigns <input type="number" min={1} max={5} value={campaigns}
        onChange={(e) => setCampaigns(+e.target.value)} /></label>
      <input type="file" accept="image/*" onChange={(e) => setPhoto(e.target.files?.[0] ?? null)} />
      {hasGpu && (
        <label> <input type="checkbox" checked={withGpu} disabled={running}
          onChange={(e) => setWithGpu(e.target.checked)} /> also WebGPU</label>
      )}
      <button disabled={running || !device} onClick={runAll}>{running ? "running…" : "run"}</button>
      {/* Disabled while running: a file downloaded halfway holds only part of the measures. */}
      <button disabled={running || log.length === 0} onClick={copyJson}>copy JSON</button>
      <button disabled={running || log.length === 0} onClick={downloadJson}>download JSON</button>
      {copyState && <span> {copyState}</span>}
      {(running || done > 0) && (
        <p>
          <progress max={total} value={done} style={{ width: "100%" }} /> {done} / {total}
        </p>
      )}
      {status && <p>{status}</p>}
      {/* Wide table: it scrolls sideways inside its box instead of being cut by the phone screen. */}
      <div style={{ overflowX: "auto", maxWidth: "100%" }}>
        <table style={{ borderCollapse: "collapse", whiteSpace: "nowrap", fontSize: 12 }}>
          <thead><tr><th>variant</th><th>backend</th><th>load ms</th><th>pre ms</th><th>median ms</th><th>p95 ms</th></tr></thead>
          <tbody>
            {log.map((r, i) => (
              <tr key={i} style={{ borderTop: "1px solid #ccc" }}>
                <td style={{ paddingRight: 8 }}>{r.variant.replace("__model_", " ")}</td><td>{r.backend}/{r.numThreads}</td>
                <td align="right">{r.loadMs.toFixed(0)}</td><td align="right">{r.preMs?.toFixed(1) ?? "-"}</td>
                <td align="right">{r.runMedian.toFixed(1)}</td><td align="right">{r.runP95.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
