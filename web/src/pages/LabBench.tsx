// Hidden benchmark page (route /lab/bench, not in the menu). Copy to CovertCook/src/pages/LabBench.tsx
// and add the route in the router. Results are copied as JSON -> results/bench_raw/<date>.json in the ML repo.
import { useState } from "react";
import { bench, type BenchResult } from "../lib/bench";
import type { Backend, ModelMeta } from "../lib/foodModel";

// Edit this list. Trained runs: files copied by scripts.copy_to_web into public/models/ (<run>__<variant>.onnx,
// one <run>__meta.json). Screening candidates: models/candidates/* copied to public/models/candidates/.
const VARIANTS = [
  "/models/candidates/tf_efficientnet_lite0_224.onnx",
  "/models/candidates/mobilenetv3_large_100_224.onnx",
  "/models/candidates/mobilenetv3_small_100_224.onnx",
  "/models/candidates/efficientnet_b0_224.onnx",
  "/models/candidates/mobilevit_xs_224.onnx",
  "/models/candidates/convnext_atto_224.onnx",
  // "/models/a_lite0__model_fp32.onnx",
  // "/models/a_lite0__model_int8_minmax_200_pc.onnx",
];

const metaUrlOf = (url: string) =>
  url.includes("__") ? url.replace(/__[^/]*$/, "__meta.json") : url.replace(/\.onnx$/, ".meta.json");

export default function LabBench() {
  const [device, setDevice] = useState("");
  const [threads, setThreads] = useState(1);
  const [campaigns, setCampaigns] = useState(1);
  const [photo, setPhoto] = useState<File | null>(null);
  const [log, setLog] = useState<BenchResult[]>([]);
  const [running, setRunning] = useState(false);

  async function runAll() {
    setRunning(true);
    const out: BenchResult[] = [];
    for (let c = 0; c < campaigns; c++) {
      for (const url of VARIANTS) {
        const meta: ModelMeta = await (await fetch(metaUrlOf(url))).json();
        const backends: Backend[] = "gpu" in navigator ? ["wasm", "webgpu"] : ["wasm"];
        for (const b of backends) {
          try {
            out.push(await bench(url, meta, b, b === "wasm" ? threads : 1, photo, device));
          } catch (e) {
            console.error(url, b, e);
          }
          setLog([...out]);
        }
      }
    }
    setRunning(false);
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
      <button disabled={running || !device} onClick={runAll}>{running ? "running…" : "run"}</button>
      <button onClick={() => navigator.clipboard.writeText(JSON.stringify(log, null, 2))}>copy JSON</button>
      <table>
        <thead><tr><th>variant</th><th>backend</th><th>load</th><th>pre</th><th>median</th><th>p95</th></tr></thead>
        <tbody>
          {log.map((r, i) => (
            <tr key={i}>
              <td>{r.variant}</td><td>{r.backend}/{r.numThreads}</td><td>{r.loadMs.toFixed(0)}</td>
              <td>{r.preMs?.toFixed(1) ?? "-"}</td><td>{r.runMedian.toFixed(1)}</td><td>{r.runP95.toFixed(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
