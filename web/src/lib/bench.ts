// On-device benchmark of one ONNX variant. Copy to CovertCook/src/lib/bench.ts.
import * as ort from "onnxruntime-web";
import { cropResize, postprocess, toTensorData, type Backend, type ModelMeta } from "./foodModel";

ort.env.wasm.wasmPaths = "/ort/";

export interface BenchResult {
  variant: string;           // file stem, e.g. "a_lite0__model_int8_minmax_200_pc" (joins with results/metrics.csv)
  backend: Backend;
  numThreads: number;
  resolution: number;
  runs: number;
  loadMs: number;            // cold start: InferenceSession.create
  preMs: number | null;      // resize + normalise of a real photo (null if no photo given)
  runMedian: number;
  runP95: number;
  postMs: number | null;
  crossOriginIsolated: boolean;
  userAgent: string;
  device: string;            // phone model, typed by you
  timestamp: string;
}

const median = (a: number[]) => a[Math.floor(a.length / 2)];
const p95 = (a: number[]) => a[Math.min(a.length - 1, Math.floor(a.length * 0.95))];

export async function bench(
  modelUrl: string, meta: ModelMeta, backend: Backend, numThreads: number,
  photo: Blob | null, device: string, runs = 100, warmup = 10,
): Promise<BenchResult> {
  ort.env.wasm.numThreads = numThreads;
  const res = meta.resolution;

  const t0 = performance.now();
  const session = await ort.InferenceSession.create(modelUrl, { executionProviders: [backend] });
  const loadMs = performance.now() - t0;

  // Input: a real photo if given (measures preprocessing too), zeros otherwise.
  let preMs: number | null = null;
  let data = new Float32Array(3 * res * res);
  if (photo) {
    const tp = performance.now();
    const bitmap = await createImageBitmap(photo, { imageOrientation: "from-image" });
    data = toTensorData(cropResize(bitmap, res), res, meta.mean, meta.std);
    preMs = performance.now() - tp;
  }
  const feeds = { [session.inputNames[0]]: new ort.Tensor("float32", data, [1, 3, res, res]) };

  for (let i = 0; i < warmup; i++) await session.run(feeds);
  const times: number[] = [];
  let last: ort.InferenceSession.OnnxValueMapType | null = null;
  for (let i = 0; i < runs; i++) {
    const t = performance.now();
    last = await session.run(feeds);
    times.push(performance.now() - t);
  }
  times.sort((a, b) => a - b);

  let postMs: number | null = null;
  if (last) {
    const tq = performance.now();
    const dish = meta.outputs.dish ? (last[meta.outputs.dish].data as Float32Array) : null;
    postprocess(last[meta.outputs.ingredients].data as Float32Array, dish, meta);
    postMs = performance.now() - tq;
  }
  await session.release();

  return {
    variant: modelUrl.split("/").pop()!.replace(/\.onnx$/, ""),
    backend, numThreads, resolution: res, runs, loadMs, preMs,
    runMedian: median(times), runP95: p95(times), postMs,
    crossOriginIsolated: globalThis.crossOriginIsolated ?? false,
    userAgent: navigator.userAgent, device, timestamp: new Date().toISOString(),
  };
}
