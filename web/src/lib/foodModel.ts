// Food recognition in the browser with onnxruntime-web.
// Copy this file to CovertCook/src/lib/foodModel.ts.
// Everything model-specific (resolution, mean/std, labels, thresholds) comes from <run>__meta.json,
// written by src/export_onnx.py: nothing is duplicated by hand between Python and JS.
import * as ort from "onnxruntime-web";

export interface ModelMeta {
  version: string;
  resolution: number;
  mean: [number, number, number];
  std: [number, number, number];
  input_name: string;
  outputs: { ingredients: string; dish?: string };
  ingredient_ids: number[];
  ingredient_names: string[];
  thresholds: number[];
  dish_names: string[];
  unknown_dish_min_prob: number | null;
  unknown_ingr_min_prob?: number | null; // highest ingredient prob below this -> "not recognised"
}

export interface Ingredient { id: number; name: string; prob: number }
export interface Dish { name: string; prob: number }
export interface Prediction {
  ingredients: Ingredient[];
  dishes: Dish[];          // top-3, empty for part A or when below unknown_dish_min_prob
  unknown: boolean;        // nothing recognised
  timings: { preMs: number; runMs: number; postMs: number };
}

export type Backend = "wasm" | "webgpu";

// onnxruntime-web loads its .wasm/.mjs files at runtime: serve them from the site itself
// (public/ort/), never from a CDN (CSP 'self' + offline PWA).
ort.env.wasm.wasmPaths = "/ort/";

let loaded: Promise<{ session: ort.InferenceSession; meta: ModelMeta }> | null = null;

/** Download progress, 0..1 (null when the server does not send the size). */
export type OnProgress = (fraction: number | null) => void;

/** fetch() that fails loudly. A missing file on a single-page app comes back as 200 + index.html:
 *  without this check the model "loads" an HTML page and never answers. */
async function fetchChecked(url: string): Promise<Response> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  if ((res.headers.get("content-type") ?? "").includes("text/html")) throw new Error(`${url}: file not found on the server`);
  return res;
}

/** Model bytes with download progress (the model is the slow part of the first use). */
async function fetchBytes(url: string, onProgress?: OnProgress): Promise<Uint8Array> {
  const res = await fetchChecked(url);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress?.(total ? Math.min(received / total, 1) : null);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) { bytes.set(c, offset); offset += c.length; }
  return bytes;
}

/** Lazy, cached: onnxruntime and the model are downloaded only when the feature is first used. */
export function loadFoodModel(modelUrl: string, metaUrl: string, backends: Backend[] = ["wasm"], numThreads = 1,
                              onProgress?: OnProgress) {
  if (!loaded) {
    loaded = (async () => {
      ort.env.wasm.numThreads = numThreads; // > 1 needs a cross-origin isolated page (COOP/COEP)
      const meta: ModelMeta = await (await fetchChecked(metaUrl)).json();
      const bytes = await fetchBytes(modelUrl, onProgress);
      const session = await ort.InferenceSession.create(bytes, { executionProviders: backends });
      return { session, meta };
    })();
    loaded.catch(() => { loaded = null; }); // a failed download must not stay cached: the next try starts over
  }
  return loaded;
}

export function resetFoodModel() {
  loaded = null;
}

/** Same geometry as Python eval: resize short side to `res`, center crop res x res. Returns RGBA pixels. */
/** A decoded photo, EXIF rotation applied (Python: ImageOps.exif_transpose).
 *  createImageBitmap first: fast, off the main thread. Some WebKit builds refuse certain JPEGs with it
 *  (progressive JPEGs, e.g. photos saved from WhatsApp): fall back to an <img>, which decodes any image
 *  the browser can show and applies the EXIF orientation by default. */
export type Photo = ImageBitmap | HTMLImageElement;

export async function decodePhoto(file: Blob): Promise<Photo> {
  try {
    return await createImageBitmap(file, { imageOrientation: "from-image" });
  } catch {
    const url = URL.createObjectURL(file);
    try {
      const img = new Image();
      img.src = url;
      await img.decode();
      return img;
    } finally {
      URL.revokeObjectURL(url);
    }
  }
}

/** Release a decoded photo now (tens of MB on a phone), not at garbage collection. */
export function releasePhoto(photo: Photo) {
  if ("close" in photo) photo.close();
}

export function cropResize(bitmap: Photo, res: number): Uint8ClampedArray {
  const canvas = typeof OffscreenCanvas !== "undefined"
    ? new OffscreenCanvas(res, res)
    : Object.assign(document.createElement("canvas"), { width: res, height: res });
  const ctx = canvas.getContext("2d") as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high"; // closest to Pillow's antialiased resize; verify with the parity page
  const side = Math.min(bitmap.width, bitmap.height); // square of the short side, centred
  const sx = (bitmap.width - side) / 2;
  const sy = (bitmap.height - side) / 2;
  ctx.drawImage(bitmap, sx, sy, side, side, 0, 0, res, res);
  return ctx.getImageData(0, 0, res, res).data;
}

/** Pixels (RGBA or RGB, row-major HWC, 0..255) -> normalised float32 CHW, exactly like torchvision. */
export function toTensorData(pixels: ArrayLike<number>, res: number, mean: number[], std: number[], channels = 4) {
  const out = new Float32Array(3 * res * res);
  const plane = res * res;
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) {
      out[c * plane + i] = (pixels[i * channels + c] / 255 - mean[c]) / std[c];
    }
  }
  return out;
}

/** Probabilities -> what the UI shows. Thresholds are per class, chosen on the validation set. */
export function postprocess(ingrProbs: ArrayLike<number>, dishProbs: ArrayLike<number> | null, meta: ModelMeta) {
  const ingredients: Ingredient[] = [];
  for (let k = 0; k < ingrProbs.length; k++) {
    if (ingrProbs[k] >= meta.thresholds[k]) {
      ingredients.push({ id: meta.ingredient_ids[k], name: meta.ingredient_names[k], prob: ingrProbs[k] });
    }
  }
  ingredients.sort((a, b) => b.prob - a.prob);
  // Open world: the model never saw non-food at training. If even its best ingredient is unsure,
  // answer "not recognised" instead of listing guesses (threshold chosen on the personal photos, A9).
  const maxProb = Math.max(...Array.from(ingrProbs));
  if (maxProb < (meta.unknown_ingr_min_prob ?? 0)) ingredients.length = 0;

  let dishes: Dish[] = [];
  if (dishProbs) {
    dishes = Array.from(dishProbs, (p, i) => ({ name: meta.dish_names[i], prob: p }))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, 3);
    const minProb = meta.unknown_dish_min_prob ?? 0;
    if (dishes[0].prob < minProb) dishes = [];
  }
  return { ingredients, dishes, unknown: ingredients.length === 0 && dishes.length === 0 };
}

/** Full pipeline on a photo picked by the user. */
export async function predictFile(file: Blob, modelUrl: string, metaUrl: string,
                                  onProgress?: OnProgress): Promise<Prediction> {
  const { session, meta } = await loadFoodModel(modelUrl, metaUrl, ["wasm"], 1, onProgress);
  const t0 = performance.now();
  const bitmap = await decodePhoto(file);
  const data = toTensorData(cropResize(bitmap, meta.resolution), meta.resolution, meta.mean, meta.std);
  releasePhoto(bitmap);
  const input = new ort.Tensor("float32", data, [1, 3, meta.resolution, meta.resolution]);
  const t1 = performance.now();
  const out = await session.run({ [meta.input_name]: input });
  const t2 = performance.now();
  const ingr = out[meta.outputs.ingredients].data as Float32Array;
  const dish = meta.outputs.dish ? (out[meta.outputs.dish].data as Float32Array) : null;
  const result = postprocess(ingr, dish, meta);
  const t3 = performance.now();
  return { ...result, timings: { preMs: t1 - t0, runMs: t2 - t1, postMs: t3 - t2 } };
}
