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

/** Lazy, cached: onnxruntime and the model are downloaded only when the feature is first used. */
export function loadFoodModel(modelUrl: string, metaUrl: string, backends: Backend[] = ["wasm"], numThreads = 1) {
  if (!loaded) {
    loaded = (async () => {
      ort.env.wasm.numThreads = numThreads; // > 1 needs a cross-origin isolated page (COOP/COEP)
      const meta: ModelMeta = await (await fetch(metaUrl)).json();
      const session = await ort.InferenceSession.create(modelUrl, { executionProviders: backends });
      return { session, meta };
    })();
  }
  return loaded;
}

export function resetFoodModel() {
  loaded = null;
}

/** Same geometry as Python eval: resize short side to `res`, center crop res x res. Returns RGBA pixels. */
export function cropResize(bitmap: ImageBitmap, res: number): Uint8ClampedArray {
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
export async function predictFile(file: Blob, modelUrl: string, metaUrl: string): Promise<Prediction> {
  const { session, meta } = await loadFoodModel(modelUrl, metaUrl);
  const t0 = performance.now();
  // imageOrientation "from-image": apply the EXIF rotation of phone photos (Python: ImageOps.exif_transpose).
  const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  const data = toTensorData(cropResize(bitmap, meta.resolution), meta.resolution, meta.mean, meta.std);
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
