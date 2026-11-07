// User-facing page: take/choose a photo -> ingredients as removable chips (+ top-3 dishes in part D).
// Copy to CovertCook/src/pages/FoodScan.tsx. Inference runs entirely on the phone.
import { useState } from "react";
import { predictFile, type Ingredient, type Prediction } from "../lib/foodModel";

const MODEL_URL = "/models/a_lite0__model_int8_minmax_200_pc.onnx"; // the variant chosen in step A13
const META_URL = "/models/a_lite0__meta.json";

export default function FoodScan() {
  const [pred, setPred] = useState<Prediction | null>(null);
  const [chips, setChips] = useState<Ingredient[]>([]);
  const [busy, setBusy] = useState(false);

  async function onFile(file: File | undefined) {
    if (!file) return;
    setBusy(true);
    const p = await predictFile(file, MODEL_URL, META_URL);
    setPred(p);
    setChips(p.ingredients);
    setBusy(false);
  }

  return (
    <div style={{ padding: 16 }}>
      <input type="file" accept="image/*" capture="environment" onChange={(e) => onFile(e.target.files?.[0])} />
      {busy && <p>…</p>}
      {pred?.unknown && <p>Nothing recognised. Add the ingredients by hand.</p>}
      {pred && pred.dishes.length > 0 && (
        <ol>{pred.dishes.map((d) => <li key={d.name}>{d.name}</li>)}</ol>
      )}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
        {chips.map((c) => (
          // Human in the loop: the user removes what is wrong (and later adds what is missing).
          <button key={c.id} onClick={() => setChips(chips.filter((x) => x.id !== c.id))}>{c.name} ✕</button>
        ))}
      </div>
      {pred && <small>pre {pred.timings.preMs.toFixed(0)} ms · model {pred.timings.runMs.toFixed(0)} ms</small>}
    </div>
  );
}
