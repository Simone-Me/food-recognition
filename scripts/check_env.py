"""Environment check: versions, GPU, Hugging Face cache location, one timm forward pass on the GPU.

    python -m scripts.check_env
"""
import os
import platform

import torch


def main() -> None:
    print("python      :", platform.python_version())
    print("torch       :", torch.__version__, "| cuda runtime in wheel:", torch.version.cuda)
    print("cuda ok     :", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("gpu         :", p.name, f"| {p.total_memory / 2**30:.1f} GB | compute capability {p.major}.{p.minor}")

    import huggingface_hub.constants as hc
    print("HF_HOME     :", hc.HF_HOME, "(env:", os.environ.get("HF_HOME"), ")")
    if "onedrive" in str(hc.HF_HOME).lower():
        print("  !! the Hugging Face cache is inside OneDrive: set HF_HOME (see TP §A1)")

    import timm, onnx, onnxruntime, datasets, mlflow
    print("timm / onnx / onnxruntime / datasets / mlflow:",
          timm.__version__, onnx.__version__, onnxruntime.__version__, datasets.__version__, mlflow.__version__)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = timm.create_model("tf_efficientnet_lite0", pretrained=True).to(device).eval()
    with torch.no_grad():
        print("forward     :", tuple(m(torch.randn(1, 3, 224, 224, device=device)).shape), "(expected (1, 1000))")


if __name__ == "__main__":
    main()
