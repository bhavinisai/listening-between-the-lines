#!/usr/bin/env python3
"""
Re-save pretrained checkpoints that only ship pytorch_model.bin as
safetensors. transformers refuses to torch.load .bin files on torch < 2.6
(CVE-2025-32434); the .bin is read with weights_only=True here instead.
Run on a node with internet (login node); output goes to models/hf_converted/.
"""
import os

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import snapshot_download

from score_models import MODELS

for name, repo in MODELS.items():
    out = f"models/hf_converted/{name}"
    if os.path.exists(f"{out}/model.safetensors"):
        continue
    src = snapshot_download(repo, ignore_patterns=["*.h5", "*.msgpack", "*.onnx", "*.ot", "tf_*"])
    cfg = AutoConfig.from_pretrained(src)
    model = AutoModelForSequenceClassification.from_config(cfg)
    if os.path.exists(f"{src}/model.safetensors"):
        from safetensors.torch import load_file
        sd = load_file(f"{src}/model.safetensors")
    else:
        sd = torch.load(f"{src}/pytorch_model.bin", map_location="cpu", weights_only=True)
    print(name, model.load_state_dict(sd, strict=False))
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(src).save_pretrained(out)
