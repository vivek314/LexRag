#!/usr/bin/env python
"""
Build FAISS indices using fastembed (open-source, no API key).
Output goes to data/indices/oss/ — commit these files for Vercel demo.
Run once locally before deploying: python scripts/build_oss_indices.py
"""
import sys
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.providers.hf_provider import FastEmbedProvider
from src.data.indexing import build_indices

with open("configs/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

oss_cfg = copy.deepcopy(cfg)
oss_cfg["data"]["indices_dir"] = "data/indices/oss"
oss_cfg["embedding"]["cache_dir"] = "data/processed/embedding_cache_oss"
oss_cfg["embedding"]["dimensions"] = 384

provider = FastEmbedProvider()
build_indices(oss_cfg, embedding_provider=provider)
print("OSS indices built -> data/indices/oss/")
