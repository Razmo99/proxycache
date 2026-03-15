# config.py
# -*- coding: utf-8 -*-

"""
Единая конфигурация для simple_proxycache:
- BACKENDS: [{"url": "...", "n_slots": N}]
- WORDS_PER_BLOCK, BIG_THRESHOLD_WORDS, LCP_TH
- PORT, REQUEST_TIMEOUT, MODEL_ID
- MAX_SAVED_CACHES, SLOT_SAVE_PATH
"""

import os
import json
import logging

# Backends
BACKENDS_RAW = os.getenv("BACKENDS")
if BACKENDS_RAW:
    try:
        BACKENDS = json.loads(BACKENDS_RAW)
    except Exception:
        BACKENDS = []
else:
    BACKENDS = [
        {
            "url": os.getenv("LLAMA_URL", "http://127.0.0.1:8000"),
            "n_slots": int(os.getenv("N_SLOTS", "2")),
        }
    ]


def _default_max_saved_caches() -> int:
    env_value = os.getenv("N_SLOTS")
    if env_value:
        try:
            return max(0, int(env_value))
        except Exception:
            pass

    if BACKENDS:
        try:
            return max(0, int(BACKENDS[0].get("n_slots", 0)))
        except Exception:
            pass

    return 0

# Words per block for LCP
WORDS_PER_BLOCK = int(os.getenv("WORDS_PER_BLOCK", "100"))

# Big request threshold
BIG_THRESHOLD_WORDS = int(os.getenv("BIG_THRESHOLD_WORDS", "500"))

# LCP threshold (0..1)
LCP_TH = float(os.getenv("LCP_TH", "0.6"))

# Meta dir
META_DIR = os.path.join(os.getcwd(), os.getenv("META_DIR", "kv_meta"))
os.makedirs(META_DIR, exist_ok=True)

# Save/restore retention
MAX_SAVED_CACHES = int(
    os.getenv("MAX_SAVED_CACHES", str(_default_max_saved_caches()))
)
SLOT_SAVE_PATH = os.getenv(
    "SLOT_SAVE_PATH",
    os.path.dirname(os.path.dirname(META_DIR))
    if os.path.basename(META_DIR.rstrip(os.sep)) == "meta"
    else os.path.dirname(META_DIR),
)

# HTTP timeout
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "600"))

# Model id
MODEL_ID = os.getenv("MODEL_ID", "llama.cpp")

# Service port
PORT = int(os.getenv("PORT", "8081"))

# Logs
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=LOG_LEVEL.upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
