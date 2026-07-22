"""Download SentenceTransformer during the Vercel build."""

from __future__ import annotations

import gc
import os
import shutil
from pathlib import Path

from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model_files"
SEMANTIC_MODEL_DIR = MODEL_DIR / "semantic_model"

MODEL_ID = os.getenv(
    "SEMANTIC_MODEL_ID",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)


def main() -> None:
    cache_dir = Path("/tmp/medinlp-hf-cache")

    shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(
        cache_dir / "transformers"
    )
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    config_file = (
        SEMANTIC_MODEL_DIR
        / "config_sentence_transformers.json"
    )

    if config_file.exists():
        print("Semantic model already bundled.")
        return

    print(f"Downloading semantic model: {MODEL_ID}")

    model = SentenceTransformer(
        MODEL_ID,
        device="cpu",
        cache_folder=str(cache_dir),
    )

    model.save(str(SEMANTIC_MODEL_DIR))

    del model
    gc.collect()

    print(
        f"Semantic model saved to: "
        f"{SEMANTIC_MODEL_DIR}"
    )


if __name__ == "__main__":
    main()
