import os
import json
import logging
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

DIMENSION = 384  # all-MiniLM-L6-v2 output size

# FAISS_DIR can be overridden via env var to point to a Render persistent disk
# e.g. FAISS_DIR=/data  on Render, defaults to ./data locally
_faiss_dir = Path(os.getenv("FAISS_DIR", "data"))
INDEX_PATH = _faiss_dir / "faiss.index"
METADATA_PATH = _faiss_dir / "faiss_metadata.json"


class VectorStore:
    def __init__(self, index_path: Path = INDEX_PATH, metadata_path: Path = METADATA_PATH):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index: faiss.IndexFlatL2
        self.metadata: List[Dict[str, Any]] = []
        self._load_or_create()

    def _load_or_create(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        if self.index_path.exists() and self.metadata_path.exists():
            logger.info("Loading existing FAISS index from disk...")
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded {self.index.ntotal} vectors from index.")
        else:
            logger.info("Creating new FAISS index.")
            self.index = faiss.IndexFlatL2(DIMENSION)
            self.metadata = []

    def _save(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f)

    def add(self, vector: np.ndarray, meta: Dict[str, Any]) -> None:
        existing_ids = {m["digest_id"] for m in self.metadata}
        if meta.get("digest_id") in existing_ids:
            logger.debug(f"Skipping duplicate: {meta.get('digest_id')}")
            return
        self.index.add(np.array([vector], dtype="float32"))
        self.metadata.append(meta)
        self._save()

    def search(self, query_vector: np.ndarray, top_k: int = 20) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty — no results.")
            return []

        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(
            np.array([query_vector], dtype="float32"), k
        )

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            entry = dict(self.metadata[idx])
            entry["_distance"] = float(dist)
            results.append(entry)

        return results

    def is_indexed(self, digest_id: str) -> bool:
        return any(m["digest_id"] == digest_id for m in self.metadata)

    def total(self) -> int:
        return self.index.ntotal

    def rebuild_from_digests(self, digests: List[Dict[str, Any]], get_embedding_fn) -> int:
        added = 0
        for d in digests:
            if self.is_indexed(d["id"]):
                continue
            text = f"{d['title']} {d['summary']}"
            vec = get_embedding_fn(text)
            self.add(vec, {
                "digest_id": d["id"],
                "article_type": d["article_type"],
                "title": d["title"],
                "summary": d["summary"],
                "url": d["url"],
            })
            added += 1
        logger.info(f"Backfill complete: {added} new vectors added.")
        return added