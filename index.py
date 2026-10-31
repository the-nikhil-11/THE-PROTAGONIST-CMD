import os
import re
import json
import math
import logging
import urllib.request
from typing import List, Dict, Set, Tuple, Any, Optional
import numpy as np

from config import cfg
from sources import CommandRecord

logger = logging.getLogger("PROTAGONIST_INDEX")


def tokenize_technical_text(text: str) -> List[str]:
    """Preserves technical tokens like flags (--message, -m, /s), subcommands, env vars ($var), etc."""
    if not text:
        return []
    # Normalize unicode and lower case
    text_clean = text.lower()
    # Match technical tokens including flags (-f, --flag, /flag), dotted paths, hyphens, underscores, colons
    pattern = r"(?:--?[\w-]+|/\w+|[\$\%][\w]+|[\w\.-]+)"
    tokens = re.findall(pattern, text_clean)
    return [t for t in tokens if len(t) > 1 or t in ["-a", "-b", "-c", "-d", "-f", "-i", "-l", "-m", "-n", "-o", "-p", "-r", "-s", "-u", "-v", "-w", "-x", "-y", "-z"]]


class IndexEngine:
    def __init__(self, records: List[CommandRecord]):
        self.records: Dict[str, CommandRecord] = {r.id: r for r in records}
        self.doc_ids: List[str] = list(self.records.keys())
        self.N: int = len(self.records)

        # BM25 & Inverted Index Data Structures
        self.inverted_index: Dict[str, Set[str]] = {}
        self.doc_freq: Dict[str, int] = {}
        self.doc_lens: Dict[str, int] = {}
        self.avg_dl: float = 0.0

        # Field-aware term frequencies: doc_id -> field -> token -> count
        self.field_tf: Dict[str, Dict[str, Dict[str, int]]] = {}

        # Vector Store: matrix shape (N_vec, D) & corresponding doc_ids
        self.vector_matrix: Optional[np.ndarray] = None
        self.vector_doc_ids: List[str] = []

    def build_lexical_index(self):
        """Constructs field-aware inverted index and BM25 corpus statistics."""
        logger.info(f"Building lexical BM25 index over {self.N} records...")
        total_len = 0

        for doc_id, record in self.records.items():
            self.field_tf[doc_id] = {}
            doc_total_len = 0

            # Tokenize individual record fields
            field_data = {
                "command": record.command,
                "syntax": record.syntax,
                "flags": " ".join(record.flags),
                "aliases": " ".join(record.aliases),
                "description": record.description,
                "examples": " ".join(record.examples),
                "ecosystem": record.ecosystem
            }

            for field_name, content in field_data.items():
                tokens = tokenize_technical_text(content)
                weight = cfg.field_weights.get(field_name, 1.0)
                doc_total_len += int(len(tokens) * weight)

                if field_name not in self.field_tf[doc_id]:
                    self.field_tf[doc_id][field_name] = {}

                for token in tokens:
                    self.field_tf[doc_id][field_name][token] = self.field_tf[doc_id][field_name].get(token, 0) + 1
                    
                    if token not in self.inverted_index:
                        self.inverted_index[token] = set()
                    self.inverted_index[token].add(doc_id)

            self.doc_lens[doc_id] = max(1, doc_total_len)
            total_len += doc_total_len

        self.avg_dl = total_len / float(self.N) if self.N > 0 else 1.0
        self.doc_freq = {token: len(doc_set) for token, doc_set in self.inverted_index.items()}
        logger.info(f"Lexical index complete. Vocabulary size: {len(self.inverted_index)} tokens. Avg doc length: {self.avg_dl:.2f}")

    def generate_embeddings_ollama(self, texts: List[str]) -> List[List[float]]:
        """Communicates with Ollama API to generate vector embeddings."""
        url = f"{cfg.ollama_host}/api/embed"
        payload = {
            "model": cfg.ollama_embed_model,
            "input": texts
        }
        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=req_data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=cfg.request_timeout) as resp:
                if resp.status == 200:
                    res = json.loads(resp.read().decode("utf-8"))
                    if "embeddings" in res:
                        return res["embeddings"]
        except Exception as e:
            # Try legacy endpoint /api/embeddings for single text or fallback
            url_legacy = f"{cfg.ollama_host}/api/embeddings"
            results = []
            for t in texts:
                try:
                    payload_leg = {"model": cfg.ollama_embed_model, "prompt": t}
                    req_data = json.dumps(payload_leg).encode("utf-8")
                    req = urllib.request.Request(url_legacy, data=req_data, headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=cfg.request_timeout) as resp:
                        if resp.status == 200:
                            res = json.loads(resp.read().decode("utf-8"))
                            if "embedding" in res:
                                results.append(res["embedding"])
                except Exception as ex:
                    logger.warning(f"Ollama embedding fallback failed: {ex}")
                    return []
            if len(results) == len(texts):
                return results

        logger.warning("Ollama embedding API call failed or returned invalid response.")
        return []

    def build_vector_index(self):
        """Precomputes vector embeddings for canonical records."""
        logger.info(f"Attempting to build vector index using model '{cfg.ollama_embed_model}'...")
        texts = [r.search_text for r in self.records.values()]
        doc_ids = list(self.records.keys())

        # Batch embed requests
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            embeddings = self.generate_embeddings_ollama(batch_texts)
            if not embeddings:
                logger.warning("Semantic embeddings generation failed/skipped. Vector index will remain empty.")
                self.vector_matrix = None
                self.vector_doc_ids = []
                return
            all_embeddings.extend(embeddings)

        if all_embeddings:
            matrix = np.array(all_embeddings, dtype=np.float32)
            # Normalize vectors for fast cosine dot product
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.vector_matrix = matrix / norms
            self.vector_doc_ids = doc_ids
            logger.info(f"Vector index built successfully. Matrix shape: {self.vector_matrix.shape}")

    def save(self):
        """Persists canonical records, inverted index, BM25 stats, and vector embeddings."""
        # 1. Corpus records
        with open(cfg.corpus_file, "w", encoding="utf-8") as f:
            json.dump({doc_id: r.to_dict() for doc_id, r in self.records.items()}, f, indent=2)

        # 2. Inverted index & BM25 stats
        inv_serializable = {token: list(doc_set) for token, doc_set in self.inverted_index.items()}
        with open(cfg.inverted_index_file, "w", encoding="utf-8") as f:
            json.dump(inv_serializable, f, indent=2)

        bm25_stats = {
            "doc_freq": self.doc_freq,
            "doc_lens": self.doc_lens,
            "avg_dl": self.avg_dl,
            "N": self.N,
            "field_tf": self.field_tf
        }
        with open(cfg.bm25_stats_file, "w", encoding="utf-8") as f:
            json.dump(bm25_stats, f, indent=2)

        # 3. Vector matrix & meta
        if self.vector_matrix is not None and len(self.vector_doc_ids) > 0:
            np.save(str(cfg.vector_index_file), self.vector_matrix)
            with open(cfg.vector_meta_file, "w", encoding="utf-8") as f:
                json.dump(self.vector_doc_ids, f, indent=2)

        logger.info("All indexes saved to disk successfully.")

    @classmethod
    def load(cls) -> "IndexEngine":
        """Loads existing indexes from disk if available."""
        if not (cfg.corpus_file.exists() and cfg.inverted_index_file.exists() and cfg.bm25_stats_file.exists()):
            raise FileNotFoundError("Index files missing.")

        with open(cfg.corpus_file, "r", encoding="utf-8") as f:
            raw_corpus = json.load(f)
        records = [CommandRecord.from_dict(data) for data in raw_corpus.values()]

        engine = cls(records)

        with open(cfg.inverted_index_file, "r", encoding="utf-8") as f:
            inv_raw = json.load(f)
        engine.inverted_index = {token: set(doc_list) for token, doc_list in inv_raw.items()}

        with open(cfg.bm25_stats_file, "r", encoding="utf-8") as f:
            stats = json.load(f)
        engine.doc_freq = stats["doc_freq"]
        engine.doc_lens = stats["doc_lens"]
        engine.avg_dl = stats["avg_dl"]
        engine.N = stats["N"]
        engine.field_tf = stats["field_tf"]

        if cfg.vector_index_file.exists() and cfg.vector_meta_file.exists():
            try:
                engine.vector_matrix = np.load(str(cfg.vector_index_file))
                with open(cfg.vector_meta_file, "r", encoding="utf-8") as f:
                    engine.vector_doc_ids = json.load(f)
                logger.info(f"Loaded vector matrix with shape {engine.vector_matrix.shape}")
            except Exception as e:
                logger.warning(f"Could not load vector index: {e}")

        return engine


def build_or_load_indexes(force_rebuild: bool = False) -> IndexEngine:
    """Entry point for index initialization."""
    if not force_rebuild:
        try:
            engine = IndexEngine.load()
            logger.info("Loaded pre-built offline indexes.")
            return engine
        except Exception as e:
            logger.info(f"Could not load existing indexes ({e}). Building from scratch...")

    from sources import load_canonical_corpus
    records = load_canonical_corpus()
    engine = IndexEngine(records)
    engine.build_lexical_index()
    engine.build_vector_index()
    engine.save()
    return engine
