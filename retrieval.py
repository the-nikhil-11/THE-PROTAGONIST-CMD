import math
import time
import re
import logging
from typing import List, Dict, Tuple, Set, Any, Optional
import numpy as np

from config import cfg
from sources import CommandRecord
from index import IndexEngine, tokenize_technical_text

logger = logging.getLogger("PROTAGONIST_RETRIEVAL")



class RetrievalResult:
    def __init__(self, record: CommandRecord, score: float, rrf_score: float, bm25_rank: Optional[int], vector_rank: Optional[int], evidence_signals: Dict[str, float]):
        self.record = record
        self.score = score
        self.rrf_score = rrf_score
        self.bm25_rank = bm25_rank
        self.vector_rank = vector_rank
        self.evidence_signals = evidence_signals

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record": self.record.to_dict(),
            "score": round(self.score, 4),
            "rrf_score": round(self.rrf_score, 4),
            "bm25_rank": self.bm25_rank,
            "vector_rank": self.vector_rank,
            "evidence_signals": self.evidence_signals
        }


class QueryProcessor:
    """Extracts technical tokens, flags, ecosystems, and query intent."""

    ECOSYSTEM_KEYWORDS = {
        "git": ["git", "repository", "commit", "branch", "rebase", "checkout", "stash", "pull", "push", "remote", "clone", "merge"],
        "powershell": ["powershell", "cmdlet", "get-process", "get-childitem", "set-executionpolicy", "test-netconnection", "where-object", "select-object"],
        "windows_cmd": ["cmd", "command prompt", "dir", "netstat", "taskkill", "robocopy", "ipconfig", "cls"],
        "python": ["python", "pip", "venv", "pytest", "unittest", "http.server", "pydoc", "wheel", "conda"],
        "linux": ["bash", "linux", "sh", "grep", "find", "chmod", "tar", "journalctl", "awk", "sed", "systemctl", "sudo", "curl"]
    }

    def process_query(self, query: str) -> Dict[str, Any]:
        raw = query.strip()
        tokens = tokenize_technical_text(raw)
        
        # Extract explicit flags (e.g., -m, --message, /s)
        flags = [t for t in tokens if t.startswith("-") or t.startswith("/")]
        
        # Infer target ecosystem
        detected_eco = None
        query_lower = raw.lower()
        for eco, keywords in self.ECOSYSTEM_KEYWORDS.items():
            for kw in keywords:
                if kw in query_lower:
                    detected_eco = eco
                    break
            if detected_eco:
                break

        return {
            "raw_query": raw,
            "tokens": tokens,
            "flags": flags,
            "inferred_ecosystem": detected_eco
        }


class InformationRetrievalEngine:
    def __init__(self, index_engine: IndexEngine):
        self.index = index_engine
        self.query_processor = QueryProcessor()

    def _calculate_bm25_idf(self, token: str) -> float:
        df = self.index.doc_freq.get(token, 0)
        if df == 0:
            return 0.0
        # BM25 standard Robertson-Spärck Jones IDF
        val = (self.index.N - df + 0.5) / (df + 0.5) + 1.0
        return math.log(max(1.0, val))

    def lexical_search(self, query_tokens: List[str], top_k: int) -> List[Tuple[str, float]]:
        """Field-aware BM25 search over inverted index."""
        if not query_tokens:
            return []

        # Find documents containing at least one query token
        candidate_doc_ids: Set[str] = set()
        for token in query_tokens:
            if token in self.index.inverted_index:
                candidate_doc_ids.update(self.index.inverted_index[token])

        if not candidate_doc_ids:
            return []

        scores: Dict[str, float] = {}

        for doc_id in candidate_doc_ids:
            doc_len = self.index.doc_lens.get(doc_id, self.index.avg_dl)
            doc_score = 0.0

            # Sum over query tokens
            for token in set(query_tokens):
                idf = self._calculate_bm25_idf(token)
                if idf <= 0:
                    continue

                # Calculate field-weighted term frequency
                field_tf_map = self.index.field_tf.get(doc_id, {})
                weighted_tf = 0.0
                for field_name, tfs in field_tf_map.items():
                    if token in tfs:
                        weight = cfg.field_weights.get(field_name, 1.0)
                        weighted_tf += tfs[token] * weight

                if weighted_tf > 0:
                    num = weighted_tf * (cfg.bm25_k1 + 1.0)
                    den = weighted_tf + cfg.bm25_k1 * (1.0 - cfg.bm25_b + cfg.bm25_b * (doc_len / self.index.avg_dl))
                    doc_score += idf * (num / den)

            if doc_score > 0:
                scores[doc_id] = doc_score

        # Sort descending by BM25 score
        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_candidates[:top_k]

    def semantic_search(self, raw_query: str, top_k: int) -> List[Tuple[str, float]]:
        """Cosine similarity search using precomputed vector embeddings."""
        if self.index.vector_matrix is None or len(self.index.vector_doc_ids) == 0:
            return []

        # Generate query embedding
        query_embeddings = self.index.generate_embeddings_ollama([raw_query])
        if not query_embeddings:
            return []

        q_vec = np.array(query_embeddings[0], dtype=np.float32)
        norm = np.linalg.norm(q_vec)
        if norm == 0:
            return []
        q_vec_norm = q_vec / norm

        # Compute cosine similarity across normalized corpus matrix
        sims = np.dot(self.index.vector_matrix, q_vec_norm)
        
        # Rank top_k
        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_indices:
            doc_id = self.index.vector_doc_ids[idx]
            sim = float(sims[idx])
            if sim > 0:
                results.append((doc_id, sim))

        return results

    def compute_rrf(self, lexical_results: List[Tuple[str, float]], semantic_results: List[Tuple[str, float]]) -> Dict[str, Dict[str, Any]]:
        """Reciprocal Rank Fusion (RRF) algorithm."""
        rrf_scores: Dict[str, float] = {}
        bm25_ranks: Dict[str, int] = {}
        vector_ranks: Dict[str, int] = {}

        # Process lexical list
        for rank, (doc_id, _) in enumerate(lexical_results, start=1):
            bm25_ranks[doc_id] = rank
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (cfg.rrf_k + rank))

        # Process semantic list
        for rank, (doc_id, _) in enumerate(semantic_results, start=1):
            vector_ranks[doc_id] = rank
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (cfg.rrf_k + rank))

        merged = {}
        for doc_id, rrf_val in rrf_scores.items():
            merged[doc_id] = {
                "rrf_score": rrf_val,
                "bm25_rank": bm25_ranks.get(doc_id),
                "vector_rank": vector_ranks.get(doc_id)
            }
        return merged

    def apply_technical_relevance_scoring(
        self,
        candidate_map: Dict[str, Dict[str, Any]],
        q_info: Dict[str, Any]
    ) -> List[RetrievalResult]:
        """Calculates final technical score based on exact match signals, flags, and source quality."""
        results: List[RetrievalResult] = []
        raw_query_lower = q_info["raw_query"].lower()
        query_tokens = set(q_info["tokens"])
        query_flags = set(q_info["flags"])
        inferred_eco = q_info["inferred_ecosystem"]

        for doc_id, meta in candidate_map.items():
            record = self.index.records.get(doc_id)
            if not record:
                continue

            rrf_score = meta["rrf_score"]
            cmd_lower = record.command.lower()
            syntax_lower = record.syntax.lower()
            rec_flags = set([f.lower() for f in record.flags])

            # Signals
            exact_cmd_signal = 1.0 if (cmd_lower in raw_query_lower or raw_query_lower.startswith(cmd_lower)) else 0.0
            exact_flag_signal = 1.0 if (query_flags and query_flags.intersection(rec_flags)) else 0.0
            syntax_match_signal = 1.0 if any(t in syntax_lower for t in query_tokens if len(t) > 2) else 0.0
            ecosystem_signal = 1.0 if (inferred_eco and record.ecosystem == inferred_eco) else 0.0
            source_quality = cfg.source_quality_map.get(record.source, 0.5)

            # Combined Final Score
            final_score = (
                cfg.alpha_rrf * rrf_score +
                cfg.beta_exact_cmd * exact_cmd_signal +
                cfg.gamma_exact_flag * exact_flag_signal +
                cfg.delta_syntax_match * syntax_match_signal +
                cfg.epsilon_ecosystem_match * ecosystem_signal +
                cfg.zeta_source_quality * source_quality
            )

            signals = {
                "exact_cmd": exact_cmd_signal,
                "exact_flag": exact_flag_signal,
                "syntax_match": syntax_match_signal,
                "ecosystem_match": ecosystem_signal,
                "source_quality": source_quality
            }

            res = RetrievalResult(
                record=record,
                score=final_score,
                rrf_score=rrf_score,
                bm25_rank=meta["bm25_rank"],
                vector_rank=meta["vector_rank"],
                evidence_signals=signals
            )
            results.append(res)

        # Sort by final score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def deduplicate_records(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Deduplicates records based on stable (ecosystem, command, normalized_syntax) key while preserving metadata."""
        seen_keys: Set[str] = set()
        deduped: List[RetrievalResult] = []

        for item in results:
            norm_syntax = re.sub(r"\s+", " ", item.record.syntax).strip().lower()
            key = f"{item.record.ecosystem}:{item.record.command.lower()}:{norm_syntax}"
            if key not in seen_keys:
                seen_keys.add(key)
                deduped.append(item)
            else:
                # Merge provenance or extra metadata into existing top item if needed
                for existing in deduped:
                    ex_key = f"{existing.record.ecosystem}:{existing.record.command.lower()}:{re.sub(r's+', ' ', existing.record.syntax).strip().lower()}"
                    if ex_key == key:
                        if item.record.source not in existing.record.source:
                            existing.record.metadata[f"alias_source_{item.record.source}"] = item.record.source_location
                        break
        return deduped

    def search(self, query: str, top_k: int = cfg.top_k_final) -> Dict[str, Any]:
        """Full hybrid IRS search pipeline execution with timing telemetry."""
        start_time = time.time()

        # 1. Query Processing
        q_info = self.query_processor.process_query(query)

        # 2. Lexical BM25 Retrieval
        t_lex_start = time.time()
        lexical_candidates = self.lexical_search(q_info["tokens"], top_k=cfg.top_k_lexical)
        lexical_latency = time.time() - t_lex_start

        # 3. Semantic Retrieval
        t_sem_start = time.time()
        semantic_candidates = self.semantic_search(query, top_k=cfg.top_k_semantic)
        semantic_latency = time.time() - t_sem_start

        # 4. RRF Candidate Merging
        candidate_map = self.compute_rrf(lexical_candidates, semantic_candidates)

        # 5. Technical Evidence Scoring
        scored_results = self.apply_technical_relevance_scoring(candidate_map, q_info)

        # 6. Deduplication
        deduped_results = self.deduplicate_records(scored_results)

        # 7. Thresholding and Top-K Selection
        final_top_k = [r for r in deduped_results if r.score >= cfg.min_relevance_threshold][:top_k]

        total_latency = time.time() - start_time

        return {
            "query": query,
            "query_info": q_info,
            "lexical_count": len(lexical_candidates),
            "semantic_count": len(semantic_candidates),
            "rrf_count": len(candidate_map),
            "results": final_top_k,
            "telemetry": {
                "total_latency_ms": round(total_latency * 1000, 2),
                "lexical_latency_ms": round(lexical_latency * 1000, 2),
                "semantic_latency_ms": round(semantic_latency * 1000, 2)
            }
        }
