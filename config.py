import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any

# Root directory of the project
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"
INDEX_DIR = BASE_DIR / "index_data"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class Config:
    # System identification
    project_name: str = "THE PROTAGONIST CMD"
    version: str = "VO-7"

    # Ollama Settings
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_gen_model: str = os.getenv("OLLAMA_GEN_MODEL", "qwen2.5-coder:7b")
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")

    # Timeouts
    request_timeout: int = 15

    # Index Persistence Paths
    corpus_file: Path = INDEX_DIR / "corpus.json"
    inverted_index_file: Path = INDEX_DIR / "inverted_index.json"
    bm25_stats_file: Path = INDEX_DIR / "bm25_stats.json"
    vector_index_file: Path = INDEX_DIR / "vectors.npy"
    vector_meta_file: Path = INDEX_DIR / "vector_meta.json"

    # BM25 Parameters
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    field_weights: Dict[str, float] = field(default_factory=lambda: {
        "command": 3.5,
        "syntax": 4.0,
        "flags": 2.5,
        "aliases": 2.0,
        "description": 1.0,
        "examples": 1.5,
        "ecosystem": 1.5
    })

    # RRF Parameters
    rrf_k: int = 60

    # Final Technical Score Coefficients
    alpha_rrf: float = 1.0
    beta_exact_cmd: float = 3.0
    gamma_exact_flag: float = 2.0
    delta_syntax_match: float = 2.5
    epsilon_ecosystem_match: float = 1.5
    zeta_source_quality: float = 0.5

    # Retrieval Limits & Thresholds
    top_k_lexical: int = 20
    top_k_semantic: int = 20
    top_k_final: int = 5
    min_relevance_threshold: float = 0.05

    # Source Priority Map (higher score = more authoritative)
    source_quality_map: Dict[str, float] = field(default_factory=lambda: {
        "official_git_docs": 1.0,
        "official_powershell_docs": 1.0,
        "official_python_docs": 1.0,
        "tldr_pages": 0.85,
        "cheat_sh": 0.75,
        "man_pages": 0.80,
        "builtin_seed": 0.90
    })

    # Data Source Configurations (Priority-ordered)
    sources: List[Dict[str, Any]] = field(default_factory=lambda: [
        {
            "name": "builtin_seed",
            "method": "python_library",
            "priority": 1,
            "timeout": 5,
        },
        {
            "name": "tldr_pages",
            "method": "local_repository",
            "url": "https://github.com/tldr-pages/tldr/archive/refs/heads/main.zip",
            "local_path": str(DATA_DIR / "tldr-main.zip"),
            "priority": 2,
            "timeout": 30,
        },
        {
            "name": "cheat_sh",
            "method": "http_api",
            "url": "https://cheat.sh",
            "priority": 3,
            "timeout": 10,
        },
        {
            "name": "official_powershell_docs",
            "method": "local_repository",
            "url": "https://github.com/PowerShell/PowerShell-Docs/archive/refs/heads/main.zip",
            "local_path": str(DATA_DIR / "powershell-docs.zip"),
            "priority": 4,
            "timeout": 30,
        },
        {
            "name": "official_python_docs",
            "method": "python_library",
            "priority": 5,
            "timeout": 10,
        }
    ])

    # Observability / Debug Mode
    debug: bool = False

cfg = Config()
