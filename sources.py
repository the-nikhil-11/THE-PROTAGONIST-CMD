import os
import re
import json
import zipfile
import hashlib
import urllib.request
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path

from config import cfg

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("PROTAGONIST_SOURCES")


@dataclass
class CommandRecord:
    id: str
    ecosystem: str
    command: str
    syntax: str
    description: str
    arguments: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    source: str = ""
    source_location: str = ""
    version: str = "latest"
    platform: str = "common"
    search_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            raw_id = f"{self.ecosystem}:{self.command}:{self.syntax}"
            self.id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]
        if not self.search_text:
            self.search_text = self.build_search_text()

    def build_search_text(self) -> str:
        parts = [
            self.command,
            self.syntax,
            self.description,
            " ".join(self.flags),
            " ".join(self.aliases),
            " ".join(self.examples),
            self.ecosystem,
            self.platform
        ]
        return " ".join(filter(None, parts)).lower()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CommandRecord":
        return cls(**data)


def parse_tldr_markdown(content: str, filename: str, source_name: str) -> List[CommandRecord]:
    """Parses a TLDR page markdown file into CommandRecord instances."""
    records = []
    lines = content.splitlines()
    command_name = ""
    description_lines = []
    examples = []
    
    # Platform usually determined by file folder path e.g. pages/windows/dir.md
    platform = "common"
    norm_path = filename.replace("\\", "/").lower()
    if "/windows/" in norm_path or "/cmd/" in norm_path:
        platform = "windows"
    elif "/linux/" in norm_path:
        platform = "linux"
    elif "/osx/" in norm_path or "/mac/" in norm_path:
        platform = "osx"

    current_example_desc = ""

    for line in lines:
        line_str = line.strip()
        if line_str.startswith("# "):
            command_name = line_str[2:].strip()
        elif line_str.startswith("> "):
            description_lines.append(line_str[2:].strip())
        elif line_str.startswith("- "):
            current_example_desc = line_str[2:].strip()
        elif line_str.startswith("`") and line_str.endswith("`"):
            syntax_code = line_str[1:-1].strip()
            if syntax_code:
                examples.append((current_example_desc, syntax_code))
                current_example_desc = ""

    if not command_name:
        command_name = Path(filename).stem

    desc = " ".join(description_lines) if description_lines else command_name
    
    # Ecosystem classification
    ecosystem = "cli"
    cmd_lower = command_name.lower()
    if cmd_lower.startswith("git") or "git" in norm_path:
        ecosystem = "git"
    elif cmd_lower.startswith("powershell") or cmd_lower.startswith("get-") or cmd_lower.startswith("set-") or platform == "windows":
        ecosystem = "powershell" if cmd_lower.startswith(("get-", "set-", "new-", "remove-")) else "windows_cmd"
    elif cmd_lower.startswith("python") or cmd_lower.startswith("pip"):
        ecosystem = "python"
    elif platform in ["linux", "osx"]:
        ecosystem = "linux"

    for idx, (ex_desc, ex_syntax) in enumerate(examples):
        # Extract flags from syntax (e.g. -m, --message, -a)
        flags = re.findall(r"(-\w|--[\w-]+)", ex_syntax)
        
        record_id = hashlib.sha256(f"{source_name}:{filename}:{idx}:{ex_syntax}".encode("utf-8")).hexdigest()[:16]
        rec = CommandRecord(
            id=record_id,
            ecosystem=ecosystem,
            command=command_name,
            syntax=ex_syntax,
            description=f"{desc} ({ex_desc})" if ex_desc else desc,
            flags=list(set(flags)),
            examples=[f"{ex_desc}: {ex_syntax}"] if ex_desc else [ex_syntax],
            source=source_name,
            source_location=filename,
            platform=platform
        )
        records.append(rec)

    # If no examples parsed, create a base record
    if not records and command_name:
        rec = CommandRecord(
            id=hashlib.sha256(f"{source_name}:{filename}:base".encode("utf-8")).hexdigest()[:16],
            ecosystem=ecosystem,
            command=command_name,
            syntax=command_name,
            description=desc,
            source=source_name,
            source_location=filename,
            platform=platform
        )
        records.append(rec)

    return records


def get_builtin_seed_records() -> List[CommandRecord]:
    """Comprehensive, authoritative built-in dataset covering major ecosystems."""
    seeds = [
        # Git Ecosystem
        {
            "ecosystem": "git", "command": "git commit",
            "syntax": "git commit -m \"<message>\"",
            "description": "Record changes to the repository with a specified commit message.",
            "flags": ["-m", "--message", "-a", "--all", "--amend"],
            "examples": ["git commit -m \"Fix authentication bug\"", "git commit -a -m \"Update all tracked files\""],
            "platform": "common"
        },
        {
            "ecosystem": "git", "command": "git checkout",
            "syntax": "git checkout -b <branch_name>",
            "description": "Create a new local branch and switch to it immediately.",
            "flags": ["-b", "-B", "--detach", "-f", "--force"],
            "examples": ["git checkout -b feature/login-page", "git checkout main"],
            "platform": "common"
        },
        {
            "ecosystem": "git", "command": "git switch",
            "syntax": "git switch -c <branch_name>",
            "description": "Switch branches or create a new branch in Git.",
            "flags": ["-c", "--create", "-d", "--detach"],
            "examples": ["git switch -c feature/new-engine", "git switch main"],
            "platform": "common"
        },
        {
            "ecosystem": "git", "command": "git rebase",
            "syntax": "git rebase -i <base_branch>",
            "description": "Reapply commits on top of another base tip interactively.",
            "flags": ["-i", "--interactive", "--onto", "--continue", "--abort"],
            "examples": ["git rebase -i main", "git rebase --continue"],
            "platform": "common"
        },
        {
            "ecosystem": "git", "command": "git status",
            "syntax": "git status -s",
            "description": "Show the working tree status in short format.",
            "flags": ["-s", "--short", "-b", "--branch", "--ignored"],
            "examples": ["git status", "git status -sb"],
            "platform": "common"
        },
        {
            "ecosystem": "git", "command": "git push",
            "syntax": "git push -u origin <branch_name>",
            "description": "Update remote refs along with associated objects and set upstream tracking.",
            "flags": ["-u", "--set-upstream", "-f", "--force", "--tags"],
            "examples": ["git push -u origin main", "git push --force-with-lease"],
            "platform": "common"
        },
        {
            "ecosystem": "git", "command": "git log",
            "syntax": "git log --oneline --graph --all",
            "description": "Display commit logs in a concise one-line graph format across all branches.",
            "flags": ["--oneline", "--graph", "--all", "-n", "--author"],
            "examples": ["git log --oneline -n 10", "git log --graph --all"],
            "platform": "common"
        },
        {
            "ecosystem": "git", "command": "git stash",
            "syntax": "git stash save \"<message>\"",
            "description": "Stash the changes in a dirty working directory away.",
            "flags": ["save", "pop", "apply", "list", "drop", "-u", "--include-untracked"],
            "examples": ["git stash -u", "git stash pop"],
            "platform": "common"
        },
        {
            "ecosystem": "git", "command": "git reset",
            "syntax": "git reset --hard HEAD~1",
            "description": "Reset current HEAD to specified state, discarding uncommitted changes.",
            "flags": ["--hard", "--soft", "--mixed", "HEAD~1"],
            "examples": ["git reset --hard HEAD", "git reset --soft HEAD~1"],
            "platform": "common"
        },

        # PowerShell Ecosystem
        {
            "ecosystem": "powershell", "command": "Get-Process",
            "syntax": "Get-Process -Name <process_name> | Stop-Process -Force",
            "description": "Gets processes running on local or remote computer and pipes to stop them forcibly.",
            "flags": ["-Name", "-Id", "-ComputerName", "-FileVersionInfo"],
            "examples": ["Get-Process -Name chrome", "Get-Process node | Stop-Process -Force"],
            "platform": "windows"
        },
        {
            "ecosystem": "powershell", "command": "Get-ChildItem",
            "syntax": "Get-ChildItem -Path <path> -Recurse -Filter *.py",
            "description": "Gets items and child items in one or more specified locations matching a filter.",
            "flags": ["-Path", "-Recurse", "-Filter", "-File", "-Directory", "-Hidden"],
            "examples": ["Get-ChildItem -Path C:\\Logs -Recurse", "ls -Recurse *.json"],
            "platform": "windows"
        },
        {
            "ecosystem": "powershell", "command": "Set-ExecutionPolicy",
            "syntax": "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser",
            "description": "Changes user preferences for PowerShell execution policy.",
            "flags": ["-ExecutionPolicy", "-Scope", "-Force"],
            "examples": ["Set-ExecutionPolicy RemoteSigned -Scope CurrentUser"],
            "platform": "windows"
        },
        {
            "ecosystem": "powershell", "command": "Test-NetConnection",
            "syntax": "Test-NetConnection -ComputerName <hostname> -Port <port>",
            "description": "Displays diagnostic info for a connection including TCP port connection tests.",
            "flags": ["-ComputerName", "-Port", "-TraceRoute", "-InformationLevel"],
            "examples": ["Test-NetConnection -ComputerName 127.0.0.1 -Port 11434"],
            "platform": "windows"
        },
        {
            "ecosystem": "powershell", "command": "Get-Service",
            "syntax": "Get-Service -Name <service_name>",
            "description": "Gets services on local or remote computer.",
            "flags": ["-Name", "-DisplayName", "-DependentServices", "-RequiredServices"],
            "examples": ["Get-Service -Name wuauserv", "Get-Service | Where-Object Status -eq Running"],
            "platform": "windows"
        },

        # Windows CMD Ecosystem
        {
            "ecosystem": "windows_cmd", "command": "dir",
            "syntax": "dir /s /b /a <path>",
            "description": "Displays a list of files and subdirectories in a directory with options for bare and recursive output.",
            "flags": ["/s", "/b", "/a", "/p", "/w", "/o:d"],
            "examples": ["dir /s /b *.txt", "dir /a:h"],
            "platform": "windows"
        },
        {
            "ecosystem": "windows_cmd", "command": "netstat",
            "syntax": "netstat -ano | findstr :<port>",
            "description": "Displays active TCP connections, listening ports, and associated Process IDs (PID).",
            "flags": ["-a", "-n", "-o", "-b", "-r"],
            "examples": ["netstat -ano | findstr :11434", "netstat -an"],
            "platform": "windows"
        },
        {
            "ecosystem": "windows_cmd", "command": "taskkill",
            "syntax": "taskkill /F /PID <pid>",
            "description": "Terminates tasks by process id or image name forcibly.",
            "flags": ["/F", "/PID", "/IM", "/T"],
            "examples": ["taskkill /F /PID 1234", "taskkill /F /IM node.exe"],
            "platform": "windows"
        },
        {
            "ecosystem": "windows_cmd", "command": "robocopy",
            "syntax": "robocopy <source> <destination> /MIR /MT:8",
            "description": "Robust file and folder copy utility for Windows with mirroring and multithreading.",
            "flags": ["/MIR", "/E", "/MT", "/LOG", "/Z", "/R:3", "/W:5"],
            "examples": ["robocopy C:\\src C:\\dst /MIR /MT:8"],
            "platform": "windows"
        },

        # Python Ecosystem
        {
            "ecosystem": "python", "command": "python -m http.server",
            "syntax": "python -m http.server <port> --directory <path>",
            "description": "Start a simple HTTP web server serving files from directory.",
            "flags": ["-m", "--directory", "--bind"],
            "examples": ["python -m http.server 8000", "python -m http.server 8080 --directory ./public"],
            "platform": "common"
        },
        {
            "ecosystem": "python", "command": "python -m venv",
            "syntax": "python -m venv <env_name>",
            "description": "Create lightweight virtual environments with their own site directories.",
            "flags": ["-m", "--system-site-packages", "--clear", "--upgrade-deps"],
            "examples": ["python -m venv .venv", "python -m venv --clear venv"],
            "platform": "common"
        },
        {
            "ecosystem": "python", "command": "pip install",
            "syntax": "pip install -r requirements.txt --no-cache-dir",
            "description": "Install Python packages from requirements file or PyPI without cache.",
            "flags": ["-r", "--no-cache-dir", "-U", "--upgrade", "-e", "--editable"],
            "examples": ["pip install -r requirements.txt", "pip install --upgrade numpy"],
            "platform": "common"
        },
        {
            "ecosystem": "python", "command": "python -m unittest",
            "syntax": "python -m unittest discover -s <directory> -p \"test_*.py\"",
            "description": "Run python unit test discovery in a directory matching pattern.",
            "flags": ["-m", "discover", "-s", "-p", "-v", "--verbose"],
            "examples": ["python -m unittest discover", "python -m unittest tests/test_core.py"],
            "platform": "common"
        },

        # Linux / Unix Ecosystem
        {
            "ecosystem": "linux", "command": "find",
            "syntax": "find <path> -name \"<pattern>\" -type f -exec <command> {} +",
            "description": "Search for files in directory hierarchy matching pattern and execute command.",
            "flags": ["-name", "-type", "-mtime", "-size", "-exec", "-delete"],
            "examples": ["find . -name \"*.log\" -type f -delete", "find /var/log -mtime -7"],
            "platform": "linux"
        },
        {
            "ecosystem": "linux", "command": "grep",
            "syntax": "grep -rnEI \"<pattern>\" <path>",
            "description": "Search recursively in path for regular expression pattern ignoring binary files.",
            "flags": ["-r", "-n", "-E", "-I", "-i", "-v", "-w", "-l"],
            "examples": ["grep -rnEI \"ERROR\" /var/log", "grep -i \"failed\" app.log"],
            "platform": "linux"
        },
        {
            "ecosystem": "linux", "command": "chmod",
            "syntax": "chmod -R 755 <path>",
            "description": "Change file permissions mode bits recursively.",
            "flags": ["-R", "--recursive", "+x", "755", "644"],
            "examples": ["chmod +x script.sh", "chmod -R 755 /var/www"],
            "platform": "linux"
        },
        {
            "ecosystem": "linux", "command": "tar",
            "syntax": "tar -czvf <archive.tar.gz> <directory>",
            "description": "Create gzipped tar archive file from directory.",
            "flags": ["-czvf", "-xzvf", "-tjvf", "-C", "--exclude"],
            "examples": ["tar -czvf backup.tar.gz ./data", "tar -xzvf archive.tar.gz -C /opt"],
            "platform": "linux"
        },
        {
            "ecosystem": "linux", "command": "journalctl",
            "syntax": "journalctl -u <service_name> -f --since \"1 hour ago\"",
            "description": "Query systemd journal logs for a specific service in real time.",
            "flags": ["-u", "-f", "-n", "--since", "-b", "-p"],
            "examples": ["journalctl -u nginx -f", "journalctl -p err -b"],
            "platform": "linux"
        }
    ]

    records = []
    for s in seeds:
        rec = CommandRecord(
            id=hashlib.sha256(f"builtin:{s['ecosystem']}:{s['syntax']}".encode("utf-8")).hexdigest()[:16],
            ecosystem=s["ecosystem"],
            command=s["command"],
            syntax=s["syntax"],
            description=s["description"],
            flags=s.get("flags", []),
            examples=s.get("examples", []),
            source="builtin_seed",
            source_location="internal",
            platform=s.get("platform", "common")
        )
        records.append(rec)
    return records


# Acquisition Manager implementing the mandatory Source Fallback System
class SourceAcquisitionEngine:
    """Manages multi-source acquisition using Local, HTTP, and Python Library methods with fallback."""

    def acquire_all(self) -> List[CommandRecord]:
        all_records: List[CommandRecord] = []
        # Always load built-in seed first to guarantee immediate functionality
        builtin_records = get_builtin_seed_records()
        all_records.extend(builtin_records)
        logger.info(f"Loaded {len(builtin_records)} built-in seed command records.")

        # Sort sources by priority ascending (priority 1 highest)
        sorted_sources = sorted(cfg.sources, key=lambda x: x.get("priority", 99))

        for source_cfg in sorted_sources:
            sname = source_cfg["name"]
            smethod = source_cfg["method"]
            if sname == "builtin_seed":
                continue

            logger.info(f"Attempting source acquisition: '{sname}' via '{smethod}'...")
            try:
                records = self._acquire_source(source_cfg)
                if records:
                    all_records.extend(records)
                    logger.info(f"Successfully acquired {len(records)} records from '{sname}'.")
                else:
                    logger.warning(f"Source '{sname}' returned 0 records. Triggering fallback.")
            except Exception as exc:
                logger.error(f"Source acquisition failed for '{sname}': {exc}. Continuing to fallback sources.")
                continue

        return all_records

    def _acquire_source(self, source_cfg: Dict[str, Any]) -> List[CommandRecord]:
        smethod = source_cfg["method"]
        sname = source_cfg["name"]

        if smethod == "local_repository":
            return self._acquire_local_repo(source_cfg)
        elif smethod == "http_api":
            return self._acquire_http_api(source_cfg)
        elif smethod == "python_library":
            return self._acquire_python_lib(source_cfg)
        else:
            raise ValueError(f"Unknown acquisition method '{smethod}'")

    def _acquire_local_repo(self, source_cfg: Dict[str, Any]) -> List[CommandRecord]:
        sname = source_cfg["name"]
        url = source_cfg.get("url")
        local_path = Path(source_cfg.get("local_path", ""))

        # If zip does not exist locally, download it
        if url and not local_path.exists():
            logger.info(f"Downloading repository archive for '{sname}' from {url}...")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "ProtagonistCMD-VO7/1.0"})
                with urllib.request.urlopen(req, timeout=source_cfg.get("timeout", 30)) as resp, open(local_path, "wb") as f:
                    f.write(resp.read())
                logger.info(f"Saved archive to {local_path}")
            except Exception as err:
                raise RuntimeError(f"Download failed for '{sname}': {err}")

        if not local_path.exists():
            raise FileNotFoundError(f"Local resource path for '{sname}' not found: {local_path}")

        records = []
        if zipfile.is_zipfile(local_path):
            with zipfile.ZipFile(local_path, "r") as zf:
                for name in zf.namelist():
                    if name.endswith(".md") and ("pages/" in name or "reference/" in name or "docs/" in name):
                        try:
                            content = zf.read(name).decode("utf-8", errors="ignore")
                            recs = parse_tldr_markdown(content, name, sname)
                            records.extend(recs)
                        except Exception as e:
                            continue
        return records

    def _acquire_http_api(self, source_cfg: Dict[str, Any]) -> List[CommandRecord]:
        # HTTP fetch example endpoints from cheat.sh or docs
        sname = source_cfg["name"]
        base_url = source_cfg["url"]
        common_topics = ["git", "tar", "find", "grep", "powershell", "curl", "ssh", "docker", "python"]
        records = []

        for topic in common_topics:
            target_url = f"{base_url}/{topic}?q"
            try:
                req = urllib.request.Request(target_url, headers={"User-Agent": "curl/7.68.0"})
                with urllib.request.urlopen(req, timeout=source_cfg.get("timeout", 5)) as resp:
                    if resp.status == 200:
                        text = resp.read().decode("utf-8", errors="ignore")
                        # Basic cheat.sh parsing
                        lines = [l for l in text.splitlines() if l.strip() and not l.startswith("#")]
                        if lines:
                            syntax = lines[0].strip()
                            rec = CommandRecord(
                                id=hashlib.sha256(f"{sname}:{topic}:{syntax}".encode("utf-8")).hexdigest()[:16],
                                ecosystem="cli",
                                command=topic,
                                syntax=syntax,
                                description=f"Cheat sheet entry for {topic}",
                                source=sname,
                                source_location=target_url
                            )
                            records.append(rec)
            except Exception:
                continue
        return records

    def _acquire_python_lib(self, source_cfg: Dict[str, Any]) -> List[CommandRecord]:
        # Python library inspection for CLI standard library tools
        py_modules = [
            ("http.server", "python -m http.server", "HTTP server module"),
            ("json.tool", "python -m json.tool", "JSON formatting CLI utility"),
            ("venv", "python -m venv", "Virtual environment creation module"),
            ("zipfile", "python -m zipfile -l <archive.zip>", "Zip archive CLI inspector"),
            ("tarfile", "python -m tarfile -l <archive.tar>", "Tar archive CLI inspector"),
            ("pydoc", "python -m pydoc <name>", "Python documentation generator and viewer"),
            ("timeit", "python -m timeit \"<code>\"", "Measure execution time of small code snippets"),
            ("trace", "python -m trace --trace <script.py>", "Trace Python program execution")
        ]
        records = []
        for mod, syntax, desc in py_modules:
            rec = CommandRecord(
                id=hashlib.sha256(f"python_lib:{mod}".encode("utf-8")).hexdigest()[:16],
                ecosystem="python",
                command=f"python -m {mod}",
                syntax=syntax,
                description=desc,
                flags=["-m"],
                source="official_python_docs",
                source_location=f"std_lib:{mod}"
            )
            records.append(rec)
        return records


def load_canonical_corpus() -> List[CommandRecord]:
    """Loads records from persistent JSON file if exists, or executes full acquisition."""
    if cfg.corpus_file.exists():
        try:
            with open(cfg.corpus_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            raw_list = raw_data.values() if isinstance(raw_data, dict) else raw_data
            records = [CommandRecord.from_dict(d) for d in raw_list]
            logger.info(f"Loaded {len(records)} command records from persistent corpus file {cfg.corpus_file}")
            return records
        except Exception as err:
            logger.warning(f"Failed to load corpus file ({err}). Triggering fresh acquisition.")

    engine = SourceAcquisitionEngine()
    records = engine.acquire_all()
    
    # Save to persistent file
    try:
        with open(cfg.corpus_file, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in records], f, indent=2)
        logger.info(f"Saved {len(records)} command records to corpus file {cfg.corpus_file}")
    except Exception as err:
        logger.error(f"Failed to persist corpus file: {err}")

    return records
