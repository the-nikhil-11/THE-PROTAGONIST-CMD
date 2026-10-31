import sys
import os
import argparse
import time
import logging

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from config import cfg
from index import build_or_load_indexes
from retrieval import InformationRetrievalEngine
from rag import GroundedRAGEngine

console = Console()
logging.getLogger("PROTAGONIST_INDEX").setLevel(logging.WARNING)
logging.getLogger("PROTAGONIST_SOURCES").setLevel(logging.WARNING)
logging.getLogger("PROTAGONIST_RETRIEVAL").setLevel(logging.WARNING)
logging.getLogger("PROTAGONIST_RAG").setLevel(logging.WARNING)


def print_banner():
    banner_text = Text()
    banner_text.append("=== ", style="bold cyan")
    banner_text.append(f"{cfg.project_name} — {cfg.version}", style="bold white on blue")
    banner_text.append(" ===", style="bold cyan")
    banner_text.append("\nLocal RAG + Information Retrieval Command Engine\n", style="italic bright_black")
    banner_text.append("Ecosystems: Windows CMD | PowerShell | Git | Python | Linux Shell", style="dim")
    console.print(Panel(banner_text, border_style="cyan"))


def display_debug_telemetry(retrieval_output: dict, rag_output: dict):
    """Displays internal search and RAG telemetry in debug mode."""
    table = Table(title="[PROTAGONIST OBSERVABILITY & TELEMETRY]", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")

    telemetry = retrieval_output.get("telemetry", {})
    table.add_row("Total IRS Latency", f"{telemetry.get('total_latency_ms')} ms")
    table.add_row("Lexical Search Latency", f"{telemetry.get('lexical_latency_ms')} ms")
    table.add_row("Semantic Search Latency", f"{telemetry.get('semantic_latency_ms')} ms")
    table.add_row("BM25 Candidate Count", str(retrieval_output.get("lexical_count", 0)))
    table.add_row("Vector Candidate Count", str(retrieval_output.get("semantic_count", 0)))
    table.add_row("RRF Fusion Count", str(retrieval_output.get("rrf_count", 0)))
    table.add_row("Final Top-K Selected", str(len(retrieval_output.get("results", []))))
    table.add_row("LLM Generation Used", str(rag_output.get("llm_used", False)))
    table.add_row("Fidelity Validated", str(rag_output.get("validated", True)))

    console.print(table)


def run_query(query: str, irs: InformationRetrievalEngine, rag: GroundedRAGEngine, top_k: int, debug: bool):
    """Executes search and grounded generation for a single query."""
    if not query.strip():
        return

    console.print(f"\n[bold green]?[/bold green] [bold white]Query:[/bold white] [italic]{query}[/italic]")
    
    with console.status("[bold cyan]Retrieving canonical records & generating grounded response...[/bold cyan]", spinner="dots"):
        # 1. IRS Retrieval
        retrieval_out = irs.search(query, top_k=top_k)
        
        # 2. RAG Generation
        rag_out = rag.generate_grounded_response(query, retrieval_out)

    # Print Final Grounded Response
    grounded_md = Markdown(rag_out["grounded_text"])
    console.print(Panel(grounded_md, title="[bold cyan]GROUNDED COMMAND RESPONSE[/bold cyan]", border_style="green"))

    # Print Debug Telemetry if enabled
    if debug or cfg.debug:
        display_debug_telemetry(retrieval_out, rag_out)


def interactive_repl(irs: InformationRetrievalEngine, rag: GroundedRAGEngine, top_k: int, debug: bool):
    """Interactive command-line shell REPL interface."""
    print_banner()
    console.print("[dim]Type your command query in plain English. Type '/help' for commands or '/exit' to quit.[/dim]\n")

    current_debug = debug

    while True:
        try:
            user_input = console.input("[bold cyan]THE PROTAGONIST CMD>[/bold cyan] ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["/exit", "/quit", "exit", "quit"]:
                console.print("[bold yellow]Exiting THE PROTAGONIST CMD. Good luck![/bold yellow]")
                break
            elif user_input.lower() == "/help":
                console.print(Panel(
                    "[bold]Available Commands:[/bold]\n"
                    "  [cyan]/debug[/cyan]     - Toggle telemetry and latency breakdown\n"
                    "  [cyan]/stats[/cyan]     - View index & corpus statistics\n"
                    "  [cyan]/rebuild[/cyan]   - Force rebuild of offline indexes\n"
                    "  [cyan]/help[/cyan]      - Display this help message\n"
                    "  [cyan]/exit[/cyan]      - Exit application",
                    title="Help Menu", border_style="cyan"
                ))
                continue
            elif user_input.lower() == "/debug":
                current_debug = not current_debug
                console.print(f"[bold yellow]Debug observability mode set to: {current_debug}[/bold yellow]")
                continue
            elif user_input.lower() == "/stats":
                console.print(Panel(
                    f"Corpus Records: {irs.index.N}\n"
                    f"Inverted Vocabulary: {len(irs.index.inverted_index)} tokens\n"
                    f"Average Doc Length: {irs.index.avg_dl:.2f}\n"
                    f"Vector Index Status: {'Active (' + str(irs.index.vector_matrix.shape) + ')' if irs.index.vector_matrix is not None else 'Inactive/Offline'}\n"
                    f"Ollama Host: {cfg.ollama_host}\n"
                    f"Ollama Gen Model: {cfg.ollama_gen_model}\n"
                    f"Ollama Embed Model: {cfg.ollama_embed_model}",
                    title="Corpus & Index Statistics", border_style="blue"
                ))
                continue
            elif user_input.lower() == "/rebuild":
                console.print("[bold yellow]Rebuilding offline indexes from sources...[/bold yellow]")
                new_index = build_or_load_indexes(force_rebuild=True)
                irs.index = new_index
                console.print("[bold green]Index rebuild complete![/bold green]")
                continue

            run_query(user_input, irs, rag, top_k, current_debug)

        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold yellow]Exiting THE PROTAGONIST CMD.[/bold yellow]")
            break


def main():
    parser = argparse.ArgumentParser(description="THE PROTAGONIST CMD — VO-7 RAG + Information Retrieval Engine")
    parser.add_argument("query", nargs="*", help="Natural language command query (optional, enters REPL if omitted)")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild of offline indexes from all available sources")
    parser.add_argument("--top-k", type=int, default=cfg.top_k_final, help="Number of top retrieved records to return (default: 5)")
    parser.add_argument("--debug", action="store_true", help="Enable observability telemetry and search latency breakdown")

    args = parser.parse_args()

    # System Initialization
    with console.status("[bold cyan]Initializing THE PROTAGONIST CMD Engine & Loading Indexes...[/bold cyan]", spinner="bouncingBar"):
        index_engine = build_or_load_indexes(force_rebuild=args.rebuild_index)
        irs = InformationRetrievalEngine(index_engine)
        rag = GroundedRAGEngine()

    if args.query:
        query_str = " ".join(args.query)
        run_query(query_str, irs, rag, args.top_k, args.debug)
    else:
        interactive_repl(irs, rag, args.top_k, args.debug)


if __name__ == "__main__":
    main()
