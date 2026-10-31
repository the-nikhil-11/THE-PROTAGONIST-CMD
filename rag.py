import json
import re
import logging
import urllib.request
from typing import List, Dict, Any, Optional, Tuple

from config import cfg
from retrieval import RetrievalResult
from sources import CommandRecord

logger = logging.getLogger("PROTAGONIST_RAG")


class GroundedRAGEngine:
    """RAG + Ollama generation layer with syntax fidelity verification and offline fallback."""

    def build_context(self, results: List[RetrievalResult]) -> str:
        """Converts retrieved records into explicit, structured context blocks."""
        if not results:
            return "NO RELEVANT COMMAND RECORDS RETRIEVED FROM CORPUS."

        blocks = []
        for idx, res in enumerate(results, start=1):
            rec = res.record
            block = (
                f"--- CANONICAL RECORD {idx} ---\n"
                f"ID: {rec.id}\n"
                f"Ecosystem: {rec.ecosystem}\n"
                f"Command: {rec.command}\n"
                f"Canonical Syntax: {rec.syntax}\n"
                f"Description: {rec.description}\n"
                f"Flags: {', '.join(rec.flags) if rec.flags else 'None'}\n"
                f"Examples: {'; '.join(rec.examples) if rec.examples else 'None'}\n"
                f"Source: {rec.source} ({rec.source_location})\n"
                f"Relevance Score: {res.score:.2f}\n"
            )
            blocks.append(block)

        return "\n".join(blocks)

    def generate_prompt(self, user_query: str, context: str) -> str:
        """Constructs the system prompt instructing Ollama to remain grounded."""
        return f"""You are the explanation and summary layer for THE PROTAGONIST CMD.
The retrieved command records below are your ONLY source of truth.

STRICT ACCURACY RULES:
1. Answer using ONLY the supplied canonical command records.
2. DO NOT invent any commands or flags that are not in the retrieved records.
3. DO NOT reconstruct or modify the canonical syntax under any circumstances.
4. When showing a command, display the EXACT 'Canonical Syntax' verbatim as given in the record.
5. Explain clearly when and how to use the retrieved command.
6. If multiple records are relevant, compare them and explain when to use each.
7. If no relevant records are provided in the evidence, state explicitly: "No sufficiently relevant command found in local corpus."

USER QUERY:
{user_query}

RETRIEVED CANONICAL EVIDENCE:
{context}

RESPONSE:"""

    def call_ollama_generate(self, prompt: str) -> Optional[str]:
        """Calls local Ollama HTTP API for text generation."""
        url = f"{cfg.ollama_host}/api/generate"
        payload = {
            "model": cfg.ollama_gen_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for strict factual adherence
                "top_p": 0.9
            }
        }
        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=req_data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=cfg.request_timeout) as resp:
                if resp.status == 200:
                    res = json.loads(resp.read().decode("utf-8"))
                    return res.get("response")
        except Exception as err:
            logger.warning(f"Ollama generation request failed ({err}).")
        return None

    def validate_response_fidelity(self, response_text: str, retrieved_records: List[CommandRecord]) -> Tuple[str, bool]:
        """Validates that commands mentioned in Ollama response match canonical syntax."""
        canonical_syntaxes = [r.syntax for r in retrieved_records]
        
        # Check if response contains code blocks
        code_blocks = re.findall(r"```(?:bash|powershell|cmd|sh|git|python)?\s*\n(.*?)\n```", response_text, re.DOTALL)
        
        hallucinated = False
        for code in code_blocks:
            code_clean = code.strip()
            # If code block does not match any canonical syntax, flag it
            if not any(code_clean == cs or code_clean in cs for cs in canonical_syntaxes):
                hallucinated = True
                break

        if hallucinated:
            logger.warning("Fidelity check: Ollama generated syntax modified from canonical record. Enforcing exact canonical syntax override.")
            # Append explicit canonical record override
            override_section = "\n\n> **Canonical Verified Command Syntax (Override):**\n"
            for rec in retrieved_records:
                override_section += f"- `{rec.syntax}` *(Source: {rec.source})*\n"
            return response_text + override_section, False

        return response_text, True

    def direct_canonical_fallback(self, results: List[RetrievalResult]) -> str:
        """Fallback response formatting when Ollama LLM service is unavailable."""
        if not results:
            return "No sufficiently relevant command found in local corpus."

        output = "### Grounded Retrieval Results (Direct Canonical Output)\n\n"
        for idx, res in enumerate(results, start=1):
            rec = res.record
            output += (
                f"**{idx}. [{rec.ecosystem.upper()}] `{rec.syntax}`**\n"
                f"- **Command:** `{rec.command}`\n"
                f"- **Description:** {rec.description}\n"
                f"- **Flags:** `{', '.join(rec.flags) if rec.flags else 'None'}`\n"
                f"- **Source:** {rec.source}\n"
                f"- **Relevance Score:** {res.score:.2f}\n\n"
            )
        return output

    def generate_grounded_response(self, user_query: str, retrieval_output: Dict[str, Any]) -> Dict[str, Any]:
        """Executes full RAG workflow: context construction -> Ollama call -> validation -> fallback."""
        results: List[RetrievalResult] = retrieval_output.get("results", [])

        if not results:
            return {
                "grounded_text": "No sufficiently relevant command found in local corpus.",
                "llm_used": False,
                "validated": True,
                "context": ""
            }

        context = self.build_context(results)
        prompt = self.generate_prompt(user_query, context)

        # Attempt Ollama LLM Generation
        raw_response = self.call_ollama_generate(prompt)

        if raw_response:
            final_text, validated = self.validate_response_fidelity(raw_response, [r.record for r in results])
            return {
                "grounded_text": final_text,
                "llm_used": True,
                "validated": validated,
                "context": context
            }

        # Fallback to Direct Canonical Output if Ollama is unavailable
        fallback_text = self.direct_canonical_fallback(results)
        return {
            "grounded_text": fallback_text,
            "llm_used": False,
            "validated": True,
            "context": context
        }
