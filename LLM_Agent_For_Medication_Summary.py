"""
ai_drug_summary.py

Refactored, production-ready Python module based on the uploaded notebook.
- Production concerns: config, logging, schema validation, error handling, extensibility
- Provider-agnostic LLM client interface (no provider keys or secrets in code)

Reference notebook (uploaded by user):
/mnt/data/extracted/40b1cb422539ebeb763f00f147627448-4cab1b2715c5d17a95b456cfcbc13551ac026fcd/copy-of-ai_powered_drug_summary.ipynb

This file is intended as a starting point for production deployment. Replace the LLM client implementation
with your provider-specific implementation and securely inject credentials via environment variables or a
secrets manager.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import pandas as pd
from pydantic import BaseModel, Field, root_validator, validator


# -----------------
# Logging & Config
# -----------------

LOG = logging.getLogger("ai_drug_summary")


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    fmt = "%(asctime)s %(levelname)s %(name)s - %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    LOG.addHandler(handler)
    LOG.setLevel(level)


class AppConfig(BaseModel):
    input_path: Path = Field(..., description="Path to input CSV file")
    output_path: Path = Field(default=Path("./drug_summary.json"))
    llm_provider: Optional[str] = Field(None, description="LLM provider identifier (for injection)")
    temperature: float = Field(0.0, ge=0.0, le=1.0)

    @validator("input_path")
    def input_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"input_path does not exist: {v}")
        return v


# -----------------
# Data Handling
# -----------------

@dataclass
class MedicationRecord:
    name: str
    dose: Optional[str] = None
    frequency: Optional[str] = None
    last_refill: Optional[datetime.date] = None
    provider: Optional[str] = None
    notes: Optional[str] = None


class DataLoader:
    """Load CSV into a normalized pandas DataFrame and provide safe accessors."""

    def __init__(self, path: Path):
        self.path = path
        self.df: Optional[pd.DataFrame] = None

    def load(self, **kwargs: Any) -> pd.DataFrame:
        LOG.info("Loading CSV from %s", self.path)
        # robust read with fallback encodings
        try:
            self.df = pd.read_csv(self.path, **kwargs)
        except UnicodeDecodeError:
            self.df = pd.read_csv(self.path, encoding="latin-1", **kwargs)

        LOG.debug("Loaded dataframe shape=%s", self.df.shape)
        return self.df

    def infer_columns(self) -> Dict[str, str]:
        """Attempt to map columns to canonical names.
        This is intentionally heuristic and conservative.
        """
        if self.df is None:
            raise RuntimeError("DataFrame is not loaded")

        cols = {c.lower(): c for c in self.df.columns}
        mapping: Dict[str, str] = {}

        def find_like(keywords: List[str]) -> Optional[str]:
            for k, orig in cols.items():
                for kw in keywords:
                    if kw in k:
                        return orig
            return None

        mapping["name"] = find_like(["drug", "med", "medication", "name"]) or list(self.df.columns)[0]
        mapping["dose"] = find_like(["dose", "dosage", "strength"]) or ""
        mapping["frequency"] = find_like(["freq", "frequency", "schedule", "howoften"]) or ""
        mapping["last_refill"] = find_like(["refill", "last_refill", "lastrefill", "refill_date"]) or ""
        mapping["provider"] = find_like(["provider", "prescriber", "doctor"]) or ""
        mapping["notes"] = find_like(["note", "notes", "comment"]) or ""

        LOG.debug("Inferred column mapping: %s", mapping)
        return mapping

    def to_med_records(self) -> List[MedicationRecord]:
        if self.df is None:
            raise RuntimeError("DataFrame is not loaded")

        mapping = self.infer_columns()
        recs: List[MedicationRecord] = []

        for _, row in self.df.iterrows():
            def get(col):
                colname = mapping.get(col)
                if not colname:
                    return None
                val = row.get(colname)
                if pd.isna(val):
                    return None
                return val

            last_refill_raw = get("last_refill")
            last_refill = self._parse_date(last_refill_raw)

            rec = MedicationRecord(
                name=str(get("name")) if get("name") is not None else "",
                dose=str(get("dose")) if get("dose") is not None else None,
                frequency=str(get("frequency")) if get("frequency") is not None else None,
                last_refill=last_refill,
                provider=str(get("provider")) if get("provider") is not None else None,
                notes=str(get("notes")) if get("notes") is not None else None,
            )
            recs.append(rec)

        LOG.info("Converted %d rows into MedicationRecord objects", len(recs))
        return recs

    @staticmethod
    def _parse_date(value: Any) -> Optional[datetime.date]:
        if value is None:
            return None
        if isinstance(value, datetime.date):
            return value
        if isinstance(value, str):
            value = value.strip()
            for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%b %d, %Y"):
                try:
                    return datetime.datetime.strptime(value, fmt).date()
                except Exception:
                    continue
        try:
            return pd.to_datetime(value).date()
        except Exception:
            LOG.debug("Unable to parse date: %s", value)
            return None


# -----------------
# LLM Abstraction
# -----------------

class LLMClientProtocol(Protocol):
    """Define the interface we expect from any LLM client implementation."""

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        ...


class DummyLLMClient:
    """A lightweight, deterministic stub used for testing and local runs.
    Replace with your provider-specific client in production (secure secrets).
    """

    def __init__(self, provider: Optional[str] = None):
        self.provider = provider

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        LOG.debug("DummyLLMClient.generate called with temperature=%s", temperature)
        # Very conservative deterministic response template for unit testing
        return (
            "This is a synthetic summary generated by DummyLLMClient. "
            "Replace with your configured LLM client for real outputs.\n\n"
            "Prompt received:\n" + prompt[:2000]
        )


# -----------------
# Agent / Summarizer
# -----------------

class Summarizer:
    def __init__(self, llm_client: LLMClientProtocol):
        self.llm = llm_client

    def build_prompt(self, records: List[MedicationRecord]) -> str:
        """Construct a robust prompt that provides structured context to the LLM while
        minimizing exposure of raw PHI (prefer anonymization as required by policy)."""
        lines = [
            "You are given a list of medications. Produce a concise, patient-friendly summary.",
            "For each medication include: name, likely purpose (therapeutic class if determinable), dosing cadence, and recent refill information.",
            "Avoid prescriptive or diagnostic statements. If information is missing, say so succinctly.",
            "",
            "MEDICATIONS:\n",
        ]

        for r in records:
            lr = r.last_refill.isoformat() if r.last_refill else "unknown"
            lines.append(f"- Name: {r.name} | Dose: {r.dose or 'unknown'} | Frequency: {r.frequency or 'unknown'} | Last refill: {lr} | Notes: {r.notes or 'none'}")

        lines.append("\nGenerate a single JSON object with keys: summary_text, medications (list of objects with name and short_summary).\n")
        prompt = "\n".join(lines)
        LOG.debug("Constructed prompt length=%d", len(prompt))
        return prompt

    def summarize(self, records: List[MedicationRecord], temperature: float = 0.0) -> Dict[str, Any]:
        prompt = self.build_prompt(records)
        raw = self.llm.generate(prompt=prompt, temperature=temperature)

        # Best-effort JSON extraction: try to find a JSON object in the output.
        json_obj = self._extract_json(raw)
        if json_obj:
            LOG.debug("Parsed JSON from LLM output")
            return json_obj

        # Fallback: return raw text under summary_text
        LOG.debug("Falling back to raw text output")
        return {"summary_text": raw.strip(), "medications": []}

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        # Attempt to find a JSON object anywhere in the text
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        try:
            candidate = m.group(0)
            return json.loads(candidate)
        except Exception:
            LOG.debug("LLM output contained braces but JSON parse failed")
            return None


# -----------------
# Command-line Interface
# -----------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="ai_drug_summary")
    parser.add_argument("--input", required=True, help="Path to medication CSV file")
    parser.add_argument("--output", default="./drug_summary.json", help="Output path for JSON summary")
    parser.add_argument("--log", default="INFO", help="Log level")
    parser.add_argument("--provider", default=None, help="LLM provider identifier (optional)")
    args = parser.parse_args(argv)

    configure_logging(getattr(logging, args.log.upper(), logging.INFO))

    config = AppConfig(input_path=Path(args.input), output_path=Path(args.output), llm_provider=args.provider)

    try:
        loader = DataLoader(config.input_path)
        df = loader.load()
        records = loader.to_med_records()

        # Swap DummyLLMClient for a real implementation via dependency injection
        llm_client = DummyLLMClient(provider=config.llm_provider)
        summarizer = Summarizer(llm_client)

        result = summarizer.summarize(records, temperature=config.temperature)

        # Write output atomically
        tmp = Path(str(config.output_path) + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2, default=str)
        tmp.replace(config.output_path)

        LOG.info("Summary written to %s", config.output_path)
        return 0

    except Exception as exc:  # broad catch for CLI: surface and exit non-zero
        LOG.exception("Fatal error: %s", exc)
        return 2


# -----------------
# If imported elsewhere, provide a programmatic API
# -----------------

__all__ = [
    "AppConfig",
    "DataLoader",
    "MedicationRecord",
    "DummyLLMClient",
    "Summarizer",
]


if __name__ == "__main__":
    raise SystemExit(main())
