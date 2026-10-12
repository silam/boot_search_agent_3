"""
EXTRACT step: document loaders.

Each loader turns a source (file, API, ...) into a list of raw dicts. They all share one
interface, so the pipeline doesn't care where data comes from — the same idea as
LangChain / LlamaIndex "document loaders".

INTERVIEW NOTE:
  Adding a new source (e.g. a supplier's REST API, a Kafka topic, an S3 bucket) means
  writing one new class with a `load()` method. Nothing downstream changes.
"""
from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseLoader(ABC):
    @abstractmethod
    def load(self) -> list[dict[str, Any]]:
        ...


class JsonCatalogLoader(BaseLoader):
    """Loads a JSON array of product records."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> list[dict[str, Any]]:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{self.path} must contain a JSON array")
        return data


class CsvCatalogLoader(BaseLoader):
    """Loads a CSV export (e.g. from a PIM or a spreadsheet). List fields use '|' separators."""

    LIST_FIELDS = ("features", "reviews", "images")   # images: "a.png|b.png|c.png"

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> list[dict[str, Any]]:
        rows = []
        with self.path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for field in self.LIST_FIELDS:
                    if row.get(field):
                        row[field] = [x.strip() for x in row[field].split("|")]
                rows.append(row)
        return rows


class HttpCatalogLoader(BaseLoader):
    """Loads products from a paginated REST API (Data Collection & APIs).

    Expects responses like {"items": [...], "next": "<url or null>"}.
    """

    def __init__(self, url: str, headers: dict[str, str] | None = None, max_pages: int = 100):
        self.url, self.headers, self.max_pages = url, headers or {}, max_pages

    def load(self) -> list[dict[str, Any]]:
        import requests  # imported lazily: only needed when this loader is used

        items, url, pages = [], self.url, 0
        while url and pages < self.max_pages:
            resp = requests.get(url, headers=self.headers, timeout=30)
            resp.raise_for_status()
            body = resp.json()
            items.extend(body.get("items", []))
            url, pages = body.get("next"), pages + 1
        return items


def loader_for(path_or_url: str | Path) -> BaseLoader:
    """Factory: choose a loader from the source string."""
    s = str(path_or_url)
    if s.startswith(("http://", "https://")):
        return HttpCatalogLoader(s)
    if s.endswith(".csv"):
        return CsvCatalogLoader(Path(s))
    return JsonCatalogLoader(Path(s))
