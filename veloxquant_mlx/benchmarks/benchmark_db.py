"""Persistent benchmark result store for the auto-selection pipeline.

Records measured KV-cache outcomes (memory, throughput, latency, quality
delta) per ``(method, model, workload, environment)`` and lets the planner
override its analytical estimates with real numbers when a close match exists.
Storage is a plain JSON directory (an ``index.json`` plus one file per record)
under ``~/.cache/veloxquant/benchmarks/`` so it is diff-able, debuggable, and
portable. No record is ever partially written: files are atomically renamed.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from veloxquant_mlx.planning.workload import WorkloadProfile, workload_from_dict
from veloxquant_mlx.profiling.model_profiler import ModelProfile

__all__ = [
    "BenchmarkRecord",
    "BenchmarkDatabase",
    "benchmark_fingerprint",
    "default_benchmark_dir",
]

_PATH_LOCK = threading.Lock()
_INDEX_VERSION = 1


def default_benchmark_dir() -> Path:
    """``~/.cache/veloxquant/benchmarks`` (respects ``$XDG_CACHE_HOME``)."""
    cache_root = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(cache_root) / "veloxquant" / "benchmarks"


def benchmark_fingerprint(
    method: str, model_id: str, context_length: int, batch_size: int
) -> str:
    """Stable record id: same method+model+workload shape -> same key.

    Environment is deliberately *not* in the fingerprint, because a record on
    a different chip is a different *matching* record, not a different *key*
    — the store keeps both and matching scores closeness.
    """
    raw = f"{method}\x00{model_id}\x00{context_length}\x00{batch_size}".encode()
    return hashlib.sha1(raw).hexdigest()[:16]


@dataclass
class BenchmarkRecord:
    """One measured KV-cache outcome, keyed by :func:`benchmark_fingerprint`.

    Attributes:
        method: Name of the KV-cache strategy measured.
        model_id: Model identifier the measurement was taken on.
        architecture: Architecture slug (``"llama"``, ...) for fuzzy matching.
        context_length: Context length the run used.
        batch_size: Attention batch the run used.
        memory_bytes: Measured resident KV-cache footprint in bytes.
        memory_reduction: Measured ratio of ``memory_bytes`` to the fp16
            baseline at the same context/batch (0..1; ``0.19`` = 81% savings).
        throughput_tok_s: Decode throughput in tokens/second.
        latency_ms_per_token: Mean per-token latency in milliseconds.
        perplexity_delta: PPL change vs fp16 baseline, when measured (None = n/a).
        chip: Apple-Silicon chip measured on (``"M4"``).
        mlx_version: MLX version measured on.
        macos_version: macOS version measured on.
        workload: Workload shape this run approximates.
        created_at: ISO-8601 timestamp of when the record was recorded.
        source: Origin of the record (``"manual"``, ``"veloxquant"``).
        tags: Free-form labels (e.g. ``["metal-kernel"]``).
    """

    id: str
    method: str
    model_id: str
    architecture: str = "unknown"
    context_length: int = 4096
    batch_size: int = 1
    memory_bytes: int = 0
    memory_reduction: float = 1.0
    throughput_tok_s: float = 0.0
    latency_ms_per_token: float = 0.0
    perplexity_delta: float | None = None
    chip: str = "unknown"
    mlx_version: str = "unknown"
    macos_version: str = "unknown"
    workload: WorkloadProfile = field(default_factory=WorkloadProfile)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source: str = "manual"
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkRecord:
        """Build a record from its JSON-decoded dict form (inverse of :meth:`to_dict`)."""
        rec = cls(**{k: v for k, v in data.items() if k != "workload"})
        rec.workload = workload_from_dict(data.get("workload") or {})
        return rec

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict form of this record, including the workload."""
        data = asdict(self)
        data["workload"] = self.workload.to_dict()
        return data

    @property
    def savings_percent(self) -> float:
        """Memory savings vs the fp16 baseline, as a percentage (0..100)."""
        return max(0.0, (1.0 - self.memory_reduction) * 100.0)

    def match_score(self, other: BenchmarkRecord | BenchmarkMatchQuery) -> float:
        """Closeness to another record/query, higher is a better match (0..1).

        Compares the variables that change the measured values most: the
        context, the batch, the chip generation and the MLX version. Perfect
        match scores 1.0; each-axis difference costs a fraction.
        """
        score = 1.0
        score -= 0.35 * _relative_gap(self.context_length, other.context_length)
        score -= 0.25 * _relative_gap(self.batch_size, other.batch_size)
        if self.chip != "unknown" and other.chip != "unknown":
            score -= 0.25 * _chip_gap_severity(self.chip, other.chip)
        if (
            self.mlx_version not in ("unknown", "")
            and other.mlx_version not in ("unknown", "")
            and self.mlx_version != other.mlx_version
        ):
            score -= 0.15
        return max(0.0, round(score, 6))


@dataclass
class BenchmarkMatchQuery:
    """A fuzzy-matching target for :meth:`BenchmarkDatabase.find_best_match`.

    Fields that the dataclass does not provide are compared against the query
    via default values matching identity.
    """

    method: str
    model_id: str
    architecture: str = "unknown"
    context_length: int = -1
    batch_size: int = -1
    chip: str = "unknown"
    mlx_version: str = "unknown"
    macos_version: str = "unknown"


def _relative_gap(a: int, b: int) -> float:
    if b <= 0:
        return 0.0
    return min(1.0, abs(a - b) / max(b, 1))


def _chip_gap_severity(a: str, b: str) -> float:
    def gen(name: str) -> int:
        """Extract the Apple-Silicon generation number from a chip name (e.g. ``"M4"`` -> 4)."""
        import re

        m = re.search(r"M([1-9])", name.upper())
        return int(m.group(1)) if m else (1 if "M" in name.upper() else 0)

    return min(1.0, abs(gen(a) - gen(b)) / 3.0)


class BenchmarkDatabase:
    """Bounded JSON store of KV-cache benchmark records.

    All mutations are serialized under a module lock and written atomically
    (temp file + ``os.replace``), so concurrent writers cannot corrupt the
    store.
    """

    def __init__(self, storage_dir: Path | str | None = None) -> None:
        self.storage_dir = Path(storage_dir or default_benchmark_dir())
        self.index_path = self.storage_dir / "index.json"
        self.records_dir = self.storage_dir / "records"
        self._records: dict[str, BenchmarkRecord] = {}
        self.load()

    # -- persistence -----------------------------------------------------------

    def load(self) -> None:
        """Load the index and every record file (skipping corrupt ones)."""
        if not self.index_path.exists():
            self._records = {}
            return
        try:
            index = json.loads(self.index_path.read_text())
        except (OSError, json.JSONDecodeError):
            self._records = {}
            return
        raw_records: dict[str, Any] = index.get("records", {})
        records: dict[str, BenchmarkRecord] = {}
        for rid, meta in raw_records.items():
            path = self.storage_dir / str(meta.get("path", f"records/{rid}.json"))
            try:
                blob = json.loads(path.read_text())
                records[rid] = BenchmarkRecord.from_dict(blob)
            except (OSError, json.JSONDecodeError, TypeError, KeyError) as exc:
                # A corrupt record must not take the whole index down.
                records[rid] = self._placeholder_record(rid, meta, str(exc))
        self._records = records

    @staticmethod
    def _placeholder_record(rid: str, meta: dict[str, Any], why: str) -> BenchmarkRecord:
        return BenchmarkRecord(
            id=rid,
            method=str(meta.get("method", "unknown")),
            model_id=str(meta.get("model_id", "unknown")),
            tags=["corrupt", why],
        )

    def save(self) -> None:
        """Persist the whole store atomically."""
        with _PATH_LOCK:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self.records_dir.mkdir(parents=True, exist_ok=True)
            for rid, rec in self._records.items():
                if "corrupt" in rec.tags:
                    continue  # don't rewrite what we couldn't parse
                self._atomic_write(self.records_dir / f"{rid}.json", rec.to_dict())
            index = {
                "version": _INDEX_VERSION,
                "records": {
                    rid: {
                        "method": rec.method,
                        "model_id": rec.model_id,
                        "path": f"records/{rid}.json",
                    }
                    for rid, rec in self._records.items()
                },
            }
            self._atomic_write(self.index_path, index)

    @staticmethod
    def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        tmp_path = Path(tmp)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            tmp_path.replace(path)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp_path.unlink()
            raise

    # -- query -----------------------------------------------------------------

    @property
    def records(self) -> dict[str, BenchmarkRecord]:
        """Shallow copy of all stored records, keyed by record id."""
        return dict(self._records)

    def find(self, record_id: str) -> BenchmarkRecord | None:
        """Look up a single record by id, or ``None`` if it isn't stored."""
        return self._records.get(record_id)

    def find_best_match(
        self,
        model: ModelProfile,
        workload: WorkloadProfile,
        methods: list[str] | None = None,
        *,
        threshold: float = 0.6,
        hardware: Any | None = None,
    ) -> dict[str, BenchmarkRecord]:
        """Best stored record per method, and only when it matches closely enough.

        :param model: Model to match against (architecture + model_id used).
        :param workload: Workload to match against.
        :param methods: Candidate methods to look up; all when ``None``.
        :param threshold: Minimum :meth:`BenchmarkRecord.match_score`; matches
            below it are considered not-measured and omitted (avoids handing
            the planner numbers from an unrelated machine).
        :param hardware: Optional :class:`HardwareProfile`; fills the chip /
            version fields on the query so chip dissimilarity is *penalized*
            rather than ignored.
        """
        wanted = set(methods) if methods is not None else {
            rec.method for rec in self._records.values()
        }
        chip = mlx_version = macos_version = "unknown"
        if hardware is not None:
            chip = getattr(hardware, "chip", "unknown") or "unknown"
            mlx_version = getattr(hardware, "mlx_version", "unknown") or "unknown"
            macos_version = getattr(hardware, "macos_version", "unknown") or "unknown"
        query = BenchmarkMatchQuery(
            method="*",
            model_id=model.model_id,
            architecture=model.architecture,
            context_length=workload.context_length,
            batch_size=workload.batch_size,
            chip=chip,
            mlx_version=mlx_version,
            macos_version=macos_version,
        )
        best: dict[str, BenchmarkRecord] = {}
        for rec in self._records.values():
            if rec.method not in wanted:
                continue
            if not (rec.model_id == query.model_id or rec.architecture == query.architecture):
                continue
            score = rec.match_score(query)
            if score < threshold:
                continue
            current = best.get(rec.method)
            if current is None:
                best[rec.method] = rec
                continue
            current_score = current.match_score(query)
            if score > current_score or (score == current_score and rec.id < current.id):
                best[rec.method] = rec
        return best

    # -- writes ----------------------------------------------------------------

    def upsert(self, record: BenchmarkRecord) -> BenchmarkRecord:
        """Insert or replace a record by its id, then persist."""
        record.id = record.id or benchmark_fingerprint(
            record.method, record.model_id, record.context_length, record.batch_size
        )
        self._records[record.id] = record
        self.save()
        return record

    def add(
        self,
        method: str,
        model_id: str,
        *,
        context_length: int,
        batch_size: int = 1,
        memory_bytes: int = 0,
        memory_reduction: float | None = None,
        throughput_tok_s: float = 0.0,
        latency_ms_per_token: float = 0.0,
        perplexity_delta: float | None = None,
        chip: str = "unknown",
        mlx_version: str = "unknown",
        macos_version: str = "unknown",
        workload: WorkloadProfile | None = None,
        source: str = "manual",
        tags: list[str] | None = None,
        architecture: str | None = None,
        model: ModelProfile | None = None,
    ) -> BenchmarkRecord:
        """Build, persist, and return a new record (keyed by fingerprint)."""
        if memory_reduction is None:
            memory_reduction = 1.0
        if architecture is None and model is not None:
            architecture = model.architecture
        wl = workload or WorkloadProfile(
            context_length=context_length, batch_size=batch_size
        )
        rid = benchmark_fingerprint(method, model_id, context_length, batch_size)
        rec = BenchmarkRecord(
            id=rid,
            method=method,
            model_id=model_id,
            architecture=architecture or "unknown",
            context_length=context_length,
            batch_size=batch_size,
            memory_bytes=memory_bytes,
            memory_reduction=memory_reduction,
            throughput_tok_s=throughput_tok_s,
            latency_ms_per_token=latency_ms_per_token,
            perplexity_delta=perplexity_delta,
            chip=chip,
            mlx_version=mlx_version,
            macos_version=macos_version,
            workload=wl,
            source=source,
            tags=tags or [],
        )
        return self.upsert(rec)

    def remove(self, record_id: str) -> bool:
        """Delete a record by id and persist. Returns ``False`` if it wasn't found."""
        if record_id not in self._records:
            return False
        del self._records[record_id]
        self.save()
        return True

    def clear(self) -> None:
        """Delete all records and persist the now-empty store."""
        self._records = {}
        self.save()
