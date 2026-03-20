from __future__ import annotations

import gzip
import hashlib
import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import torch
from torch.utils.data import Dataset, IterableDataset


@dataclass(frozen=True)
class DolmaRecord:
    doc_id: str
    text: str
    source: str
    metadata: dict


class PackedSequenceDataset(Dataset):
    def __init__(self, sequences: list[list[int]]):
        self._sequences = [torch.tensor(sequence, dtype=torch.long) for sequence in sequences]

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._sequences[index]


class CachedStreamingDataset(Dataset):
    """Wraps a StreamingPackedDataset, materializing up to `max_sequences` entries on
    first iteration and serving them from memory on all subsequent calls.
    Avoids re-reading and re-tokenizing files on every eval pass."""

    def __init__(self, streaming: "StreamingPackedDataset", max_sequences: int):
        self._streaming = streaming
        self._max_sequences = max_sequences
        self._cache: list[torch.Tensor] | None = None

    def _ensure_cache(self) -> None:
        if self._cache is not None:
            return
        self._cache = []
        for seq in self._streaming:
            self._cache.append(seq)
            if len(self._cache) >= self._max_sequences:
                break

    def __len__(self) -> int:
        self._ensure_cache()
        return len(self._cache)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> torch.Tensor:
        self._ensure_cache()
        return self._cache[index]  # type: ignore[index]


class StreamingPackedDataset(IterableDataset):
    """Streams sequences from disk, tokenizing on the fly. Never loads the full dataset into RAM."""

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | Path,
        tokenizer,
        *,
        split: str,
        seq_len: int,
        max_docs: int | None = None,
        max_sequences: int | None = None,
    ):
        self._dataset_name = dataset_name
        self._dataset_path = dataset_path
        self._tokenizer = tokenizer
        self._split = split
        self._seq_len = seq_len
        self._max_docs = max_docs
        self._max_sequences = max_sequences

    def __iter__(self) -> Iterator[torch.Tensor]:
        eos = self._tokenizer.eos_token_id
        buf: list[int] = []
        sequences_yielded = 0

        for record in iter_dataset_records(
            self._dataset_name,
            self._dataset_path,
            max_docs=self._max_docs,
        ):
            if document_split(record.doc_id) != self._split:
                continue
            encoded = self._tokenizer(record.text, add_special_tokens=False, truncation=False)
            token_ids = encoded["input_ids"]
            if not token_ids:
                continue
            buf.extend(token_ids)
            buf.append(eos)
            while len(buf) >= self._seq_len:
                yield torch.tensor(buf[: self._seq_len], dtype=torch.long)
                del buf[: self._seq_len]
                sequences_yielded += 1
                if self._max_sequences is not None and sequences_yielded >= self._max_sequences:
                    return


def resolve_dataset_files(dataset_path: str | Path) -> list[Path]:
    path = Path(dataset_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    if path.is_file():
        return [path]

    preferred_patterns = ("*.json.gz", "*.jsonl.gz", "*.ndjson.gz")
    files: list[Path] = []
    for pattern in preferred_patterns:
        files.extend(path.rglob(pattern))

    if not files:
        files = list(path.rglob("*.gz"))

    files = sorted(file_path for file_path in files if file_path.is_file())
    if not files:
        raise FileNotFoundError(
            f"Dataset directory does not contain any gzip files: {path}"
        )
    return files


def resolve_fineweb_files(dataset_path: str | Path) -> list[Path]:
    path = Path(dataset_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    if path.is_file():
        if path.suffix != ".parquet":
            raise ValueError(f"FineWeb files must be parquet files, got: {path}")
        return [path]

    files = sorted(file_path for file_path in path.rglob("*.parquet") if file_path.is_file())
    if not files:
        raise FileNotFoundError(
            f"Dataset directory does not contain any parquet files: {path}"
        )
    return files


_GZIP_MAGIC = b"\x1f\x8b"
_GZIP_EOF_TRAILER_LEN = 8  # CRC32 (4 bytes) + ISIZE (4 bytes)


def _gzip_file_complete(path: Path) -> bool:
    """Return True only if the file looks like a fully written gzip stream.

    A valid gzip file starts with the two-byte magic number and ends with an
    8-byte trailer (CRC32 + uncompressed size mod 2^32).  A file that is still
    being downloaded will typically be missing the trailer entirely.
    """
    try:
        size = path.stat().st_size
        if size < len(_GZIP_MAGIC) + _GZIP_EOF_TRAILER_LEN:
            return False
        with open(path, "rb") as fh:
            header = fh.read(2)
            if header != _GZIP_MAGIC:
                return False
        return True
    except OSError:
        return False


class _PendingFileWatcher:
    """Watches a set of incomplete gzip files in a background thread.

    When a file finishes downloading (passes _gzip_file_complete), it is placed
    on `ready_queue` so the caller can process it without re-scanning the directory.
    Call `stop()` when done.
    """

    _POLL_INTERVAL = 5.0  # seconds between completion checks

    def __init__(self, paths: list[Path]):
        self.ready_queue: queue.Queue[Path] = queue.Queue()
        self._pending: list[Path] = list(paths)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while self._pending and not self._stop_event.is_set():
            still_pending: list[Path] = []
            for path in self._pending:
                if _gzip_file_complete(path):
                    print(f"pending file now ready: {path}")
                    self.ready_queue.put(path)
                else:
                    still_pending.append(path)
            self._pending = still_pending
            if self._pending:
                self._stop_event.wait(timeout=self._POLL_INTERVAL)

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join()

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)


def _iter_one_file(path: Path, docs_seen: int, max_docs: int | None) -> Iterator[tuple[DolmaRecord, int]]:
    """Yield (record, updated_docs_seen) for every valid record in a single gzip file."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            text = raw.get("text", "")
            doc_id = raw.get("id")
            if not doc_id or not text:
                continue
            docs_seen += 1
            yield DolmaRecord(
                doc_id=doc_id,
                text=text,
                source=raw.get("source", ""),
                metadata=raw.get("metadata", {}),
            ), docs_seen
            if max_docs is not None and docs_seen >= max_docs:
                return


def iter_dolma_records(dataset_path: str | Path, max_docs: int | None = None) -> Iterator[DolmaRecord]:
    dataset_files = resolve_dataset_files(dataset_path)
    docs_seen = 0

    ready: list[Path] = []
    pending: list[Path] = []
    for path in dataset_files:
        if _gzip_file_complete(path):
            ready.append(path)
        else:
            print(f"file still downloading, will wait: {path}")
            pending.append(path)

    watcher = _PendingFileWatcher(pending) if pending else None

    try:
        # Process all files that were ready at startup first.
        for path in ready:
            try:
                for record, docs_seen in _iter_one_file(path, docs_seen, max_docs):
                    yield record
                    if max_docs is not None and docs_seen >= max_docs:
                        return
            except (gzip.BadGzipFile, EOFError, OSError) as exc:
                print(f"skipping unreadable file {path}: {exc}")

        # Drain files as they finish downloading.
        if watcher is not None:
            while watcher.has_pending or not watcher.ready_queue.empty():
                try:
                    path = watcher.ready_queue.get(timeout=_PendingFileWatcher._POLL_INTERVAL)
                except queue.Empty:
                    continue
                try:
                    for record, docs_seen in _iter_one_file(path, docs_seen, max_docs):
                        yield record
                        if max_docs is not None and docs_seen >= max_docs:
                            return
                except (gzip.BadGzipFile, EOFError, OSError) as exc:
                    print(f"skipping unreadable file {path}: {exc}")
    finally:
        if watcher is not None:
            watcher.stop()


def _build_fineweb_doc_id(path: Path, row_index: int, row: dict) -> str:
    existing_id = row.get("id")
    if existing_id:
        return str(existing_id)

    fingerprint = "|".join(
        [
            str(path),
            str(row_index),
            str(row.get("url", "")),
            str(row.get("dump", "")),
            str(row.get("file_path", "")),
            str(row.get("text", ""))[:256],
        ]
    )
    return hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()


def _iter_one_fineweb_file(path: Path, docs_seen: int, max_docs: int | None) -> Iterator[tuple[DolmaRecord, int]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "FineWeb support requires pyarrow. Install the project requirements first."
        ) from exc

    parquet_file = pq.ParquetFile(path)
    available_columns = set(parquet_file.schema.names)
    text_column = "text" if "text" in available_columns else None
    if text_column is None:
        raise ValueError(f"FineWeb parquet file does not contain a 'text' column: {path}")

    candidate_metadata = (
        "id",
        "dump",
        "url",
        "file_path",
        "language",
        "language_score",
        "token_count",
        "score",
        "source",
    )
    selected_columns = [text_column] + [
        name for name in candidate_metadata if name in available_columns and name != text_column
    ]

    row_index = 0
    for batch in parquet_file.iter_batches(batch_size=1024, columns=selected_columns):
        rows = batch.to_pylist()
        for row in rows:
            text = row.get(text_column, "")
            if not text:
                row_index += 1
                continue

            doc_id = _build_fineweb_doc_id(path, row_index, row)
            source = row.get("dump") or row.get("source") or "fineweb"
            metadata = {
                key: value
                for key, value in row.items()
                if key not in {text_column, "id", "dump", "source"} and value is not None
            }

            docs_seen += 1
            yield DolmaRecord(
                doc_id=doc_id,
                text=text,
                source=str(source),
                metadata=metadata,
            ), docs_seen
            row_index += 1
            if max_docs is not None and docs_seen >= max_docs:
                return


def iter_fineweb_records(dataset_path: str | Path, max_docs: int | None = None) -> Iterator[DolmaRecord]:
    dataset_files = resolve_fineweb_files(dataset_path)
    docs_seen = 0

    for path in dataset_files:
        try:
            for record, docs_seen in _iter_one_fineweb_file(path, docs_seen, max_docs):
                yield record
                if max_docs is not None and docs_seen >= max_docs:
                    return
        except (OSError, ValueError) as exc:
            print(f"skipping unreadable parquet file {path}: {exc}")


def iter_dataset_records(
    dataset_name: str,
    dataset_path: str | Path,
    max_docs: int | None = None,
) -> Iterator[DolmaRecord]:
    if dataset_name == "dolma":
        yield from iter_dolma_records(dataset_path, max_docs=max_docs)
        return
    if dataset_name == "fineweb":
        yield from iter_fineweb_records(dataset_path, max_docs=max_docs)
        return
    raise ValueError(f"Unsupported dataset '{dataset_name}'. Expected 'dolma' or 'fineweb'.")



def document_split(doc_id: str, validation_percent: int = 1) -> str:
    digest = hashlib.sha1(doc_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "val" if bucket < validation_percent else "train"


def pack_tokenized_documents(
    tokenized_documents: Iterable[list[int]],
    *,
    seq_len: int,
    eos_token_id: int,
    max_sequences: int | None = None,
) -> list[list[int]]:
    sequences: list[list[int]] = []
    buffer: list[int] = []
    offset = 0
    for document_tokens in tokenized_documents:
        if not document_tokens:
            continue
        buffer.extend(document_tokens)
        buffer.append(eos_token_id)
        while len(buffer) - offset >= seq_len:
            sequences.append(buffer[offset : offset + seq_len])
            offset += seq_len
            if max_sequences is not None and len(sequences) >= max_sequences:
                return sequences
    del buffer[:offset]
    return sequences


def build_split_sequences(
    dataset_name: str,
    dataset_path: str | Path,
    tokenizer,
    *,
    seq_len: int,
    max_docs: int | None = None,
    max_sequences_train: int | None = None,
    max_sequences_val: int | None = None,
) -> tuple[list[list[int]], list[list[int]]]:
    """Single-pass eager load. Only use when max_sequences is small enough to fit in RAM."""
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must have an EOS token id for sequence packing.")

    eos = tokenizer.eos_token_id
    train_seqs: list[list[int]] = []
    val_seqs: list[list[int]] = []
    train_buf: list[int] = []
    val_buf: list[int] = []
    train_offset = 0
    val_offset = 0
    train_done = False
    val_done = False

    for record in iter_dataset_records(dataset_name, dataset_path, max_docs=max_docs):
        split = document_split(record.doc_id)
        if split == "train" and train_done:
            continue
        if split == "val" and val_done:
            continue

        encoded = tokenizer(record.text, add_special_tokens=False, truncation=False)
        token_ids = encoded["input_ids"]
        if not token_ids:
            continue

        if split == "train":
            train_buf.extend(token_ids)
            train_buf.append(eos)
            while len(train_buf) - train_offset >= seq_len:
                train_seqs.append(train_buf[train_offset : train_offset + seq_len])
                train_offset += seq_len
                if max_sequences_train is not None and len(train_seqs) >= max_sequences_train:
                    train_done = True
                    break
        else:
            val_buf.extend(token_ids)
            val_buf.append(eos)
            while len(val_buf) - val_offset >= seq_len:
                val_seqs.append(val_buf[val_offset : val_offset + seq_len])
                val_offset += seq_len
                if max_sequences_val is not None and len(val_seqs) >= max_sequences_val:
                    val_done = True
                    break

        if train_done and val_done:
            break

    return train_seqs, val_seqs


class CodeInstructionDataset(Dataset):
    """Instruction-tuning dataset for m-a-p/Code-Feedback."""

    def __init__(self, data: list[dict], tokenizer, seq_len: int):
        self._data = data
        self._tokenizer = tokenizer
        self._seq_len = seq_len

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        item = self._data[idx]
        # Format: <|user|>\n{query}\n<|assistant|>\n{answer}
        text = f"<|user|>\n{item['query']}\n<|assistant|>\n{item['answer']}"
        tokens = self._tokenizer(text, add_special_tokens=False, truncation=True, max_length=self._seq_len)
        token_ids = tokens["input_ids"]
        # Pad or truncate to seq_len
        if len(token_ids) < self._seq_len:
            token_ids = token_ids + [self._tokenizer.eos_token_id] * (self._seq_len - len(token_ids))
        else:
            token_ids = token_ids[:self._seq_len]
        return torch.tensor(token_ids, dtype=torch.long)


def load_datasets(
    dataset_name: str,
    dataset_path: str | Path,
    tokenizer,
    *,
    seq_len: int,
    max_docs: int | None = None,
    max_sequences: int | None = None,
    val_cache_sequences: int = 500,
) -> tuple[StreamingPackedDataset | PackedSequenceDataset | CodeInstructionDataset, CachedStreamingDataset | PackedSequenceDataset | CodeInstructionDataset]:
    # Code instruction dataset
    if dataset_name == "code_feedback":
        from datasets import load_dataset
        ds = load_dataset("m-a-p/Code-Feedback", split="train")
        # 90/10 train/val split
        split_idx = int(len(ds) * 0.9)
        # Check column names and adapt
        first_item = ds[0]
        if "query" in first_item:
            train_data = [{"query": ds[i]["query"], "answer": ds[i]["answer"]} for i in range(split_idx)]
            val_data = [{"query": ds[i]["query"], "answer": ds[i]["answer"]} for i in range(split_idx, len(ds))]
        else:
            # Fallback: use first two text columns
            cols = [k for k, v in first_item.items() if isinstance(v, str)]
            train_data = [{"query": ds[i][cols[0]], "answer": ds[i][cols[1]]} for i in range(split_idx)]
            val_data = [{"query": ds[i][cols[0]], "answer": ds[i][cols[1]]} for i in range(split_idx, len(ds))]
        return CodeInstructionDataset(train_data, tokenizer, seq_len), CodeInstructionDataset(val_data, tokenizer, seq_len)

    # If max_sequences is set, the dataset is small enough to load eagerly (original behaviour).
    # Otherwise stream from disk to avoid loading the full dataset into RAM.
    if max_sequences is not None:
        train_sequences, val_sequences = build_split_sequences(
            dataset_name,
            dataset_path,
            tokenizer,
            seq_len=seq_len,
            max_docs=max_docs,
            max_sequences_train=max_sequences,
            max_sequences_val=max_sequences,
        )
        return PackedSequenceDataset(train_sequences), PackedSequenceDataset(val_sequences)

    train_dataset = StreamingPackedDataset(
        dataset_name,
        dataset_path,
        tokenizer,
        split="train",
        seq_len=seq_len,
        max_docs=max_docs,
    )
    val_streaming = StreamingPackedDataset(
        dataset_name,
        dataset_path,
        tokenizer,
        split="val",
        seq_len=seq_len,
        max_docs=max_docs,
    )
    # Wrap val in a cache so repeated eval passes don't re-read from disk.
    val_dataset = CachedStreamingDataset(val_streaming, max_sequences=val_cache_sequences)
    return train_dataset, val_dataset
