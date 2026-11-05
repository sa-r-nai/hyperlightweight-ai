"""Bounded, deterministic process readers for raw Hugging Face streams."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .config import configure_cpu_worker


SOURCE_BATCH_SIZE = 64
SOURCE_PREFETCH_FACTOR = 2
MAX_SOURCE_WORKERS = 24
SERIAL_READING_METHOD = "hf-stream-serial-v1"
PARALLEL_READING_METHOD = "hf-dataloader-batch-round-robin-v1"


def _raw_batch(records: list) -> list:
    """Keep nested Python records intact instead of tensor-collating fields."""
    return records


def _initialize_reader(worker_id: int) -> None:
    # Reader processes decode data; they do not need nested numerical pools.
    configure_cpu_worker()
    try:
        import pyarrow
    except ImportError:
        return
    # Arrow's pools may already exist before the worker initializer runs.
    pyarrow.set_cpu_count(1)
    pyarrow.set_io_thread_count(1)


class _RawShardedDataset(IterableDataset):
    """Adapt shardable raw streams without applying tensor formatting."""

    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def __iter__(self):
        dataset = self.dataset
        worker = get_worker_info()
        # datasets 4.1.1 automatically makes its raw IterableDataset a Torch
        # IterableDataset and shards it in __iter__. Sharding it here again
        # would drop data. Non-Torch adapters must partition explicitly.
        if worker is not None and not isinstance(dataset, IterableDataset):
            dataset = dataset.shard(num_shards=worker.num_workers, index=worker.id, contiguous=False)
        records = iter(dataset)
        try:
            yield from records
        finally:
            close = getattr(records, "close", None)
            if close is not None:
                close()


def iter_huggingface_source(source: Mapping, *, workers: int = 1,
                            reading_info: dict | None = None) -> Iterator[object]:
    """Read source shards in bounded processes while retaining raw field types.

    Multiple readers are consumed in deterministic batches, so their order
    differs from a serial stream. The reading metadata records that choice.
    Sources with one shard or unknown shard metadata retain their serial order.
    Closing this generator also shuts down prefetched reader processes.
    """
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer.")
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("Install requirements-native-600m.txt before downloading datasets.") from error

    kwargs = {"revision": source["revision"], "split": source["split"], "streaming": True}
    if "config" in source:
        kwargs["name"] = source["config"]
    dataset = load_dataset(source["id"], **kwargs)
    shards = getattr(dataset, "num_shards", None)
    if type(shards) is not int or shards < 1:
        shards = None
    active = min(workers, MAX_SOURCE_WORKERS, shards) if shards is not None else 1
    method = PARALLEL_READING_METHOD if active > 1 else SERIAL_READING_METHOD
    details = {"method": method, "requested_workers": workers, "active_workers": active,
               "num_shards": shards, "batch_size": SOURCE_BATCH_SIZE if active > 1 else None,
               "prefetch_factor": SOURCE_PREFETCH_FACTOR if active > 1 else None}
    if reading_info is not None:
        reading_info.update(details)
    label = source.get("config", source["id"])
    print(f"[source:{label}] requested_workers={workers} active_workers={active} "
          f"num_shards={shards} method={method}", flush=True)

    if active == 1:
        records = iter(dataset)
        try:
            yield from records
        finally:
            close = getattr(records, "close", None)
            if close is not None:
                close()
        return

    loader = DataLoader(_RawShardedDataset(dataset), batch_size=SOURCE_BATCH_SIZE,
                        num_workers=active, collate_fn=_raw_batch, worker_init_fn=_initialize_reader,
                        prefetch_factor=SOURCE_PREFETCH_FACTOR, multiprocessing_context="spawn",
                        persistent_workers=False, in_order=True)
    batches = iter(loader)
    try:
        for batch in batches:
            yield from batch
    finally:
        # Torch 2.8 exposes no public iterator.close(). Its pinned iterator
        # cleanup is also used on exhaustion/GC; call it explicitly so an early
        # data quota or a reader failure does not leave processes downloading.
        shutdown = getattr(batches, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
