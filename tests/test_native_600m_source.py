"""Offline source-reader checks, including real spawned process cleanup."""

from __future__ import annotations

import importlib.util
import multiprocessing
import os
import sys
import types
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from native_600m.source import (
    PARALLEL_READING_METHOD, SERIAL_READING_METHOD, SOURCE_BATCH_SIZE,
    iter_huggingface_source,
)


SOURCE = {"kind": "huggingface", "id": "fixture/source", "revision": "pinned-revision",
          "split": "train", "config": "fixture-config"}


class ShardedFixture:
    """Picklable raw source; process-local events expose actual reader activity."""

    def __init__(self, shards=tuple(range(4)), *, rows=70, events=None, fail_at=None):
        self.shards, self.rows, self.events, self.fail_at = shards, rows, events, fail_at
        self.num_shards = len(shards)

    def shard(self, num_shards, index, contiguous=False):
        if contiguous:
            raise AssertionError("Readers must partition source shards deterministically.")
        return ShardedFixture(self.shards[index::num_shards], rows=self.rows,
                              events=self.events, fail_at=self.fail_at)

    def __iter__(self):
        try:
            import pyarrow
            arrow_cpu, arrow_io = pyarrow.cpu_count(), pyarrow.io_thread_count()
        except ImportError:
            arrow_cpu, arrow_io = None, None
        handle = (Path(self.events) / f"{os.getpid()}.log").open("a", buffering=1) if self.events else None
        try:
            for shard in self.shards:
                for row in range(self.rows):
                    if (shard, row) == self.fail_at:
                        raise RuntimeError("synthetic source failure")
                    if handle:
                        handle.write("read\n")
                    yield {"text": f"문서 {shard}:{row}", "number": row, "worker_pid": os.getpid(),
                           "arrow_cpu_threads": arrow_cpu, "arrow_io_threads": arrow_io,
                           "omp_threads": os.environ.get("OMP_NUM_THREADS"),
                           "messages": [{"role": "user", "content": f"질문 {row}"}]}
        finally:
            if handle:
                handle.close()


def hf_fixture(shards, rows):
    yield from ShardedFixture(shards, rows=rows)


class SourceReaderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.children_before = {process.pid for process in multiprocessing.active_children()}

    def reader(self, dataset, *, workers, info=None):
        # The loader is called in the parent; the resulting fixture itself is
        # pickled into real spawned readers, with no network or worker mocks.
        module = types.ModuleType("datasets")
        module.load_dataset = load = Mock(return_value=dataset)
        with patch.dict(sys.modules, {"datasets": module}):
            records = iter_huggingface_source(SOURCE, workers=workers, reading_info=info)
            first = next(records)
        load.assert_called_once_with("fixture/source", revision="pinned-revision", split="train",
                                     streaming=True, name="fixture-config")
        self.addCleanup(records.close)
        return first, records

    def assert_readers_stopped(self):
        remaining = [process for process in multiprocessing.active_children()
                     if process.pid not in self.children_before]
        self.assertEqual(remaining, [])

    def test_spawned_readers_preserve_raw_records_and_deterministic_batch_order(self):
        values = []
        for _ in range(2):
            info = {}
            first, records = self.reader(ShardedFixture(), workers=2, info=info)
            rows = [first, *records]
            self.assertEqual(len(rows), 280)
            self.assertEqual(len({row["text"] for row in rows}), 280)
            self.assertEqual(len({row["worker_pid"] for row in rows}), 2)
            self.assertNotIn(os.getpid(), {row["worker_pid"] for row in rows})
            self.assertTrue(all(type(row["number"]) is int for row in rows))
            self.assertTrue(all(type(row["messages"]) is list for row in rows))
            self.assertTrue(all(row["omp_threads"] == "1" for row in rows))
            self.assertTrue(all(row["arrow_cpu_threads"] in (None, 1) and row["arrow_io_threads"] in (None, 1)
                                for row in rows))
            self.assertEqual([row["text"] for row in rows[:64]], [f"문서 0:{i}" for i in range(64)])
            self.assertEqual([row["text"] for row in rows[64:128]], [f"문서 1:{i}" for i in range(64)])
            self.assertEqual(info, {"method": PARALLEL_READING_METHOD, "requested_workers": 2,
                                    "active_workers": 2, "num_shards": 4, "batch_size": 64,
                                    "prefetch_factor": 2})
            values.append([row["text"] for row in rows])
            self.assert_readers_stopped()
        self.assertEqual(values[0], values[1])

    def test_single_worker_single_shard_and_unknown_shards_fall_back_to_serial(self):
        for dataset, workers in ((ShardedFixture(rows=2), 1), (ShardedFixture((0,), rows=2), 24),
                                 (iter([{"text": "그대로"}]), 24)):
            with self.subTest(workers=workers, dataset=type(dataset).__name__):
                info = {}
                first, records = self.reader(dataset, workers=workers, info=info)
                rows = [first, *records]
                self.assertEqual(info["active_workers"], 1)
                self.assertEqual(info["method"], SERIAL_READING_METHOD)
                self.assertTrue(all(row.get("worker_pid", os.getpid()) == os.getpid() for row in rows))
                self.assert_readers_stopped()

    def test_prefetch_is_bounded_and_early_close_stops_spawned_readers(self):
        first, records = self.reader(ShardedFixture(rows=100_000, events=str(self.root)), workers=2)
        self.assertNotEqual(first["worker_pid"], os.getpid())
        time.sleep(0.2)
        records.close()
        consumed = sum(path.read_text().count("read\n") for path in self.root.glob("*.log"))
        # Two prefetched batches per worker, plus the batch delivered to the
        # flattening generator. The full source contains 400,000 records.
        self.assertLessEqual(consumed, (2 * 2 + 1) * SOURCE_BATCH_SIZE)
        self.assertGreaterEqual(consumed, SOURCE_BATCH_SIZE)
        self.assert_readers_stopped()

    def test_worker_error_propagates_and_shuts_down_other_readers(self):
        first, records = self.reader(ShardedFixture(fail_at=(0, 65)), workers=2)
        with self.assertRaisesRegex(RuntimeError, "synthetic source failure"):
            list(records)
        self.assert_readers_stopped()

    def test_invalid_worker_counts_are_rejected_before_loading(self):
        for workers in (0, -1, True, 1.5):
            with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, "workers"):
                next(iter_huggingface_source(SOURCE, workers=workers))

    def test_reader_budget_is_limited_by_requested_workers_shards_and_twenty_four(self):
        for requested, shards, expected in ((100, 40, 24), (24, 2, 2), (2, 40, 2)):
            with self.subTest(requested=requested, shards=shards):
                info = {}
                with patch("native_600m.source.DataLoader", return_value=[[{"text": "목록"}]]) as loader:
                    first, records = self.reader(ShardedFixture(tuple(range(shards))), workers=requested, info=info)
                    self.assertEqual([first, *records], [{"text": "목록"}])
                self.assertEqual(info["active_workers"], expected)
                self.assertEqual(info["requested_workers"], requested)
                self.assertEqual(loader.call_args.kwargs["num_workers"], expected)
                self.assertEqual(loader.call_args.kwargs["batch_size"], 64)
                self.assertEqual(loader.call_args.kwargs["prefetch_factor"], 2)
                self.assertEqual(loader.call_args.kwargs["multiprocessing_context"], "spawn")
                self.assertTrue(loader.call_args.kwargs["in_order"])

    @unittest.skipIf(importlib.util.find_spec("datasets") is None, "datasets is not installed")
    def test_real_huggingface_iterable_shards_once_without_tensor_formatting(self):
        from datasets import IterableDataset

        with patch("datasets.config.HF_DATASETS_CACHE", self.root / "cache"):
            dataset = IterableDataset.from_generator(hf_fixture, gen_kwargs={"shards": list(range(4)), "rows": 3})
        self.assertEqual(dataset.num_shards, 4)
        from torch.utils.data import IterableDataset as TorchIterableDataset
        self.assertIsInstance(dataset, TorchIterableDataset)
        self.assertIsNone(dataset._formatting)
        first, records = self.reader(dataset, workers=2)
        rows = [first, *records]
        self.assertEqual(len(rows), 12)
        self.assertEqual({row["text"] for row in rows}, {f"문서 {shard}:{row}" for shard in range(4) for row in range(3)})
        self.assertEqual(len({row["worker_pid"] for row in rows}), 2)
        self.assertTrue(all(type(row["number"]) is int and type(row["messages"]) is list for row in rows))
        self.assert_readers_stopped()


if __name__ == "__main__":
    unittest.main()
