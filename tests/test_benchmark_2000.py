import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("benchmark_2000", ROOT / "experimentos/benchmark_2000.py")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class BenchmarkSelectionTest(unittest.TestCase):
    def test_raise_selects_all_1000_deterministically(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "RAISE.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["NEF", "Device"])
                writer.writeheader()
                for index in range(1000):
                    writer.writerow({"NEF": f"http://193.205.194.113/RAISE/NEF/r{index:05d}.NEF",
                                     "Device": "camera_A" if index % 2 else "camera_B"})
            a = benchmark.select_raise(path)
            b = benchmark.select_raise(path)
            self.assertEqual(a, b)
            self.assertEqual(len(a), 1000)
            self.assertEqual({r["label"] for r in a}, {0})
            self.assertEqual(dict(__import__("collections").Counter(r["group"] for r in a)),
                             {"camera_A": 500, "camera_B": 500})

    def test_mnw_selects_25_per_generator_and_pins_commit(self):
        commit = "a" * 40

        def fake_api(url):
            if "/commits/main" in url:
                return {"sha": commit}
            if url.endswith("/git/trees/" + commit):
                return {"tree": [{"path": "AI_Images", "type": "tree", "sha": "b" * 40}]}
            if url.endswith("/git/trees/" + "b" * 40 + "?recursive=1"):
                return {"truncated": False, "tree": [
                    {"type": "blob", "path": f"generator_{group}/{index}.jpg"}
                    for group in range(45) for index in range(30)]}
            raise AssertionError(url)

        with patch.object(benchmark, "api_json", side_effect=fake_api):
            version, rows = benchmark.select_mnw()
        self.assertEqual(version, commit)
        self.assertEqual(len(rows), 1000)
        self.assertEqual(len({r["group"] for r in rows}), 40)
        self.assertTrue(all(__import__("collections").Counter(r["group"] for r in rows)[g] == 25
                            for g in {r["group"] for r in rows}))
        self.assertTrue(all(commit in r["url"] for r in rows))

    def test_stage_requires_frozen_complete_unchanged_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "dados_externos" / benchmark.NAME
            rows = []
            for label, cls in ((0, "real"), (1, "fake")):
                for index in range(1000):
                    rel = f"images/{cls}/{index:03d}.jpg"
                    path = source / rel
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f"{cls}:{index}".encode())
                    rows.append({"label": label, "image": rel,
                                 "image_sha256": benchmark.digest(path)})
            (source / "manifest.json").write_text(
                json.dumps({"rows": rows}), encoding="utf-8")
            target = root / "content"
            args = SimpleNamespace(root=root, local_dir=target)
            benchmark.verify_and_stage(args)
            self.assertEqual(len(list(target.rglob("*.jpg"))), 2000)
            (target / rows[0]["image"]).write_bytes(b"altered")
            with self.assertRaisesRegex(ValueError, "diferente"):
                benchmark.verify_and_stage(args)
            (target / rows[0]["image"]).write_bytes(f"real:0".encode())
            (target / "manifest.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Manifesto local de outra versão"):
                benchmark.verify_and_stage(args)

    def test_resumes_existing_image_without_recorded_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "already.jpg"
            from PIL import Image
            Image.new("RGB", (300, 300), "blue").save(path)
            benchmark.verify_existing(path)
            path.write_bytes(b"interrupted")
            with self.assertRaises(Exception):
                benchmark.verify_existing(path)

    def test_same_jpeg_processing_for_fake_and_real(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = root / "source.png"
            Image.new("RGB", (2500, 2300), "green").save(original)
            for cls in ("real", "fake"):
                output = root / cls / "sample.jpg"
                benchmark.convert(original, output)
                with Image.open(output) as converted:
                    self.assertEqual(converted.format, "JPEG")
                    self.assertLessEqual(max(converted.size), 2048)
            self.assertEqual(benchmark.digest(root / "real/sample.jpg"),
                             benchmark.digest(root / "fake/sample.jpg"))


if __name__ == "__main__":
    unittest.main()
