import csv
import hashlib
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
    def test_preparacao_reparada_reusa_1999_arquivos_e_ignora_retired(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            work = root / "dados_externos" / (benchmark.NAME + ".incompleto")
            work.mkdir(parents=True)
            rows, hashes = [], {}
            for index in range(2000):
                real = index < 1000
                ident = f"item{index:04d}"
                image = f"images/{'real' if real else 'fake'}/{ident}.jpg"
                row = {"id": ident, "image": image, "label": int(not real),
                       "source": "RAISE" if real else "MNW", "group": "g",
                       "url": f"http://193.205.194.113/RAISE/NEF/{ident}.NEF"}
                rows.append(row)
                data = b"old-jpeg" if index == 1 else f"jpeg-{index}".encode()
                path = work / image
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                hashes[image] = benchmark.digest(path)
                source_data = b"old-raw" if index == 1 else f"raw-{index}".encode()
                hashes["source:" + ident] = hashlib.sha256(source_data).hexdigest()
            (work / "plan.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")
            new_row = {**rows[1], "id": "replacement", "url": "http://193.205.194.113/RAISE/NEF/replacement.NEF",
                       "image": "images/real/replacement.jpg",
                       "expected_source_sha256": hashlib.sha256(b"new-raw").hexdigest(),
                       "expected_image_sha256": hashlib.sha256(b"new-jpeg").hexdigest()}
            repaired_rows = list(rows)
            repaired_rows[1] = new_row
            (work / "plan_repaired.json").write_text(json.dumps({
                "rows": repaired_rows, "retired_rows": [rows[1]]}), encoding="utf-8")
            (work / "hashes.json").write_text(json.dumps(hashes), encoding="utf-8")

            def fake_download(url, path):
                path.write_bytes(b"new-raw")

            def fake_convert(original_path, target):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"new-jpeg")

            args = SimpleNamespace(root=root, local_dir=root / "content/benchmark",
                                   raise_csv=root / "RAISE_1k.csv")
            with patch.object(benchmark, "download", side_effect=fake_download) as dl, patch.object(
                    benchmark, "convert", side_effect=fake_convert), patch.object(
                    benchmark, "verify_no_known_overlap"):
                benchmark.prepare(args)
            self.assertEqual(dl.call_count, 1)
            manifest = json.loads((root / "dados_externos" / benchmark.NAME / "manifest.json").read_text())
            self.assertEqual(len(manifest["rows"]), 2000)
            self.assertEqual(len({row["image_sha256"] for row in manifest["rows"]}), 2000)
            self.assertTrue((root / "dados_externos" / benchmark.NAME / rows[1]["image"]).exists())

    def test_reparo_cria_plano_derivado_sem_alterar_selecao_original(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            work = root / "dados_externos" / (benchmark.NAME + ".incompleto")
            work.mkdir(parents=True)
            rows = [
                {"id": ident, "image": f"images/real/{ident}.jpg", "label": 0,
                 "source": "RAISE", "group": "camera", "url": f"http://193.205.194.113/RAISE/NEF/{ident}.NEF"}
                for ident in ("r1", "r2")
            ]
            rows.append({"id": "AI_Images/g/f1.jpg", "image": "images/fake/g/f1.jpg",
                         "label": 1, "source": "MNW", "group": "g", "url": "https://github.com/f1.jpg"})
            original = {"rows": rows, "mnw_commit": "a" * 40}
            (work / "plan.json").write_text(json.dumps(original), encoding="utf-8")
            hashes = {"source:r1": "same-raw", "source:r2": "same-raw",
                      "images/real/r1.jpg": "same-jpeg", "images/real/r2.jpg": "same-jpeg",
                      "source:AI_Images/g/f1.jpg": "fake-raw", "images/fake/g/f1.jpg": "fake-jpeg"}
            (work / "hashes.json").write_text(json.dumps(hashes), encoding="utf-8")
            extra = root / "RAISE_2k.csv"
            with extra.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["NEF", "Device"])
                writer.writeheader()
                writer.writerow({"NEF": "http://193.205.194.113/RAISE/NEF/r3.NEF",
                                 "Device": "camera"})

            def fake_download(url, path):
                path.write_bytes(b"new-raw")

            def fake_convert(original_path, target):
                target.write_bytes(b"new-jpeg")

            args = SimpleNamespace(root=root, raise_extra_csv=extra,
                                   local_dir=root / "content/benchmark")
            with patch.object(benchmark, "download", side_effect=fake_download), patch.object(
                    benchmark, "convert", side_effect=fake_convert):
                benchmark.repair_raise_duplicates(args)
            self.assertEqual(json.loads((work / "plan.json").read_text()), original)
            repaired = json.loads((work / "plan_repaired.json").read_text())
            self.assertEqual(len(repaired["retired_rows"]), 1)
            self.assertEqual({row["id"] for row in repaired["rows"] if row["label"] == 0},
                             {"r1", "r3"})

    def test_auditoria_distingue_hash_faltante_de_duplicata(self):
        plan = {"rows": [
            {"id": "r1", "image": "images/real/r1.jpg", "label": 0, "group": "camera"},
            {"id": "f1", "image": "images/fake/f1.jpg", "label": 1, "group": "gerador"},
        ]}
        hashes = {"source:r1": "a", "images/real/r1.jpg": "b",
                  "source:f1": "c", "images/fake/f1.jpg": "b"}
        result = benchmark.audit_hashes(plan, hashes)
        self.assertEqual(result["missing"], [])
        self.assertEqual(len(result["duplicates"]), 1)
        self.assertEqual({item["label"] for item in result["duplicates"][0]}, {0, 1})
        del hashes["source:f1"]
        result = benchmark.audit_hashes(plan, hashes)
        self.assertEqual(result["missing"], ["source:f1"])

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
