"""Download (optional) and organise DIBCO / H-DIBCO sets into a uniform layout.

  data/dibco/<year>/img/<id>.png      input image (converted losslessly to PNG)
  data/dibco/<year>/gt/<id>.png       binary GT, text = 0, background = 255
  data/dibco/2017/weights/<id>_{R,P}Weights.dat   official pseudo-FM weights
  data/dibco/manifest.csv             year, id, split, type, source files

Split (decided before any experiment): 2009-2016 = development, 2017 = test.
Type: 'printed' / 'handwritten' from the file-name prefix where the set
mixes both (P/PR = printed, H/HW = handwritten); H-DIBCO sets are handwritten;
DIBCO 2017 types are not encoded in file names and are left as 'unknown'.

Usage:
  python scripts/prepare_dibco.py --download raw_downloads --extracted dibco_extracted --out data/dibco
Extraction needs `unar` (apt install unar) for the .rar/.7z archives.
"""
import argparse
import csv
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

import cv2
import numpy as np

URLS = {
    "2009": ["https://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/DIBC02009_Test_images-handwritten.rar",
             "https://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/DIBCO2009-GT-Test-images_handwritten.rar",
             "https://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/DIBCO2009-GT-Test-images_printed.rar",
             "https://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/DIBCO2009_Test_images-printed.rar"],
    "2010": ["https://users.iit.demokritos.gr/~bgat/H-DIBCO2010/benchmark/H_DIBCO2010_GT.rar",
             "https://users.iit.demokritos.gr/~bgat/H-DIBCO2010/benchmark/H_DIBCO2010_test_images.rar"],
    "2011": ["https://utopia.duth.gr/~ipratika/DIBCO2011/benchmark/dataset/DIBCO11-handwritten.rar",
             "https://utopia.duth.gr/~ipratika/DIBCO2011/benchmark/dataset/DIBCO11-machine_printed.rar"],
    "2012": ["https://utopia.duth.gr/~ipratika/HDIBCO2012/benchmark/dataset/H-DIBCO2012-dataset.rar"],
    "2013": ["https://utopia.duth.gr/~ipratika/DIBCO2013/benchmark/dataset/DIBCO2013-dataset.rar"],
    "2014": ["https://users.iit.demokritos.gr/~bgat/HDIBCO2014/benchmark/dataset/GT.rar",
             "https://users.iit.demokritos.gr/~bgat/HDIBCO2014/benchmark/dataset/original_images.rar"],
    "2016": ["https://vc.ee.duth.gr/h-dibco2016/benchmark/DIBCO2016_dataset-original.zip",
             "https://vc.ee.duth.gr/h-dibco2016/benchmark/DIBCO2016_dataset-GT.zip"],
    "2017": ["https://vc.ee.duth.gr/dibco2017/benchmark/DIBCO2017_Dataset.7z",
             "https://vc.ee.duth.gr/dibco2017/benchmark/DIBCO2017_GT.7z",
             "https://vc.ee.duth.gr/dibco2017/benchmark/EvalWeights.7z"],
}
IMG_EXT = {".bmp", ".png", ".jpg", ".tif", ".tiff"}
# GT file-name suffixes; for 2010/2013/2014 the "_estGT" file is the GT used for
# F-measure (the "_skelGT" files are skeletons, not used).
GT_RE = re.compile(r"^(?P<id>.+?)(_estGT|_GT|_gt)$")


def download(raw: Path):
    for year, urls in URLS.items():
        (raw / year).mkdir(parents=True, exist_ok=True)
        for u in urls:
            dst = raw / year / u.rsplit("/", 1)[1]
            if not dst.exists():
                urllib.request.urlretrieve(u, dst)


def extract(raw: Path, ext: Path):
    for f in sorted(raw.glob("*/*")):
        subprocess.run(["unar", "-q", "-f", "-o", str(ext / f.parent.name), str(f)], check=True)


def doc_type(year, doc_id):
    if year in ("2010", "2012", "2014", "2016"):
        return "handwritten"
    if year == "2017":
        return "unknown"
    return "printed" if re.match(r"^(P|PR)\d", doc_id) else "handwritten"


def organise(ext: Path, out: Path):
    rows = []
    for ydir in sorted(ext.iterdir()):
        year = ydir.name
        files = [p for p in ydir.rglob("*") if p.suffix.lower() in IMG_EXT]
        gts, imgs = {}, {}
        for p in files:
            if "skelGT" in p.stem:
                continue
            m = GT_RE.match(p.stem)
            # DIBCO 2009 GT files have no suffix; they are identified by folder name.
            in_gt_dir = any("GT" in part for part in p.relative_to(ydir).parts[:-1])
            is_gt = bool(m) or in_gt_dir
            doc_id = m.group("id") if m else p.stem
            target = gts if is_gt else imgs
            if doc_id in target:
                raise RuntimeError(f"{year}/{doc_id}: duplicate {target[doc_id]} / {p}")
            target[doc_id] = p
        for doc_id in sorted(imgs):
            if doc_id not in gts:
                raise RuntimeError(f"{year}/{doc_id}: no GT found")
            img = cv2.imread(str(imgs[doc_id]), cv2.IMREAD_UNCHANGED)
            gt = cv2.imread(str(gts[doc_id]), cv2.IMREAD_GRAYSCALE)
            if img.shape[:2] != gt.shape:
                raise RuntimeError(f"{year}/{doc_id}: size mismatch {img.shape} vs {gt.shape}")
            gt = np.where(gt > 0, 255, 0).astype(np.uint8)  # same rule as official code
            for sub in ("img", "gt"):
                (out / year / sub).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / year / "img" / f"{doc_id}.png"), img)
            cv2.imwrite(str(out / year / "gt" / f"{doc_id}.png"), gt)
            rows.append({"year": year, "id": doc_id, "split": "test" if year == "2017" else "dev",
                         "type": doc_type(year, doc_id), "height": gt.shape[0], "width": gt.shape[1],
                         "text_fraction": round(float((gt == 0).mean()), 4),
                         "src_img": str(imgs[doc_id].relative_to(ext)),
                         "src_gt": str(gts[doc_id].relative_to(ext))})
        if year == "2017":
            wdir = out / year / "weights"; wdir.mkdir(exist_ok=True)
            for w in ydir.rglob("*.dat"):
                shutil.copy(w, wdir / w.name)
    with open(out / "manifest.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download"); ap.add_argument("--extracted", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.download:
        download(Path(a.download)); extract(Path(a.download), Path(a.extracted))
    rows = organise(Path(a.extracted), Path(a.out))
    from collections import Counter
    print(Counter((r["year"], r["split"], r["type"]) for r in rows))
