import os, sys, json, hashlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import cv2
import numpy as np
from pathlib import Path
from paddleocr import PaddleOCR

ROOT = Path(__file__).parent.parent
IMG_DIR = ROOT / "images"
OUT = ROOT / "output" / "temp"
OUT.mkdir(parents=True, exist_ok=True)


def imread_cn(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None


def phash(img, size=8):
    """感知哈希：和图的内容有关，和文件大小无关。用来判断两张图是不是同一张"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    avg = g.mean()
    bits = (g > avg).flatten()
    return "".join("1" if b else "0" for b in bits)


def hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


# 先看图片尺寸和哈希
print("=" * 70)
print("第一步：检查图片是否重复")
print("=" * 70)
infos = {}
for p in sorted(IMG_DIR.iterdir()):
    if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        continue
    img = imread_cn(p)
    if img is None:
        print(f"  {p.name}  读图失败")
        continue
    h, w = img.shape[:2]
    ph = phash(img)
    infos[p.name] = (w, h, ph)
    print(f"  {p.name[:36]:<38} {w:>5}x{h:<5}")

names = list(infos.keys())
print("\n  图片相似度对比（汉明距离，0=完全相同，<10 视为重复）：")
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        d = hamming(infos[names[i]][2], infos[names[j]][2])
        flag = "  ← 疑似重复!" if d <= 10 else ""
        print(f"    {names[i][:20]:<22} vs {names[j][:20]:<22} 距离 {d:>3}{flag}")

# OCR 识别所有图
print("\n" + "=" * 70)
print("第二步：识别所有图片")
print("=" * 70)

ocr = PaddleOCR(
    lang="ch",
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False, use_doc_unwarping=False,
    use_textline_orientation=False, enable_mkldnn=False,
)

results = {}
for p in sorted(IMG_DIR.iterdir()):
    if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        continue
    print(f"\n{'─' * 70}")
    print(f"  {p.name}")
    print(f"{'─' * 70}")
    res = ocr.predict(str(p))
    d = dict(res[0])
    texts = d["rec_texts"]
    scores = [float(s) for s in d["rec_scores"]]

    order = sorted(range(len(texts)), key=lambda i: d["rec_polys"][i][0][1])
    ordered_texts = [texts[i] for i in order]
    ordered_scores = [scores[i] for i in order]

    print(f"  文字块 {len(texts)}   平均置信度 {sum(scores)/len(scores):.1%}")
    print()
    for t, s in zip(ordered_texts, ordered_scores):
        mark = "  ← 低" if s < 0.8 else ""
        print(f"    {s:.3f}  {t}{mark}")

    results[p.name] = {
        "文字块数": len(texts),
        "平均置信度": round(sum(scores) / len(scores), 4),
        "文本": "\n".join(ordered_texts),
        "逐块结果": [
            {"文字": t, "置信度": round(s, 4)}
            for t, s in zip(ordered_texts, ordered_scores)
        ],
    }

    # 存一份给人工核对用
    (OUT / f"{p.stem}_待核对.txt").write_text(
        "\n".join(ordered_texts), encoding="utf-8")

with open(OUT / "_全部识别结果.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 70)
print("已保存 *_待核对.txt，供人工逐字核对")
print("=" * 70)
