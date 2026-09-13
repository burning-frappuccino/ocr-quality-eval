"""
src/04_人工核对工具.py —— 生成待核对清单 + 核对完一键转换
================================================
两个功能：
    默认运行  → 生成 eval/待核对.csv（带置信度，低置信度优先看）
    加 convert 参数 → 把核对好的文件转成 eval/answers.csv

完整流程：
    1. python src/04_人工核对工具.py              # 生成核对清单
    2. 用 Excel 打开 eval/待核对.csv，对照图片改错
    3. 另存为 eval/核对完成.csv
    4. python src/04_人工核对工具.py convert       # 转成标准答案格式
    5. python src/03_质量评测_CER.py                       # 出真实 CER
"""

import csv
import json
import sys
from pathlib import Path

if sys.version_info >= (3, 7):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "output" / "temp"
EVAL_DIR = ROOT / "tests"
EVAL_DIR.mkdir(exist_ok=True)

SRC_JSON = OUT_DIR / "_全部识别结果.json"
CHECK_CSV = OUT_DIR / "待核对.csv"   # 中间产物，不进版本库
DONE_CSV = ROOT / "tests" / "逐块核对记录.csv"
ANSWERS_CSV = ROOT / "eval" / "answers.csv"

LOW_CONF = 0.8


# ============================================================
# 模式二：把核对结果转成标准答案格式
# ============================================================
def convert():
    print("=" * 66)
    print("把核对结果转成标准答案")
    print("=" * 66)

    src = DONE_CSV if DONE_CSV.exists() else CHECK_CSV
    if not src.exists():
        print(f"\n❌ 找不到 {src.name}")
        sys.exit(1)

    if src == CHECK_CSV:
        print(f"\n⚠️  没找到 {DONE_CSV.name}（你另存的文件名）")
        print(f"   现在读的是 {CHECK_CSV.name} —— 如果你还没核对，结果就等于没核对")

    rows = []
    with open(src, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            fname = (r.get("文件名") or "").strip()
            if not fname:
                continue
            recognized = (r.get("识别结果") or "").strip()
            corrected = (r.get("正确答案") or "").strip()

            # 标记为删除的：这是误检（把图标当文字了），不纳入标准答案
            if corrected == "（删除）":
                continue

            rows.append({
                "文件名": fname,
                "行号": int(r.get("行号") or 0),
                "最终": corrected if corrected else recognized,
                "是否改过": bool(corrected) and corrected != recognized,
                "误检剔除": False,
            })

    if not rows:
        print("❌ 数据为空")
        sys.exit(1)

    # 按文件名分组，拼成全文
    by_file = {}
    for r in rows:
        by_file.setdefault(r["文件名"], []).append(r)

    total_corrected = sum(1 for r in rows if r["是否改过"])

    # 输出标准答案格式
    out_rows = []
    for fname, items in by_file.items():
        items.sort(key=lambda x: x["行号"])
        full_text = "\n".join(i["最终"] for i in items if i["最终"])
        out_rows.append({"文件名": fname, "字段名": "全文", "正确答案": full_text})

    # 写标准答案
    # ⚠️ 如果 answers.csv 正被 Excel/WPS 打开，会写不进去（文件锁），
    #    这时先写临时文件，再尝试替换，失败则提示用户关闭
    tmp_csv = EVAL_DIR / "answers_new.csv"
    with open(tmp_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["文件名", "字段名", "正确答案"])
        writer.writeheader()
        writer.writerows(out_rows)

    target = ANSWERS_CSV
    try:
        if ANSWERS_CSV.exists():
            ANSWERS_CSV.unlink()
        tmp_csv.rename(ANSWERS_CSV)
    except PermissionError:
        target = tmp_csv
        print(f"\n⚠️  {ANSWERS_CSV.name} 正被 Excel / WPS 打开，无法覆盖")
        print(f"    已改写到: {tmp_csv.name}")
        print("    请关闭该文件后重新运行，或手动把 answers_new.csv 改名")

    print(f"\n源文件   : {src.name}")
    print(f"文字块   : {len(rows)} 条")
    print(f"你改过的 : {total_corrected} 条  ({total_corrected/len(rows):.1%})")
    print(f"图片数   : {len(by_file)}")
    print(f"\n已生成   : {target}")

    if total_corrected == 0:
        print("\n⚠️  警告：你一条都没改。")
        print("   如果确实全都识别对了，那没问题；")
        print("   但更可能是你直接另存了、没真正逐条看图核对。")
        print("   标准答案必须人工核对过，否则 CER 是自欺欺人的数字。")
    else:
        print(f"\n✅ 你修正了 {total_corrected} 处错误，这些就是你真实的识别错误。")

    print("\n下一步：运行 src/03_质量评测_CER.py 算出真实 CER")


# ============================================================
# 模式一：生成核对清单
# ============================================================
def make_checklist():
    print("=" * 66)
    print("人工核对清单生成器")
    print("=" * 66)

    if not SRC_JSON.exists():
        print(f"\n❌ 找不到 {SRC_JSON.name}")
        print("   请先运行 _识别新图.py 生成识别结果")
        sys.exit(1)

    with open(SRC_JSON, "r", encoding="utf-8") as f:
        results = json.load(f)

    rows = []
    for fname, data in results.items():
        blocks = data.get("逐块结果")
        if blocks:
            for i, b in enumerate(blocks, 1):
                rows.append({
                    "文件名": fname,
                    "行号": i,
                    "置信度": b["置信度"],
                    "识别结果": b["文字"],
                    "正确答案": "",
                    "备注": "",
                })
        else:
            for i, line in enumerate(
                    [l for l in data["文本"].split("\n") if l.strip()], 1):
                rows.append({
                    "文件名": fname, "行号": i, "置信度": "",
                    "识别结果": line, "正确答案": "", "备注": "",
                })

    with open(CHECK_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["文件名", "行号", "置信度", "识别结果", "正确答案", "备注"])
        writer.writeheader()
        writer.writerows(rows)

    # ---- 统计 ----
    print(f"\n共 {len(rows)} 个文字块，覆盖 {len(results)} 张图片\n")
    print(f"  {'图片':<44}{'文字块':>7}{'平均置信度':>11}{'低置信度':>9}")
    print("  " + "─" * 70)

    for fname, data in results.items():
        n = len([l for l in data["文本"].split("\n") if l.strip()])
        blocks = data.get("逐块结果", [])
        low = sum(1 for b in blocks if b["置信度"] < LOW_CONF)
        print(f"  {fname[:42]:<44}{n:>7}{data['平均置信度']:>10.1%}{low:>9}")

    # ---- 按置信度挑出最可疑的 ----
    suspicious = sorted(
        [r for r in rows if r["置信度"] != "" and r["置信度"] < LOW_CONF],
        key=lambda x: x["置信度"]
    )

    print(f"\n{'=' * 66}")
    print(f"优先核对清单 —— 置信度低于 {LOW_CONF} 的 {len(suspicious)} 个块")
    print(f"{'=' * 66}")
    print("这些是最可能识别错的，建议先核对这些：\n")

    for i, r in enumerate(suspicious, 1):
        name = r["文件名"][:16]
        print(f"  {i:>2}. [{name}..] 置信度 {r['置信度']:.3f}   「{r['识别结果']}」")

    print(f"""
{'=' * 66}
【核对步骤】

1. 用 Excel / WPS 打开：eval/待核对.csv
2. 【必须】同时打开 images 文件夹里的图片对照着看
3. 逐个文字块检查：
     ✅ 识别对了  → "正确答案"列留空
     ❌ 识别错了  → 在"正确答案"列填正确内容
4. 核对完另存为：eval/核对完成.csv （格式选 CSV UTF-8）
5. 运行：python src/04_人工核对工具.py convert
6. 运行：python src/03_质量评测_CER.py

【效率建议】
   - 先核对上面那 {len(suspicious)} 个低置信度的，它们错得最多
   - 一张图核对完再换下一张，别来回切换
   - 剩下 {len(rows) - len(suspicious)} 个高置信度的大多是对的，但你仍要看一眼
     （⚠️ 高置信度 ≠ 正确，模型认错字时也可能很自信）

【为什么必须做这一步】
  这是整个项目唯一"机器替代不了"的环节。
  也正因如此，你的评测数据才是别人抄不走的资产。
{'=' * 66}
""")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "convert":
        convert()
    else:
        make_checklist()
