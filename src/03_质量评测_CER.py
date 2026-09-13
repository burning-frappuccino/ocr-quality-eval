"""
src/03_质量评测_CER.py —— 给 OCR 打分
================================================
这个脚本是整个项目的核心。它回答三个问题：
    1. 识别准不准？      → CER 字符错误率
    2. 关键字段抽对没？  → 字段准确率
    3. 多少要人工复核？  → 复核率

用法：
    1. 先把标准答案整理进 eval/answers.csv
    2. 运行本脚本，它会读 images/ 里的图，和标准答案对照打分

运行：Ctrl + Shift + F10
"""

import csv
import json
import sys
import time
from pathlib import Path

if sys.version_info >= (3, 7):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
IMG_DIR = ROOT / "images"
EVAL_DIR = ROOT / "eval"
OUT_DIR = ROOT / "output"
EVAL_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

ANSWERS_CSV = EVAL_DIR / "answers.csv"
REVIEW_CSV = OUT_DIR / "人工复核清单.csv"
REPORT_JSON = OUT_DIR / "评测报告.json"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LOW_CONF_THRESHOLD = 0.8      # 置信度低于这个值就送人工复核


# ============================================================
# 核心算法：编辑距离（Levenshtein Distance）
# ============================================================
def edit_distance(s1, s2):
    """算两个字符串"最少改几步能变成一样"

    允许三种操作，每种算 1 步：
        替换：把 a 改成 b
        删除：删掉一个字符
        插入：插入一个字符

    例：
        "北京市朝阳区建国路88号"  →  "北京市朝阳区建国路8号"
        需要"删除一个 8"，距离 = 1

    实现方式：动态规划（这是算法面试常客）
        建一个表格 dp[i][j] 表示
        "s1 的前 i 个字符" 变成 "s2 的前 j 个字符" 需要几步
    """
    n, m = len(s1), len(s2)

    # dp[i][j] = 把 s1[:i] 变成 s2[:j] 的最少步数
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # 边界：从空串变成 s2 的前 j 个字符，需要插入 j 次
    for j in range(m + 1):
        dp[0][j] = j
    # 从 s1 的前 i 个字符变成空串，需要删除 i 次
    for i in range(n + 1):
        dp[i][0] = i

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                # 这个字符相同，不用操作，沿用左上角
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # 三种操作取最小：替换 / 删除 / 插入
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],    # 替换
                    dp[i - 1][j],        # 删除
                    dp[i][j - 1],        # 插入
                )
    return dp[n][m]


def cer(predicted, truth):
    """字符错误率 CER = 编辑距离 / 标准答案长度"""
    if not truth:
        return 0.0 if not predicted else 1.0
    return edit_distance(predicted, truth) / len(truth)


def normalize(text):
    """把文本规范化后再比较，避免"空格/换行不同"这种无关差异拉低分数"""
    if text is None:
        return ""
    # 去掉所有空白字符（空格、换行、制表符）
    return "".join(str(text).split())


# ============================================================
# 读取标准答案
# ============================================================
print("=" * 68)
print("OCR 评测")
print("=" * 68)

if not ANSWERS_CSV.exists():
    print(f"\n❌ 找不到标准答案文件: {ANSWERS_CSV}")
    print("""
请先按下面格式创建 eval/answers.csv：

    文件名,字段名,正确答案
    测试.jpg,全文,"北京市朝阳区建国路88号
    合同编号：HT-2026-0315
    签订日期：2026年3月15日"
    测试.jpg,合同编号,HT-2026-0315
    测试.jpg,金额,￥128,000.00元

说明：
    字段名 = "全文" 的行是整篇文字的正确答案（用来算 CER）
    其他字段名的行是关键字段（用来算字段准确率）
""")
    sys.exit(0)

# 读 CSV，按文件名分组
truth_data = {}      # {文件名: {"全文": "...", "字段": {字段名: 答案}}}
with open(ANSWERS_CSV, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        fname = (row.get("文件名") or "").strip()
        field = (row.get("字段名") or "").strip()
        answer = row.get("正确答案") or ""
        if not fname or not field:
            continue
        truth_data.setdefault(fname, {"全文": "", "字段": {}})
        if field == "全文":
            truth_data[fname]["全文"] = answer
        else:
            truth_data[fname]["字段"][field] = answer

if not truth_data:
    print("❌ answers.csv 里没有有效数据（检查表头是不是：文件名,字段名,正确答案）")
    sys.exit(0)

print(f"\n标准答案: 覆盖 {len(truth_data)} 张图片")
for fname, d in truth_data.items():
    print(f"    {fname:<34} 全文 {len(d['全文'])} 字, 字段 {len(d['字段'])} 个")


# ============================================================
# 加载 OCR
# ============================================================
print("\n加载 OCR 模型...")

from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang="ch",
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,
)
print("    完成")


# ============================================================
# 逐张评测
# ============================================================
print("\n" + "=" * 68)
print("开始评测")
print("=" * 68)

report = []
review_rows = []

for fname, truth in truth_data.items():
    img_path = IMG_DIR / fname
    if not img_path.exists():
        print(f"\n⚠️  跳过 {fname}：images 文件夹里找不到这张图")
        continue

    print(f"\n{'─' * 68}")
    print(f"  {fname}")
    print(f"{'─' * 68}")

    t0 = time.time()
    try:
        res = ocr.predict(str(img_path))
    except Exception as e:
        print(f"  ❌ 识别失败: {e}")
        continue
    elapsed = time.time() - t0

    data = dict(res[0])
    texts = data["rec_texts"]
    scores = [float(s) for s in data["rec_scores"]]

    # 按从上到下排序后拼成整篇文本
    order = sorted(range(len(texts)), key=lambda i: data["rec_polys"][i][0][1])
    predicted_text = "\n".join(texts[i] for i in order)

    # ---- 指标 1：CER ----
    truth_text = truth["全文"]
    if truth_text:
        c = cer(normalize(predicted_text), normalize(truth_text))
        n_truth = len(normalize(truth_text))
        n_pred = len(normalize(predicted_text))
        dist = edit_distance(normalize(predicted_text), normalize(truth_text))

        print(f"  字符错误率 CER : {c:.2%}")
        print(f"      标准答案 {n_truth} 字 → 识别 {n_pred} 字")
        print(f"      编辑距离 = {dist}")

        # ⚠️ 评测有效性自检 —— 这个检查非常重要
        if n_truth == 0 or n_pred == 0:
            print("      ⚠️ 警告：文本为空，CER 不可信")
        elif abs(n_truth - n_pred) / max(n_truth, n_pred) < 0.02 and dist > n_truth * 0.10:
            print("      ⚠️ 警告：标准答案和识别结果长度几乎一样，但差异很大")
            print("         → 很可能你把【模型输出】直接当标准答案了，")
            print("           那 99% 的差异只是换行/空格不同，CER 是假的。")
            print("         → 标准答案必须【人工逐字核对】过才有意义。")
    else:
        c = None
        print("  字符错误率 CER : (未提供全文标准答案，跳过)")

    # ---- 指标 2：字段准确率 ----
    field_total = len(truth["字段"])
    field_ok = 0
    field_detail = []

    if field_total:
        print(f"\n  字段抽取（共 {field_total} 个）:")
        for fname_, answer in truth["字段"].items():
            # 简单策略：在识别结果里找包含答案的那一行
            target = normalize(answer)
            hit = any(target in normalize(line) for line in predicted_text.split("\n"))
            # 再宽松一点：整篇文本里能不能找到
            if not hit:
                hit = target in normalize(predicted_text)
            if hit:
                field_ok += 1
                print(f"      ✅ {fname_:<12} {answer}")
            else:
                print(f"      ❌ {fname_:<12} {answer}   (未在识别结果中找到)")
            field_detail.append({"字段": fname_, "答案": answer, "命中": hit})

        acc = field_ok / field_total
        print(f"\n  字段准确率: {acc:.1%}  ({field_ok}/{field_total})")
    else:
        acc = None
        print("\n  字段抽取: (未提供字段标准答案，跳过)")

    # ---- 指标 3：复核率 ----
    low_idx = [i for i, s in enumerate(scores) if s < LOW_CONF_THRESHOLD]
    review_rate = len(low_idx) / len(scores) if scores else 0
    print(f"\n  人工复核率: {review_rate:.1%}  ({len(low_idx)}/{len(scores)} 个文字块置信度 < {LOW_CONF_THRESHOLD})")

    for i in low_idx:
        review_rows.append({
            "文件名": fname,
            "文字": texts[i],
            "置信度": round(scores[i], 4),
            "原因": "置信度低于阈值",
        })

    avg_conf = sum(scores) / len(scores) if scores else 0
    print(f"  平均置信度: {avg_conf:.1%}    耗时: {elapsed:.2f}s")

    report.append({
        "文件名": fname,
        "CER": round(c, 4) if c is not None else None,
        "字段准确率": round(acc, 4) if acc is not None else None,
        "字段明细": field_detail,
        "人工复核率": round(review_rate, 4),
        "平均置信度": round(avg_conf, 4),
        "文字块数": len(texts),
        "耗时秒": round(elapsed, 2),
    })


# ============================================================
# 总结报告
# ============================================================
print("\n" + "=" * 68)
print("总结")
print("=" * 68)

if report:
    # ---- 明细表 ----
    print(f"\n  {'文件':<30}{'CER':>9}{'字段准确率':>11}{'复核率':>9}{'耗时':>8}")
    print("  " + "─" * 66)
    for r in report:
        cer_s = f"{r['CER']:.2%}" if r["CER"] is not None else "—"
        acc_s = f"{r['字段准确率']:.1%}" if r["字段准确率"] is not None else "—"
        name = r["文件名"] if len(r["文件名"]) <= 28 else r["文件名"][:25] + "..."
        print(f"  {name:<30}{cer_s:>9}{acc_s:>11}{r['人工复核率']:>8.1%}{r['耗时秒']:>7.2f}s")

    # ---- 汇总 ----
    cers = [r["CER"] for r in report if r["CER"] is not None]
    accs = [r["字段准确率"] for r in report if r["字段准确率"] is not None]
    rates = [r["人工复核率"] for r in report]

    print("\n  " + "─" * 40)
    if cers:
        print(f"  平均 CER       : {sum(cers) / len(cers):.2%}")
    if accs:
        print(f"  平均字段准确率 : {sum(accs) / len(accs):.1%}")
    print(f"  平均复核率     : {sum(rates) / len(rates):.1%}")
    print(f"  图片总数       : {len(report)}")

    # ---- 存 JSON 报告 ----
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "图片数": len(report),
            "平均CER": round(sum(cers) / len(cers), 4) if cers else None,
            "平均字段准确率": round(sum(accs) / len(accs), 4) if accs else None,
            "平均复核率": round(sum(rates) / len(rates), 4),
            "明细": report,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  报告已保存: {REPORT_JSON.name}")

# ---- 存人工复核清单 ----
if review_rows:
    with open(REVIEW_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["文件名", "文字", "置信度", "原因"])
        writer.writeheader()
        writer.writerows(review_rows)
    print(f"  人工复核清单已保存: {REVIEW_CSV.name}（{len(review_rows)} 条）")
    print("     ↑ 这就是真实 OCR 系统的『转人工队列』，业务价值最高的一环")
else:
    print("\n  没有低置信度的文字块，不需要人工复核")

print("\n" + "=" * 68)


# ============================================================
# 【评测的三个大坑 —— 这段比代码重要】
# ============================================================
"""
坑 1：拿模型输出当标准答案（最常见，最致命）
    我第一版就是这么干的：把 ocr 识别的结果复制进 answers.csv。
    结果 CER 算出来是 17%，但里面 99% 的差异只是换行符不同，
    真实的识别错误一个都没测出来 —— 等于没测。

    → 标准答案必须【人工逐字核对】。
    → 判断方法：如果"标准答案长度"和"识别结果长度"完全一样，
       那你多半是复制粘贴的，不是人工核对的。

坑 2：标准答案不完整
    我第二版只抄了 30 行干净文字（126 字），
    但模型实际识别出 75 个文字块（375 字）。
    结果 CER = 200%（因为"多出来的字"全被算成插入错误）。

    → 标准答案必须覆盖图上【所有】文字，一个字都不能漏。
    → 否则 CER 会虚高得离谱。

坑 3：只看置信度，不看 CER
    置信度是模型自己说的"我有多确定"，它会撒谎 ——
    模型认错字的时候也可能给出 0.98 的高置信度。
    上面那个 "藏猫庵藏猫庵cockt露照相馆" 置信度 0.98，
    但明显是多个店铺名粘连错了。

    → 置信度只能用来"分流人工复核"，
       衡量准不准必须用 CER（对比人工标准答案）。
    → 面试时讲这个区别，能直接区分出你懂不懂评测。

【正确的评测流程】

    1. 挑 15-30 张真实图片
    2. 跑一遍识别，导出结果（这一步可以省掉人工打字的功夫）
    3. 【人工逐字核对】每一张，把错的改对 —— 这一步不能省
    4. 存成 answers.csv
    5. 之后每次改代码，都拿这份"考卷"重跑，看 CER 变化

    评测集就是你的"考卷"。考卷是假的，分数就没意义。
"""


# ============================================================
# 【你的作业】
# ============================================================
r"""
作业 1：造你自己的评测集（最重要，必须做）
    1. 往 images/ 里放 10-20 张不同类型的图
    2. 先跑一次批量识别（src/01_批量识别.py），把结果复制出来
    3. 【人工逐字核对】每一张，错的改对，填进 eval/answers.csv
    4. 重新运行本脚本，看你的真实 CER

    标准：CER < 10% 算不错，< 5% 算优秀（印刷体文档）

作业 2：对比不同方案在同一评测集上的表现
    把预处理实验（research/预处理消融实验.py）接入评测流程，
    看哪个方案的 CER 最低。
    ⚠️ 注意：不要再用"平均置信度"当唯一指标了，
        CER 才是能写进简历的硬指标。

作业 3：优化字段抽取
    现在的字段抽取用的是"字符串包含"这种笨办法。
    试试改成正规做法：
      - 用正则表达式匹配"合同编号[：:]\s*(\S+)"
      - 或者让大模型从识别文本里抽字段（后面学 RAG 时再做）

作业 4：挑战 —— 找出 CER 最高的那张图
    逐字对比标准答案和识别结果，找出错在哪里，
    分析是哪类错误（形近字？符号？粘连？），
    写进 README 的"失败模式分析"。
"""
