"""
src/00_单图识别入门.py —— 让电脑读出图片里的字
================================================
这是你的第一个真正的 AI 项目。

运行方式：在 PyCharm 里打开本文件，按 Ctrl + Shift + F10

环境已装好：paddlepaddle 3.3.1 + paddleocr 3.7.0
模型已下载：PP-OCRv5_mobile（检测 + 识别），所以第一次运行就很快
"""

import json
import sys
import time
from pathlib import Path

# 让中文在 Windows 终端正常显示
if sys.version_info >= (3, 7):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WORK_DIR = Path(__file__).parent
IMG_DIR = WORK_DIR / "images"
IMG_DIR.mkdir(exist_ok=True)

print("=" * 62)
print("单图 OCR 识别入门")
print("=" * 62)


# ============================================================
# 第 1 步：加载 OCR 模型
# ============================================================
print("\n[1/5] 加载 OCR 模型...")

from paddleocr import PaddleOCR

t0 = time.time()
ocr = PaddleOCR(
    lang="ch",                                    # 识别中文（英文数字也能认）
    text_detection_model_name="PP-OCRv5_mobile_det",    # 检测模型：找出字在哪
    text_recognition_model_name="PP-OCRv5_mobile_rec",  # 识别模型：认出是什么字
    use_doc_orientation_classify=False,           # 不做整页方向分类（快一些）
    use_doc_unwarping=False,                      # 不做文档展平（快一些）
    use_textline_orientation=False,               # 不做单行方向判断（快一些）
    enable_mkldnn=False,                          # ⚠️ 关键！见文件末尾说明
)
print(f"      模型加载完成（{time.time() - t0:.1f} 秒）")


# ============================================================
# 第 2 步：造一张测试图片
# ============================================================
print("\n[2/5] 生成测试图片...")

from PIL import Image, ImageDraw, ImageFont

# 找系统里的中文字体
font = None
for fp in [
    r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑
    r"C:\Windows\Fonts\simhei.ttf",    # 黑体
    r"C:\Windows\Fonts\simsun.ttc",    # 宋体
]:
    if Path(fp).exists():
        try:
            font = ImageFont.truetype(fp, 36)
            break
        except Exception:
            continue

if font is None:
    font = ImageFont.load_default()
    print("      ⚠️ 没找到中文字体，改用默认字体")

# 画一张白底黑字的"合同"图片
img = Image.new("RGB", (700, 320), color="white")
draw = ImageDraw.Draw(img)

lines = [
    "北京市朝阳区建国路88号",
    "合同编号：HT-2026-0315",
    "签订日期：2026年3月15日",
    "金额：￥128,000.00元",
    "甲方：北京某某科技有限公司",
]

for i, line in enumerate(lines):
    draw.text((40, 30 + i * 56), line, fill="black", font=font)

test_img_path = IMG_DIR / "测试.jpg"
img.save(test_img_path)
print(f"      已生成: {test_img_path.name}（{img.width}×{img.height}）")


# ============================================================
# 第 3 步：识别
# ============================================================
print("\n[3/5] 开始识别...")

t1 = time.time()
results = ocr.predict(str(test_img_path))
elapsed = time.time() - t1

# ⚠️ 新版 PaddleOCR 的结果是一个"字典式对象"，直接当字典用
res = results[0]
data = dict(res)

texts = data["rec_texts"]        # 识别出的文字，列表
scores = data["rec_scores"]      # 每条文字的置信度，列表
polys = data["rec_polys"]        # 每条文字的四角坐标，列表

print(f"      识别完成，耗时 {elapsed:.2f} 秒")
print("\n" + "-" * 62)
print("AI 吐出来的原始结果（这就是真实的模型输出）")
print("-" * 62)

for i, (text, score, poly) in enumerate(zip(texts, scores, polys)):
    x = int(poly[0][0])
    y = int(poly[0][1])
    print(f"  #{i}  位置({x:>4},{y:>4})  置信度 {score:.3f}   {text}")

print("-" * 62)
print(f"  共识别到 {len(texts)} 个文字块")
print(f"  结果对象里的字段: {list(data.keys())[:8]} ...")


# ============================================================
# 第 4 步：整理成人能看的样子
# ============================================================
print("\n[4/5] 整理结果...")

# 按 y 坐标从上到下排序（AI 返回的顺序不保证是从上到下的）
items = sorted(
    zip(texts, scores, polys),
    key=lambda item: item[2][0][1]      # item[2] 是坐标，[0][1] 是左上角 y
)

print("\n识别结果（已按从上到下排序）：")
for i, (text, score, poly) in enumerate(items, 1):
    flag = "" if score >= 0.8 else "   ← 置信度低，需要人工复核"
    print(f"  {i}. {text}   ({score:.1%}){flag}")

whole_text = "\n".join(t for t, s, p in items)
print("\n完整文本：")
print("-" * 62)
print(whole_text)
print("-" * 62)


# ============================================================
# 第 5 步：统计 + 存文件
# ============================================================
print("\n[5/5] 统计与保存...")

avg_conf = sum(scores) / len(scores) if scores else 0
low_conf = [(t, s) for t, s, p in items if s < 0.8]

print(f"  文字块数量: {len(texts)}")
print(f"  平均置信度: {avg_conf:.1%}")
print(f"  低置信度块: {len(low_conf)} 个")
if low_conf:
    for t, s in low_conf:
        print(f"      ! {t} ({s:.1%})")

out_data = {
    "图片": test_img_path.name,
    "文字块数量": len(texts),
    "平均置信度": round(avg_conf, 4),
    "耗时秒": round(elapsed, 3),
    "文本": whole_text,
    "详情": [
        {
            "文字": t,
            "置信度": round(float(s), 4),
            "位置": [[int(px), int(py)] for px, py in p],
        }
        for t, s, p in items
    ],
}

json_path = IMG_DIR / "识别结果.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(out_data, f, ensure_ascii=False, indent=2)

print(f"  已保存: {json_path.name}")


# ============================================================
# 完成
# ============================================================
print("\n" + "=" * 62)
print("恭喜！你刚刚跑通了完整的 OCR 流程：")
print("  图片 → 检测文字位置 → 识别文字 → 算置信度 → 结构化输出")
print("=" * 62)


# ============================================================
# 【你的作业】
# ============================================================
"""
作业 1（简单）：换成你自己的图片
    找一张带字的照片（书页、发票、说明书、微信截图都行）放进 images 文件夹，
    把第 2 步整段删掉，改成直接指向你的图片：

        test_img_path = IMG_DIR / "你的图片名.jpg"

    然后重新运行，看识别结果。真实照片的准确率通常比上面这张"干净图"低，
    亲眼看到这个差距，你就理解第 2 阶段为什么要做图像预处理了。

作业 2（中等）：批量处理一整个文件夹
    在文件末尾加上：

        for p in IMG_DIR.glob("*.jpg"):
            print("处理:", p.name)
            r = dict(ocr.predict(str(p))[0])
            for t, s in zip(r["rec_texts"], r["rec_scores"]):
                print(f"   {t}  ({s:.2f})")

作业 3（进阶）：图像预处理对比 —— 这是面试加分项
    OCR 最大的坑是"直接丢原图"。加上预处理再对比：

        import cv2
        img = cv2.imread(str(path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite("处理后.png", binary)

    分别跑"原图"和"处理后"，把两次的【平均置信度】记下来对比。
    把这个数字写进 README，就是你第一个量化成果。

作业 4（挑战）：倾斜校正
    拍一张故意拍歪的照片，用 cv2.minAreaRect + cv2.warpAffine 把它摆正，
    对比校正前后的识别准确率。这是 OCR 系统里最值钱的一步优化。
"""


# ============================================================
# 【踩坑记录 —— 这段请务必看，面试可以直接讲】
# ============================================================
"""
问题：装好 PaddleOCR 后一运行就报错：

    NotImplementedError: (Unimplemented) ConvertPirAttribute2RuntimeAttribute
    not support [pir::ArrayAttribute<pir::DoubleAttribute>]
    (at onednn_instruction.cc:118)

原因：这是 PaddlePaddle 3.3.x 的已知 Bug（上游 issue #77340）。
      新版飞桨默认开启 oneDNN（CPU 加速库），但它在处理某些参数类型时有缺陷，
      导致 CPU 推理直接崩溃。

排查过程（这才是重点）：
    1. 看报错最后一行 → 定位到 onednn_instruction.cc，说明是 oneDNN 的问题
    2. 而不是去乱改自己的代码
    3. 搜报错关键字 → 确认是上游 bug，不是我用错了 API
    4. 想办法关掉 oneDNN

试过的三种方案：
    ❌ 环境变量 FLAGS_use_mkldnn=0        → 无效，飞桨 3.x 不认这个变量了
    ✅ 构造参数 enable_mkldnn=False       → 成功！
    ❌ 构造参数 device="cpu"              → 无效，它本来就跑在 CPU 上

结论：PaddleOCR(enable_mkldnn=False)

顺便说，新版 PaddleOCR 3.x 的 API 和老教程完全不一样，也踩了坑：
    老写法（网上教程都是这个，会直接报错）：
        ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
        result = ocr.ocr(img_path, cls=True)
        # 结果是嵌套列表 [[[坐标], (文字, 置信度)], ...]

    新写法（3.7 版正确用法）：
        ocr = PaddleOCR(lang="ch", use_textline_orientation=False, enable_mkldnn=False)
        result = ocr.predict(img_path)
        # 结果是字典式对象，取 dict(result[0])["rec_texts"] / ["rec_scores"]

    这就是为什么我说：中文教程只用来建立概念，具体代码必须看官方文档或自己实测。
"""
