"""
修复答案文件.py —— 合并重复的答案文件
================================================
背景：
    excel/WPS 打开了 eval/answers.csv 并持有文件锁，
    导致核对后的最终版只能存成 eval/answers_new.csv。
    仓库里因此有两个答案文件，看起来会让人困惑。

这个脚本做的事：
    把 answers_new.csv（最终核对版）替换掉 answers.csv（旧版），
    并删掉重复文件，只保留一个 eval/answers.csv。

使用前提：
    ⚠️ 先关闭 WPS / Excel 里打开的这个 CSV 文件

运行方式：双击本文件，或在 PyCharm 里运行
"""

import subprocess
import sys
from pathlib import Path

if sys.version_info >= (3, 7):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
EVAL = ROOT / "eval"
OLD = EVAL / "answers.csv"        # 旧版（未核对）
NEW = EVAL / "answers_new.csv"    # 最终版（已核对）

print("=" * 62)
print("修复答案文件")
print("=" * 62)

if not NEW.exists():
    print("\n✅ 已经处理过了，无需重复运行")
    print(f"   当前只有一个答案文件: {OLD.name} ({(OLD.stat().st_size if OLD.exists() else 0)} 字节)")
    sys.exit(0)

if not OLD.exists():
    print("\n⚠️  answers.csv 不存在，直接重命名")
    NEW.rename(OLD)
    print(f"✅ 已重命名为 answers.csv")
    sys.exit(0)

print(f"\n旧版 answers.csv     : {OLD.stat().st_size} 字节（未经人工核对）")
print(f"最终版 answers_new.csv: {NEW.stat().st_size} 字节（已逐条核对）")

# 检查是否还被锁
locked = False
try:
    OLD.unlink()
except PermissionError:
    locked = True

if locked:
    print("""
❌ answers.csv 仍被 WPS / Excel 占用，无法替换。

请这样做：
    1. 切到 WPS / Excel 窗口
    2. 关闭 answers.csv 这个文件（其他文件可以留着）
    3. 重新运行本脚本

如果找不到是哪个文件占用了，重启一次 WPS 最省事。

（临时办法：评测脚本会自动优先读取更新的那个文件，
  所以现在运行 src/03_质量评测_CER.py 结果也是正确的。）
""")
    sys.exit(1)

# 删除成功，把最终版改名
NEW.rename(OLD)
print(f"\n✅ 已替换为最终版: answers.csv ({OLD.stat().st_size} 字节)")
print("✅ 已删除重复文件: answers_new.csv")

# 提交到 git
print("\n" + "=" * 62)
print("提交到 GitHub")
print("=" * 62)


def git(*args):
    r = subprocess.run(("git",) + args, cwd=str(ROOT), capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    return r.returncode, (r.stdout or "") + (r.stderr or "")


code, out = git("add", "-A", "eval")
if code != 0:
    print(f"⚠️  git add 失败: {out.strip()[:200]}")
    print("    你可以手动执行: git add -A && git commit -m '...' && git push")
    sys.exit(1)

code, out = git("commit", "-m", "chore: 合并重复的答案文件，只保留核对后的最终版")
print(out.strip()[:300] if out.strip() else "（无变更需要提交）")

code, out = git("push")
if code == 0:
    print("\n✅ 已推送到 GitHub")
    print("   https://github.com/burning-frappuccino/ocr-quality-eval")
else:
    print(f"\n⚠️  推送失败: {out.strip()[:300]}")
    print("    可以手动执行: git push")

print("\n" + "=" * 62)
print("完成！现在仓库里只有一个答案文件了。")
print("=" * 62)
