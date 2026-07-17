"""reference 静态扫描 CLI：跑 A100 前扫 cases/<case>/reference.py + config.py 的弱 baseline 危险写法。

用法：
    python skill/scripts/check_reference.py --case rbf

纯静态（不 import torch / 不跑 GPU，只读源码正则匹配），命中危险模式打印 WARN，末行输出机读
    REF_CHECK=CLEAN            无命中
    REF_CHECK=WARN n=<数量>    有 n 条疑似弱 baseline / 红线写法

定位：弱 baseline 的源头在 reference 写法（for / T^2 Toeplitz / 规模分支 / cumprod 脆弱 / 挑输入分布 /
高层算子落回）。auto-scale 管"规模不够"，本脚本管"写法可疑"——两者正交互补，在跑 A100 前就预警省算力。
默认只 WARN 不拦（因有合法例外，如变系数递推的 O(T^2) 下三角）；--strict 时有命中则 exit 1。

判据依据见约定文件"必须向量化"章节与 memory project-baseline-traps 的弱 baseline 三/四变种。
"""

import sys
import os
import re
import argparse


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


# 去掉 docstring/字符串字面量 + 行内注释，避免注释/说明文字里的字样误报
# （如 layernorm docstring 写"不落回 F.layer_norm"、codex 注释"cumprod underflows..."）
def _strip_comments(src):
    # 1. 去三引号块（docstring 及多行字符串）
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"'''[\s\S]*?'''", "", src)
    out = []
    for line in src.splitlines():
        # 2. 去单行字符串字面量（"..." / '...'）——避免字符串里的字样误报
        line = re.sub(r'"[^"]*"', '""', line)
        line = re.sub(r"'[^']*'", "''", line)
        # 3. 去行内 # 注释
        if "#" in line:
            line = line[: line.index("#")]
        out.append(line)
    return "\n".join(out)


def scan_reference(src):
    """返回 [(severity, code, msg), ...]。severity: RED(红线)/WARN(疑似弱baseline)。"""
    findings = []
    code = _strip_comments(src)

    # 1. 红线§1：高层融合算子落回
    for pat, name in [
        (r"scaled_dot_product_attention", "F.scaled_dot_product_attention"),
        (r"F\.(softmax|layer_norm|conv1d|conv2d|linear|multi_head|cross_entropy|scaled)", "F.* 高层算子"),
        (r"nn\.functional\.", "nn.functional.*"),
        (r"nn\.(MultiheadAttention|Conv1d|Conv2d|Linear|LayerNorm)", "nn.* 高层模块"),
    ]:
        if re.search(pat, code):
            findings.append(("RED", "hi-level-op", f"疑似落回高层算子（{name}）——红线§1 禁止；reference 须用基础算子表达"))

    # 2. Python for 遍历张量维度（弱 baseline 第一变种 / 或变系数递推合法例外）
    if re.search(r"\bfor\b[^\n]{0,40}\brange\s*\(", code):
        findings.append(("WARN", "python-for",
                         "reference 出现 for...range 循环——若遍历张量维度(B/T/H/C)是弱 baseline("
                         "torch.compile 编译爆炸+eager 畸形慢)；仅变系数递推(门控 SSM)无稳定 O(N) 时才合法退 for/T^2"))

    # 3. O(T^2) Toeplitz 伪向量化：tril/triu 配 einsum（scan/递推矩阵化的特征签名，如 linear_ssm
    #    的 einsum('tk,bkc->btc')）。注意排除 attention 类：其 matmul/bmm(Q·K^T) 是算法固有 O(T^2)、
    #    不是 scan 伪向量化，故只认 einsum、不认 matmul/bmm（避免 attention 误报）。
    if re.search(r"\b(tril|triu)\b", code) and re.search(r"\beinsum\b", code):
        findings.append(("WARN", "tri-toeplitz",
                         "reference 同时用 tril/triu + einsum——疑似 O(T^2) 密集矩阵伪向量化(把 O(N) scan 恶化成 O(N^2))；"
                         "固定系数递推应用 cumsum；仅变系数递推数值必需时 O(T^2) 下三角合法"))

    # 4. 规模/条件专属分支：if 里用 numel/shape/size 选实现
    for m in re.finditer(r"\bif\b[^\n:]{0,80}", code):
        seg = m.group(0)
        if re.search(r"\.(numel|nelement)\s*\(", seg) or re.search(r"\.shape\s*\[", seg) or re.search(r"\.size\s*\(", seg):
            # 排除单纯断言/校验（raise 在同行或紧邻）——只对"分支选实现"告警较难静态判定，保守告警
            findings.append(("WARN", "size-branch",
                             "reference 出现按 numel/shape/size 的 if 分支——若据规模切换快/慢实现是弱 baseline"
                             "(bench 规模命中慢分支虚高)；金标准须始终单一最干净向量化。若仅为参数校验(raise)可忽略"))
            break

    # 5. 数值脆弱 cumprod + 除法（变系数递推的脆弱 O(N) 形式）
    if re.search(r"\bcumprod\b", code) and re.search(r"/", code):
        findings.append(("WARN", "cumprod-div",
                         "reference 用 cumprod 且含除法——变系数递推的 cumprod/cumsum(x/cumprod) 形式数值脆弱"
                         "(系数连乘下溢→除法 NaN)且反向图畸形；应改 log 空间 O(T^2) 下三角(exp(L_t-L_j))"))

    return findings


def scan_make_inputs(src):
    """扫 make_inputs 有无挑异常输入分布迁就脆弱 reference。返回 [(severity, code, msg)]。"""
    findings = []
    code = _strip_comments(src)
    # 截取 make_inputs 函数体（到下一个 def 或文件尾）
    m = re.search(r"def\s+make_inputs\b.*?(?=\ndef\s|\Z)", code, re.DOTALL)
    body = m.group(0) if m else code

    # 6. sigmoid/exp 门控被偏置成恒饱和：uniform_(大数..) 或 + 大常数 喂给随后 sigmoid
    #    典型作弊：b.uniform_(3.5, 4.5) 让 sigmoid(wx+b)≈1；或 randn()*s + 4
    for m2 in re.finditer(r"uniform_\(\s*([0-9.+\-]+)\s*,\s*([0-9.+\-]+)", body):
        lo, hi = m2.group(1), m2.group(2)
        try:
            if min(abs(float(lo)), abs(float(hi))) >= 2.5:  # 区间整体远离 0（偏置饱和信号）
                findings.append(("WARN", "input-bias",
                                 f"make_inputs 用 uniform_({lo},{hi}) 区间远离 0——若该量随后过 sigmoid/exp 会恒接近饱和值,"
                                 f"疑似挑输入分布迁就脆弱 reference(评测作弊);应用自然分布(如 randn/常规范围)"))
        except ValueError:
            pass

    # 7. randn/rand 后加大常数偏置
    if re.search(r"(randn|rand)\s*\([^\n]*\)\s*\*?\s*[0-9.]*\s*\+\s*([3-9]|[1-9][0-9])", body):
        findings.append(("WARN", "input-shift",
                         "make_inputs 给随机张量加较大常数偏置(≥3)——若随后过 sigmoid/exp 疑似令激活饱和,"
                         "疑似挑分布迁就脆弱 reference;确认是自然分布"))

    return findings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="case 名，如 rbf")
    ap.add_argument("--workdir", default=None, help="仓库根路径（默认按脚本位置推断；扫其他 clone 的 case 时指定）")
    ap.add_argument("--strict", action="store_true", help="有命中则 exit 1（供闭环拦截；默认只 WARN exit 0）")
    args = ap.parse_args()

    root = args.workdir or _repo_root()
    ref_path = os.path.join(root, "cases", args.case, "reference.py")
    if not os.path.exists(ref_path):
        print(f"[check_reference] 找不到 {ref_path}", file=sys.stderr)
        print("REF_CHECK=WARN n=1 (reference.py 缺失)")
        return 1 if args.strict else 0

    src = _read(ref_path)
    findings = scan_reference(src) + scan_make_inputs(src)

    print(f"=== reference 静态扫描: case={args.case} ===")
    if not findings:
        print("  无可疑写法（for/T^2-Toeplitz/规模分支/cumprod-脆弱/高层算子/挑输入分布 均未命中）")
        print("REF_CHECK=CLEAN")
        return 0

    reds = [f for f in findings if f[0] == "RED"]
    for sev, code, msg in findings:
        print(f"  [{sev}] ({code}) {msg}")
    print(f"REF_CHECK=WARN n={len(findings)}" + (f" red={len(reds)}" if reds else ""))
    # 红线命中或 --strict：非零退出（红线是硬违规；WARN 类默认放行因有合法例外）
    if reds or args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
