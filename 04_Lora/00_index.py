"""
LoRA 全家桶 — 8 个文件，从基础到进阶
======================================

01_linear_lora.py  — Linear LoRA (基座×2, ÷2)
02_conv_lora.py    — Conv LoRA (1×1卷积旁路, 边缘检测)
03_gpt_lora.py     — GPT Attention QKV + LoRA
04_auto_lora.py    — 自动挂载 ResNet (named_modules)
05_mha_lora.py     — nn.MultiheadAttention + LoRA (手写MHA)
06_qlora.py        — QLoRA (NF4量化基座 + LoRA旁路)
07_dora.py         — DoRA (权重分解: 幅度×方向, LoRA只管方向)
08_adalora.py      — AdaLoRA (自适应rank: SVD分解+剪不重要秩)
"""

if __name__ == "__main__":
    files = [
        ("01_linear_lora.py",  "Linear LoRA",    "基座冻结，A·B旁路学×2和÷2"),
        ("02_conv_lora.py",    "Conv LoRA",      "1×1卷积旁路，恒等→边缘检测"),
        ("03_gpt_lora.py",     "GPT + LoRA",     "注意力QKV挂LoRA，微调新风格"),
        ("04_auto_lora.py",    "自动挂载",        "named_modules()递归挂ResNet"),
        ("05_mha_lora.py",     "MHA + LoRA",     "手写MultiheadAttention，QKV各独立"),
        ("06_qlora.py",        "QLoRA",          "量化基座(4-bit)+LoRA，省显存"),
        ("07_dora.py",         "DoRA",           "权重分解为幅度×方向，分离微调"),
        ("08_adalora.py",      "AdaLoRA",        "SVD形式，自适应分配rank给各层"),
    ]
    for name, tag, desc in files:
        print(f"  {tag:<18s} {desc}")
    print(f"\n  学习顺序: 1→2→4→5→3→6→7→8")
