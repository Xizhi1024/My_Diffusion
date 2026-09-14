from __future__ import annotations

import math
import re
from pathlib import Path

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, Color
from reportlab.lib.pagesizes import A4, landscape


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output" / "pdf"
OUT_PATH = OUT_DIR / "SLMF_BBDM_model_report.pdf"

PAGE_W, PAGE_H = landscape(A4)

NAVY = HexColor("#102A43")
BLUE = HexColor("#2563EB")
CYAN = HexColor("#0891B2")
TEAL = HexColor("#0F766E")
ORANGE = HexColor("#D97706")
RED = HexColor("#C2413B")
PURPLE = HexColor("#6D5BD0")
INK = HexColor("#172B4D")
MUTED = HexColor("#5F6C7B")
LIGHT = HexColor("#F4F7FB")
LINE = HexColor("#D7E0EA")
WHITE = HexColor("#FFFFFF")
GREEN_BG = HexColor("#E8F7F2")
BLUE_BG = HexColor("#EAF2FF")
ORANGE_BG = HexColor("#FFF3DF")
PURPLE_BG = HexColor("#F1EEFF")
RED_BG = HexColor("#FCEBE9")


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("CN", r"C:\Windows\Fonts\msyh.ttc", subfontIndex=0))
    pdfmetrics.registerFont(TTFont("CN-Bold", r"C:\Windows\Fonts\msyhbd.ttc", subfontIndex=0))


def fit_lines(text: str, font: str, size: float, max_width: float) -> list[str]:
    """Wrap mixed Chinese/Latin text using measured glyph widths."""
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if paragraph == "":
            lines.append("")
            continue
        current = ""
        # Keep Latin terms such as pred_x0, Gabor and CT→PET intact while
        # retaining character-level wrapping for Chinese prose.
        tokens = re.findall(r"[A-Za-z0-9_./+<>=×→≈μστβ₀₁₂ₓ-]+|[ \t]+|.", paragraph)
        for token in tokens:
            candidate = current + token
            if current and pdfmetrics.stringWidth(candidate, font, size) > max_width:
                lines.append(current.rstrip())
                current = token.lstrip()
            else:
                current = candidate
        if current:
            lines.append(current.rstrip())
    return lines


def draw_text(c: canvas.Canvas, text: str, x: float, y: float, width: float,
              size: float = 12, color=INK, font: str = "CN",
              leading: float | None = None, max_lines: int | None = None) -> float:
    leading = leading or size * 1.55
    lines = fit_lines(text, font, size, width)
    if max_lines is not None:
        lines = lines[:max_lines]
    c.setFillColor(color)
    c.setFont(font, size)
    cursor = y
    for line in lines:
        c.drawString(x, cursor, line)
        cursor -= leading
    return cursor


def draw_bullets(c: canvas.Canvas, items: list[str], x: float, y: float, width: float,
                 size: float = 11, color=INK, bullet_color=BLUE,
                 leading: float | None = None, gap: float = 7) -> float:
    leading = leading or size * 1.55
    cursor = y
    for item in items:
        lines = fit_lines(item, "CN", size, width - 18)
        c.setFillColor(bullet_color)
        c.circle(x + 4, cursor + 3, 2.4, fill=1, stroke=0)
        c.setFillColor(color)
        c.setFont("CN", size)
        for i, line in enumerate(lines):
            c.drawString(x + 16, cursor, line)
            cursor -= leading
        cursor -= gap
    return cursor


def round_rect(c: canvas.Canvas, x: float, y: float, w: float, h: float,
               fill, stroke=LINE, radius: float = 12, stroke_width: float = 1) -> None:
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(stroke_width)
    c.roundRect(x, y, w, h, radius, fill=1, stroke=1)


def card(c: canvas.Canvas, x: float, y: float, w: float, h: float,
         title: str, body: str, accent=BLUE, fill=WHITE,
         title_size: float = 14, body_size: float = 10.5) -> None:
    round_rect(c, x, y, w, h, fill)
    c.setFillColor(accent)
    c.roundRect(x, y, 6, h, 4, fill=1, stroke=0)
    c.setFillColor(accent)
    c.setFont("CN-Bold", title_size)
    c.drawString(x + 20, y + h - 28, title)
    draw_text(c, body, x + 20, y + h - 52, w - 38, size=body_size,
              color=MUTED, leading=body_size * 1.55)


def page_header(c: canvas.Canvas, section: str, page_num: int) -> None:
    c.setFillColor(WHITE)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.rect(0, PAGE_H - 8, PAGE_W, 8, fill=1, stroke=0)
    c.setFont("CN-Bold", 10)
    c.setFillColor(MUTED)
    c.drawString(34, PAGE_H - 30, "SLMF-BBDM 模型汇报")
    c.setFont("CN", 9)
    c.drawRightString(PAGE_W - 34, PAGE_H - 30, section)
    c.setStrokeColor(LINE)
    c.line(34, 27, PAGE_W - 34, 27)
    c.setFillColor(MUTED)
    c.setFont("CN", 8.5)
    c.drawString(34, 13, "基于当前工作目录源码与配置整理 | 2026-06-21")
    c.drawRightString(PAGE_W - 34, 13, f"{page_num:02d}")


def section_title(c: canvas.Canvas, title: str, subtitle: str = "") -> None:
    c.setFont("CN-Bold", 24)
    c.setFillColor(NAVY)
    c.drawString(42, PAGE_H - 72, title)
    if subtitle:
        c.setFont("CN", 10.5)
        c.setFillColor(MUTED)
        c.drawString(43, PAGE_H - 92, subtitle)


def arrow(c: canvas.Canvas, x1: float, y1: float, x2: float, y2: float,
          color=BLUE, width: float = 2, dashed: bool = False) -> None:
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(width)
    c.setDash(5, 4) if dashed else c.setDash()
    c.line(x1, y1, x2, y2)
    angle = math.atan2(y2 - y1, x2 - x1)
    length = 8
    spread = 0.48
    p1 = (x2 - length * math.cos(angle - spread), y2 - length * math.sin(angle - spread))
    p2 = (x2 - length * math.cos(angle + spread), y2 - length * math.sin(angle + spread))
    path = c.beginPath()
    path.moveTo(x2, y2)
    path.lineTo(*p1)
    path.lineTo(*p2)
    path.close()
    c.drawPath(path, fill=1, stroke=0)
    c.setDash()


def node(c: canvas.Canvas, x: float, y: float, w: float, h: float,
         title: str, sub: str = "", fill=WHITE, stroke=BLUE,
         dashed: bool = False, title_size: float = 11) -> None:
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(1.5)
    c.setDash(5, 3) if dashed else c.setDash()
    c.roundRect(x, y, w, h, 9, fill=1, stroke=1)
    c.setDash()
    c.setFillColor(stroke)
    c.setFont("CN-Bold", title_size)
    c.drawCentredString(x + w / 2, y + h - 21, title)
    if sub:
        lines = fit_lines(sub, "CN", 8.3, w - 14)[:2]
        c.setFillColor(MUTED)
        c.setFont("CN", 8.3)
        yy = y + h - 38
        for line in lines:
            c.drawCentredString(x + w / 2, yy, line)
            yy -= 12


def cover(c: canvas.Canvas) -> None:
    c.setFillColor(NAVY)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(BLUE)
    c.circle(PAGE_W - 70, PAGE_H - 65, 145, fill=1, stroke=0)
    c.setFillColor(Color(0.05, 0.70, 0.82, alpha=0.45))
    c.circle(PAGE_W - 160, 40, 125, fill=1, stroke=0)
    c.setFillColor(Color(1, 1, 1, alpha=0.08))
    for i in range(7):
        c.circle(PAGE_W - 165 + i * 24, PAGE_H - 215 - (i % 2) * 16, 5, fill=1, stroke=0)

    c.setFillColor(CYAN)
    c.roundRect(56, PAGE_H - 112, 158, 28, 14, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("CN-Bold", 11)
    c.drawCentredString(135, PAGE_H - 103, "CT → PET 医学影像合成")

    c.setFont("CN-Bold", 34)
    c.drawString(56, PAGE_H - 180, "SLMF-BBDM")
    c.setFont("CN-Bold", 24)
    c.drawString(56, PAGE_H - 222, "小病灶代谢保真布朗桥扩散模型")
    c.setFillColor(HexColor("#C8D6E5"))
    c.setFont("CN", 14)
    c.drawString(58, PAGE_H - 260, "模型架构 · 工作内容 · 三个创新点 · 当前完成度")

    c.setStrokeColor(Color(1, 1, 1, alpha=0.3))
    c.line(56, 172, 560, 172)
    c.setFillColor(WHITE)
    c.setFont("CN-Bold", 12)
    c.drawString(56, 143, "核心目标")
    c.setFont("CN", 12)
    c.setFillColor(HexColor("#D8E4EF"))
    c.drawString(56, 119, "输入 CT 切片，生成兼顾小病灶、SUV 定量与可信度的 PET 代谢图像")

    c.setFont("CN", 9.5)
    c.setFillColor(HexColor("#AFC2D4"))
    c.drawString(56, 44, "工作目录：E:/研究生课题（子宫内膜癌）/工作3/My_diffusion")
    c.drawRightString(PAGE_W - 44, 44, "2026-06-21")
    c.showPage()


def page_overview(c: canvas.Canvas) -> None:
    page_header(c, "01 / 研究任务与模型定位", 2)
    section_title(c, "模型解决什么问题？", "当前项目是 CT→PET 合成模型，不是肿瘤分割网络")

    card(c, 42, 322, 235, 145, "输入", "单张 192×192 CT 切片；训练时配对真实 PET 与病灶标签；可选器官掩膜、距离场、衰减图和扫描元数据。", BLUE, BLUE_BG)
    card(c, 303, 322, 235, 145, "核心映射", "在 CT 与 PET 之间构造布朗桥扩散过程，由条件 U-Net 直接预测无噪声 PET（pred_x0）。", TEAL, GREEN_BG)
    card(c, 564, 322, 235, 145, "输出", "合成 PET、像素级 logvar；多次采样还可得到认知不确定性、总方差与置信度图。", PURPLE, PURPLE_BG)

    arrow(c, 279, 395, 298, 395, BLUE, 2)
    arrow(c, 540, 395, 559, 395, TEAL, 2)

    round_rect(c, 42, 82, 757, 205, LIGHT)
    c.setFont("CN-Bold", 15)
    c.setFillColor(NAVY)
    c.drawString(62, 257, "模型承担的四项工作")
    items = [
        ("01", "解剖到代谢的跨模态生成", "利用 CT 解剖结构恢复 PET 代谢分布。", BLUE),
        ("02", "小病灶细节保护", "降低扩散噪声对高频边缘和弱小热点的破坏。", CYAN),
        ("03", "临床定量约束", "显式优化 SUVmax、SUVmean、TBR 与假热点。", ORANGE),
        ("04", "生成可信度估计", "区分偶然不确定性与认知不确定性。", PURPLE),
    ]
    x_positions = [62, 247, 432, 617]
    for (num, title, body, col), x in zip(items, x_positions):
        c.setFillColor(col)
        c.circle(x + 16, 216, 16, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont("CN-Bold", 10)
        c.drawCentredString(x + 16, 212, num)
        c.setFillColor(INK)
        c.setFont("CN-Bold", 11.5)
        c.drawString(x, 178, title)
        draw_text(c, body, x, 152, 157, size=9.5, color=MUTED, leading=15)
    c.showPage()


def page_architecture(c: canvas.Canvas) -> None:
    page_header(c, "02 / 总体模型架构", 3)
    section_title(c, "SLMF-BBDM 总体架构图", "实线为当前完整配置启用路径；虚线为已实现但默认关闭的扩展路径")

    # Main generative path
    node(c, 42, 372, 105, 70, "真实 PET", "训练目标 x₀", ORANGE_BG, ORANGE)
    node(c, 42, 270, 105, 70, "源域 CT", "桥接端点 x_source", BLUE_BG, BLUE)
    node(c, 190, 318, 170, 95, "尺度自适应布朗桥", "xₜ=mₜCT+(1-mₜ)PET+σₜε\n高/中/低频分带加噪", GREEN_BG, TEAL)
    node(c, 418, 318, 160, 95, "条件 BBDM U-Net", "4 层编码-解码\n直接预测 pred_x0", BLUE_BG, BLUE)
    node(c, 635, 357, 160, 76, "合成 PET", "代谢图像", GREEN_BG, TEAL)
    node(c, 635, 260, 160, 76, "不确定性", "logvar + MC variance", PURPLE_BG, PURPLE)

    arrow(c, 147, 407, 185, 380, ORANGE)
    arrow(c, 147, 305, 185, 342, BLUE)
    arrow(c, 360, 365, 413, 365, TEAL, 2.5)
    arrow(c, 578, 377, 630, 394, TEAL, 2.5)
    arrow(c, 578, 350, 630, 298, PURPLE, 2.5)

    # Conditioning lane
    c.setFillColor(NAVY)
    c.setFont("CN-Bold", 12)
    c.drawString(42, 224, "多源条件与先验通道")
    node(c, 42, 123, 115, 72, "多尺度 CT", "4 级结构特征", BLUE_BG, BLUE)
    node(c, 174, 123, 115, 72, "Gabor 先验", "纹理/边缘/能量", GREEN_BG, TEAL)
    node(c, 306, 123, 115, 72, "器官先验", "mask/距离/μ-map", ORANGE_BG, ORANGE)
    node(c, 438, 123, 115, 72, "热点先验", "CT→病灶候选", RED_BG, RED)
    node(c, 570, 123, 105, 72, "语义 token", "默认关闭", PURPLE_BG, PURPLE, dashed=True)
    node(c, 702, 123, 93, 72, "元数据", "FiLM/关闭", LIGHT, MUTED, dashed=True)

    round_rect(c, 215, 52, 410, 42, WHITE, stroke=BLUE, radius=10, stroke_width=1.5)
    c.setFillColor(BLUE)
    c.setFont("CN-Bold", 11)
    c.drawCentredString(420, 68, "时变 β 调制的 Zero-Conv → 注入 U-Net 多尺度 skip 与瓶颈")

    for x in [99, 231, 363, 495]:
        arrow(c, x, 120, 330 + (x - 99) * 0.25, 95, BLUE, 1.3)
    arrow(c, 622, 120, 530, 95, PURPLE, 1.2, dashed=True)
    arrow(c, 748, 120, 610, 95, MUTED, 1.2, dashed=True)
    arrow(c, 420, 95, 470, 313, BLUE, 1.8)
    arrow(c, 231, 195, 264, 313, TEAL, 1.4)

    c.showPage()


def page_modules(c: canvas.Canvas) -> None:
    page_header(c, "03 / 关键模块", 4)
    section_title(c, "关键模块如何协同？", "每个模块均可通过配置开关独立消融")

    cards = [
        (42, 325, 235, 145, "Gabor 高频先验", "32 个可学习 Gabor 滤波器提取纹理与边缘；能量图同时用于条件注入、频域约束和高频噪声调节。", TEAL, GREEN_BG),
        (303, 325, 235, 145, "器官与衰减先验", "融合 6 类器官掩膜、器官距离场及 511keV 衰减图，抑制骨、脂肪、肌肉等区域的异常高摄取。", ORANGE, ORANGE_BG),
        (564, 325, 235, 145, "轻量热点先验", "小型 CT→hotspot U-Net，联合 CT 与 Gabor 能量预测病灶候选图，参数预算小于 0.5M。", RED, RED_BG),
        (42, 135, 235, 145, "时变 Zero-Conv 注入", "不同分辨率条件通过零初始化 1×1 卷积注入 skip；随 τ 调节全局结构与局部细节的权重。", BLUE, BLUE_BG),
        (303, 135, 235, 145, "Self-conditioning + CFG", "训练时以 0.5 概率回馈前一次 x₀ 预测；条件随机丢弃支持弱分类器自由引导采样。", CYAN, LIGHT),
        (564, 135, 235, 145, "双层不确定性", "异方差头估计偶然不确定性；MC 多次采样估计认知不确定性，并组合生成置信度图。", PURPLE, PURPLE_BG),
    ]
    for args in cards:
        card(c, *args)
    c.showPage()


def innovation_card(c: canvas.Canvas, x: float, y: float, w: float, h: float,
                    number: str, title: str, tagline: str, bullets: list[str], color, bg) -> None:
    round_rect(c, x, y, w, h, bg, stroke=color, radius=16, stroke_width=1.4)
    c.setFillColor(color)
    c.circle(x + 34, y + h - 36, 22, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("CN-Bold", 15)
    c.drawCentredString(x + 34, y + h - 42, number)
    c.setFillColor(NAVY)
    c.setFont("CN-Bold", 15)
    c.drawString(x + 66, y + h - 31, title)
    c.setFillColor(color)
    c.setFont("CN-Bold", 9.8)
    c.drawString(x + 66, y + h - 49, tagline)
    draw_bullets(c, bullets, x + 20, y + h - 83, w - 38, size=9.4,
                 bullet_color=color, leading=14.5, gap=5)


def page_innovations(c: canvas.Canvas) -> None:
    page_header(c, "04 / 三个创新点", 5)
    section_title(c, "三个可凝练的方法创新点", "下列表述均对应当前代码实现；“首次提出”仍需进一步文献检索证明")

    innovation_card(c, 42, 105, 235, 350, "01", "尺度自适应布朗桥", "任务对齐 + 小病灶频率保护", [
        "以 CT 和 PET 作为布朗桥两端，而非从纯高斯噪声无条件生成。",
        "通过拉普拉斯金字塔分解高/中/低频带，分别控制噪声强度。",
        "高频噪声倍率降至 0.45，且 Gabor 能量越强，局部高频破坏越小。",
        "目标是降低小病灶、弱边缘和细粒度热点在扩散中的信息损失。",
    ], TEAL, GREEN_BG)

    innovation_card(c, 303, 105, 235, 350, "02", "多源先验分层时变融合", "在哪里注入 + 何时起作用", [
        "联合多尺度 CT、Gabor 纹理、器官结构、衰减图和热点候选信息。",
        "浅层强调边缘和热点，深层强调器官与全局解剖，语义 token 可进入瓶颈。",
        "Zero-Conv 从近似基础模型的稳定状态开始学习，避免强先验初期干扰。",
        "时变 β 调度使早期重结构、后期重局部代谢细节。",
    ], BLUE, BLUE_BG)

    innovation_card(c, 564, 105, 235, 350, "03", "临床代谢保真与可信生成", "不只像 PET，还要定量合理、风险可见", [
        "联合优化 SUVmax、SUVmean、TBR、小病灶 Top-K、频域和局部对应。",
        "引入器官代谢一致性和假热点抑制，减少正常组织异常高摄取。",
        "损失按扩散阶段动态启用，使结构、病灶和 SUV 目标各司其时。",
        "异方差与 MC 采样联合给出置信度，为临床使用提供风险提示。",
    ], PURPLE, PURPLE_BG)
    c.showPage()


def page_training(c: canvas.Canvas) -> None:
    page_header(c, "05 / 训练与评价", 6)
    section_title(c, "训练目标与评价闭环", "从像素相似度扩展到病灶、频率、SUV、器官合理性与可信度")

    # Timeline
    c.setFont("CN-Bold", 14)
    c.setFillColor(NAVY)
    c.drawString(42, 455, "扩散阶段调度")
    x0, x1, y = 70, 775, 405
    c.setStrokeColor(LINE)
    c.setLineWidth(7)
    c.line(x0, y, x1, y)
    for x, col, label in [(x0, BLUE, "τ≈1\n高噪声/早期"), ((x0+x1)/2, ORANGE, "τ≈0.5\n中期"), (x1, TEAL, "τ≈0\n低噪声/后期")]:
        c.setFillColor(col)
        c.circle(x, y, 10, fill=1, stroke=0)
        lines = label.split("\n")
        c.setFont("CN-Bold", 9)
        c.setFillColor(col)
        c.drawCentredString(x, y + 22, lines[0])
        c.setFont("CN", 8.5)
        c.drawCentredString(x, y - 27, lines[1])
    c.setFillColor(BLUE)
    c.roundRect(90, 342, 190, 34, 9, fill=1, stroke=0)
    c.setFillColor(WHITE); c.setFont("CN-Bold", 9.5); c.drawCentredString(185, 354, "器官一致性 / 全局结构")
    c.setFillColor(ORANGE)
    c.roundRect(300, 342, 185, 34, 9, fill=1, stroke=0)
    c.setFillColor(WHITE); c.drawCentredString(392, 354, "热点先验监督")
    c.setFillColor(TEAL)
    c.roundRect(505, 342, 250, 34, 9, fill=1, stroke=0)
    c.setFillColor(WHITE); c.drawCentredString(630, 354, "Top-K / 频域 / SUV / 假热点")

    card(c, 42, 102, 235, 190, "基础重建目标", "MSE + L1 + 梯度一致性；使用 Min-SNR 权重缓解不同时间步训练不平衡。", BLUE, BLUE_BG, body_size=10.5)
    card(c, 303, 102, 235, 190, "训练协议", "图像 192×192；batch=4；梯度累积 2；AdamW；1000 epochs；EMA=0.999；AMP；余弦学习率。", TEAL, GREEN_BG, body_size=10.5)
    card(c, 564, 102, 235, 190, "评价指标", "MAE、MSE、PSNR、SSIM；SUVmax、SUVmean、TBR；假热点数量/密度；患者级汇总。", PURPLE, PURPLE_BG, body_size=10.5)
    c.showPage()


def page_status(c: canvas.Canvas) -> None:
    page_header(c, "06 / 当前完成度与汇报结论", 7)
    section_title(c, "当前工程处于什么阶段？", "模型设计与代码框架基本完成；真实训练、对比与统计证据尚未在本目录闭环")

    card(c, 42, 304, 360, 168, "已完成", "• CT/PET/标签清单与 DICOM 配准映射\n• HU、SUV、器官先验和语义缓存流程\n• 模型、训练器、EMA、采样与 checkpoint\n• 基线及 11 类消融配置\n• 79 个 smoke test；静态编译检查通过", TEAL, GREEN_BG, body_size=10.5)
    card(c, 439, 304, 360, 168, "尚缺少的证据", "• 当前目录无预处理 cache/tensors\n• 无正式训练 checkpoint 与训练曲线\n• 无验证/测试定量结果和可视化\n• 参数量、显存、推理速度尚未正式记录\n• 当前终端缺少 Pixi/PyTorch，单测未实际执行", ORANGE, ORANGE_BG, body_size=10.5)

    round_rect(c, 42, 105, 757, 155, NAVY, stroke=NAVY, radius=16)
    c.setFillColor(CYAN)
    c.setFont("CN-Bold", 12)
    c.drawString(64, 228, "汇报结论")
    conclusion = (
        "SLMF-BBDM 以尺度自适应布朗桥为生成主线，通过多源医学先验的分层时变注入，"
        "并联合临床 SUV、病灶细节、器官合理性及不确定性目标，实现面向小病灶代谢保真的 CT→PET 合成。"
        "当前最重要的下一步不是继续堆叠模块，而是完成数据预处理、正式训练、三项创新的消融验证及患者级统计分析。"
    )
    draw_text(c, conclusion, 64, 200, 710, size=12.2, color=WHITE, font="CN", leading=22)
    c.showPage()


def build() -> Path:
    register_fonts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT_PATH), pagesize=(PAGE_W, PAGE_H))
    c.setTitle("SLMF-BBDM 模型汇报")
    c.setAuthor("Codex - based on current project source code")
    c.setSubject("CT-to-PET Brownian Bridge Diffusion Model")
    cover(c)
    page_overview(c)
    page_architecture(c)
    page_modules(c)
    page_innovations(c)
    page_training(c)
    page_status(c)
    c.save()
    return OUT_PATH


if __name__ == "__main__":
    print(build())
