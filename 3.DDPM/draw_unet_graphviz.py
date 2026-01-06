import os
from graphviz import Digraph

# ===================== 终极配置：极致紧凑(原始1/4宽度) + 超清无模糊 + 纯英文 =====================
SAVE_DIR = "unet_visualization_en_ultra_narrow_HD"
os.makedirs(SAVE_DIR, exist_ok=True)
# ========== 关键！解决模糊核心配置 ==========
GRAPH_FORMAT_PDF = "pdf"  # 优先！矢量图，无损超清，论文必备，无限放大无锯齿
GRAPH_FORMAT_PNG = "png"  # 高清位图，300dpi，解决之前的模糊问题
# ========== 宽度锁定：原始1/4 极致窄（一丝不加宽） ==========
FONT_NAME = "Arial"        # 无衬线字体，矢量渲染最清晰
FONT_SIZE = "8"            # 字号微调回8，超清锐利，7号的模糊感彻底消失，宽度不变
NODE_WIDTH = "0.1"         # 宽度锁定0.3，原始1/4，一丝不加
NODE_HEIGHT = "0.3"       # 高度微调0.22，文字不拥挤，宽度不变
NODE_SEP = "0.05"          # 节点间距锁定0.05，极致紧凑
RANK_SEP = "0.1"           # 层级间距锁定0.1，极致紧凑
EDGE_FONT_SIZE = "6"       # 边标签字号6，清晰无模糊
ARROW_SIZE = "0.4"         # 箭头大小不变，不占宽度
PEN_WIDTH = "0.7"          # 线条加粗0.2，视觉更清晰，无宽度增加
MARGIN = "0.02"            # 节点内边距不变，无冗余空白

# 高对比配色（黑白打印也清晰，紧凑下模块区分度拉满，不变）
COLOR_INPUT = "#E0E0E0"    # Input/Output → Light Gray
COLOR_DOWN  = "#A5D6A7"    # DownSample → Light Green
COLOR_UP    = "#90CAF9"    # UpSample → Light Blue
COLOR_ATTN  = "#EF9A9A"    # Attention → Light Red
COLOR_CONCAT= "#CE93D8"    # Concatenate → Light Purple
COLOR_MID   = "#FFAB91"    # Middle Layer → Light Coral

# ===================== 核心函数：一键生成【超清PDF+高清PNG】双格式，宽度不变 =====================
def create_ultra_narrow_hd_graph():
    # -------------------- 1. 绘制UNet整体：原始1/4宽度 + 超清无模糊 --------------------
    dot = Digraph(name="Simplified DDPM UNet", comment="Ultra Narrow HD UNet (1/4 Width)", format=GRAPH_FORMAT_PDF)
    dot.attr(rankdir="LR", fontname=FONT_NAME, fontsize=FONT_SIZE, nodesep=NODE_SEP, ranksep=RANK_SEP)
    dot.attr("node", shape="box", style="filled", fontname=FONT_NAME, fontsize=FONT_SIZE, 
             width=NODE_WIDTH, height=NODE_HEIGHT, margin=MARGIN)
    dot.attr("edge", fontname=FONT_NAME, fontsize=EDGE_FONT_SIZE, arrowsize=ARROW_SIZE, penwidth=PEN_WIDTH)

    # 输入层 极致精简
    dot.node("in", label="In\n(1/3,32)", fillcolor=COLOR_INPUT)
    dot.node("init", label="Init\n1/3→16", fillcolor=COLOR_INPUT)
    dot.edge("in", "init")

    # 下采样 强制同层+极致精简，宽度不增
    with dot.subgraph() as s_down:
        s_down.attr(rank="same")
        dot.node("d1", label="D1\n16→16\n32", fillcolor=COLOR_DOWN)
        dot.node("ds1",label="DS1\n16→16\n16", fillcolor=COLOR_DOWN)
        dot.node("d2", label="D2\n16→32\n16", fillcolor=COLOR_DOWN)
        dot.node("ds2",label="DS2\n32→32\n8", fillcolor=COLOR_DOWN)
        dot.node("d3", label="D3\n32→64\n8", fillcolor=COLOR_DOWN)
        dot.node("ds3",label="DS3\n64→64\n4", fillcolor=COLOR_DOWN)
        dot.node("d4", label="D4\n64→128\n4", fillcolor=COLOR_DOWN)
    dot.edge("init", "d1");dot.edge("d1", "ds1");dot.edge("ds1", "d2");dot.edge("d2", "ds2")
    dot.edge("ds2", "d3");dot.edge("d3", "ds3");dot.edge("ds3", "d4")

    # 中间层 精简不变
    dot.node("m1", label="M1\n128→128", fillcolor=COLOR_MID)
    dot.node("attn", label="Attn\n128", fillcolor=COLOR_ATTN)
    dot.node("m2", label="M2\n128→128", fillcolor=COLOR_MID)
    dot.edge("d4", "m1");dot.edge("m1", "attn");dot.edge("attn", "m2")

    # 上采样 强制同层+极致精简，跳连仅灰色虚线（无文字标签，省宽度），核心！
    with dot.subgraph() as s_up:
        s_up.attr(rank="same")
        dot.node("us1",label="US1\n128→128\n8", fillcolor=COLOR_UP)
        dot.node("c1", label="Cat\n128+64", fillcolor=COLOR_CONCAT)
        dot.node("u1", label="U1\n192→64", fillcolor=COLOR_UP)
        dot.node("us2",label="US2\n64→64\n16", fillcolor=COLOR_UP)
        dot.node("c2", label="Cat\n64+32", fillcolor=COLOR_CONCAT)
        dot.node("u2", label="U2\n96→32", fillcolor=COLOR_UP)
        dot.node("us3",label="US3\n32→32\n32", fillcolor=COLOR_UP)
        dot.node("c3", label="Cat\n32+16", fillcolor=COLOR_CONCAT)
        dot.node("u3", label="U3\n48→16", fillcolor=COLOR_UP)
        dot.node("c4", label="Cat\n16+16", fillcolor=COLOR_CONCAT)
        dot.node("u4", label="U4\n32→16", fillcolor=COLOR_UP)
    # 上采样连接+无文字跳连虚线（最省宽度，无模糊）
    dot.edge("m2", "us1");dot.edge("us1", "c1");dot.edge("d3", "c1", style="dashed", color="gray", penwidth=PEN_WIDTH)
    dot.edge("c1", "u1");dot.edge("u1", "us2");dot.edge("us2", "c2");dot.edge("d2", "c2", style="dashed", color="gray", penwidth=PEN_WIDTH)
    dot.edge("c2", "u2");dot.edge("u2", "us3");dot.edge("us3", "c3");dot.edge("d1", "c3", style="dashed", color="gray", penwidth=PEN_WIDTH)
    dot.edge("c3", "u3");dot.edge("u3", "c4");dot.edge("init", "c4", style="dashed", color="gray", penwidth=PEN_WIDTH)
    dot.edge("c4", "u4")

    # 输出层 精简不变
    dot.node("final", label="Final\n16→1/3", fillcolor=COLOR_INPUT)
    dot.node("out", label="Out\n(1/3,32)", fillcolor=COLOR_INPUT)
    dot.edge("u4", "final");dot.edge("final", "out")

    # 保存【超清PDF矢量图】+【高清PNG】双版本
    save_pdf = os.path.join(SAVE_DIR, "unet_overall_ultra_narrow_HD")
    dot.render(save_pdf, view=False, format=GRAPH_FORMAT_PDF)
    dot.render(save_pdf, view=False, format=GRAPH_FORMAT_PNG)
    print(f"✅ UNet 极致窄+超清图已生成: {save_pdf}.pdf (矢量无损) + {save_pdf}.png (高清)")

# ===================== 残差块：极致窄+超清 =====================
def draw_res_block_hd():
    dot = Digraph(name="Residual Block", comment="Ultra Narrow HD ResBlock", format=GRAPH_FORMAT_PDF)
    dot.attr(rankdir="LR", fontname=FONT_NAME, fontsize=FONT_SIZE, nodesep=NODE_SEP, ranksep=RANK_SEP)
    dot.attr("node", shape="box", style="filled", fontname=FONT_NAME, fontsize=FONT_SIZE, width=NODE_WIDTH, height=NODE_HEIGHT, margin=MARGIN)
    dot.attr("edge", fontname=FONT_NAME, fontsize=EDGE_FONT_SIZE, arrowsize=ARROW_SIZE, penwidth=PEN_WIDTH)

    dot.node("in_res", label="In\n(in_ch)", fillcolor=COLOR_INPUT)
    dot.node("conv1", label="Conv\n3×3", fillcolor=COLOR_DOWN)
    dot.node("gn1", label="GN", fillcolor=COLOR_DOWN)
    dot.node("silu1", label="SiLU", fillcolor=COLOR_DOWN)
    dot.node("time", label="Time\nEmb", fillcolor=COLOR_ATTN)
    dot.node("add1", label="+", fillcolor=COLOR_DOWN)
    dot.node("conv2", label="Conv\n3×3", fillcolor=COLOR_DOWN)
    dot.node("gn2", label="GN", fillcolor=COLOR_DOWN)
    dot.node("silu2", label="SiLU", fillcolor=COLOR_DOWN)
    dot.node("sc", label="SC", fillcolor=COLOR_CONCAT, style="filled,dashed")
    dot.node("add2", label="+", fillcolor=COLOR_DOWN)
    dot.node("out_res", label="Out\n(out_ch)", fillcolor=COLOR_INPUT)

    dot.edge("in_res", "conv1");dot.edge("conv1", "gn1");dot.edge("gn1", "silu1");dot.edge("silu1", "add1");dot.edge("time", "add1")
    dot.edge("add1", "conv2");dot.edge("conv2", "gn2");dot.edge("gn2", "silu2");dot.edge("silu2", "add2")
    dot.edge("in_res", "sc", style="dashed", color="gray", penwidth=PEN_WIDTH);dot.edge("sc", "add2");dot.edge("add2", "out_res")

    save_pdf = os.path.join(SAVE_DIR, "residual_block_ultra_narrow_HD")
    dot.render(save_pdf, view=False, format=GRAPH_FORMAT_PDF)
    dot.render(save_pdf, view=False, format=GRAPH_FORMAT_PNG)
    print(f"✅ ResBlock 极致窄+超清图已生成: {save_pdf}.pdf + {save_pdf}.png")

# ===================== 注意力块：极致窄+超清 =====================
def draw_attn_block_hd():
    dot = Digraph(name="Attention Block", comment="Ultra Narrow HD AttnBlock", format=GRAPH_FORMAT_PDF)
    dot.attr(rankdir="LR", fontname=FONT_NAME, fontsize=FONT_SIZE, nodesep=NODE_SEP, ranksep=RANK_SEP)
    dot.attr("node", shape="box", style="filled", fontname=FONT_NAME, fontsize=FONT_SIZE, width=NODE_WIDTH, height=NODE_HEIGHT, margin=MARGIN)
    dot.attr("edge", fontname=FONT_NAME, fontsize=EDGE_FONT_SIZE, arrowsize=ARROW_SIZE, penwidth=PEN_WIDTH)

    dot.node("in_attn", label="In\n(ch,H×W)", fillcolor=COLOR_INPUT)
    dot.node("gn", label="GN", fillcolor=COLOR_ATTN)
    dot.node("qkv", label="QKV\n1×1", fillcolor=COLOR_ATTN)
    dot.node("res1", label="Resh1", fillcolor=COLOR_ATTN)
    dot.node("score", label="Score", fillcolor=COLOR_ATTN)
    dot.node("attn_out", label="Attn\nOut", fillcolor=COLOR_ATTN)
    dot.node("res2", label="Resh2", fillcolor=COLOR_ATTN)
    dot.node("proj", label="Proj\n1×1", fillcolor=COLOR_ATTN)
    dot.node("add", label="+", fillcolor=COLOR_ATTN)
    dot.node("out_attn", label="Out\n(ch,H×W)", fillcolor=COLOR_INPUT)

    dot.edge("in_attn", "gn");dot.edge("gn", "qkv");dot.edge("qkv", "res1");dot.edge("res1", "score")
    dot.edge("score", "attn_out");dot.edge("attn_out", "res2");dot.edge("res2", "proj");dot.edge("proj", "add")
    dot.edge("in_attn", "add", style="dashed", color="gray", penwidth=PEN_WIDTH);dot.edge("add", "out_attn")

    save_pdf = os.path.join(SAVE_DIR, "attention_block_ultra_narrow_HD")
    dot.render(save_pdf, view=False, format=GRAPH_FORMAT_PDF)
    dot.render(save_pdf, view=False, format=GRAPH_FORMAT_PNG)
    print(f"✅ AttnBlock 极致窄+超清图已生成: {save_pdf}.pdf + {save_pdf}.png")

# ===================== 下采样块：极致窄+超清 =====================
def draw_downsample_hd():
    dot = Digraph(name="DownSample", comment="Ultra Narrow HD DownSample", format=GRAPH_FORMAT_PDF)
    dot.attr(rankdir="LR", fontname=FONT_NAME, fontsize=FONT_SIZE, nodesep=NODE_SEP, ranksep=RANK_SEP)
    dot.attr("node", shape="box", style="filled", fontname=FONT_NAME, fontsize=FONT_SIZE, width=NODE_WIDTH, height=NODE_HEIGHT, margin=MARGIN)
    dot.attr("edge", fontname=FONT_NAME, fontsize=EDGE_FONT_SIZE, arrowsize=ARROW_SIZE, penwidth=PEN_WIDTH)

    dot.node("in_down", label="In\n(ch,H×W)", fillcolor=COLOR_INPUT)
    dot.node("conv", label="Conv\n3×3,s=2", fillcolor=COLOR_DOWN)
    dot.node("out_down", label="Out\n(ch,H/2×W/2)", fillcolor=COLOR_INPUT)
    dot.edge("in_down", "conv");dot.edge("conv", "out_down")

    save_pdf = os.path.join(SAVE_DIR, "downsample_block_ultra_narrow_HD")
    dot.render(save_pdf, view=False, format=GRAPH_FORMAT_PDF)
    dot.render(save_pdf, view=False, format=GRAPH_FORMAT_PNG)
    print(f"✅ DownSample 极致窄+超清图已生成: {save_pdf}.pdf + {save_pdf}.png")

# ===================== 上采样块：极致窄+超清 =====================
def draw_upsample_hd():
    dot = Digraph(name="UpSample", comment="Ultra Narrow HD UpSample", format=GRAPH_FORMAT_PDF)
    dot.attr(rankdir="LR", fontname=FONT_NAME, fontsize=FONT_SIZE, nodesep=NODE_SEP, ranksep=RANK_SEP)
    dot.attr("node", shape="box", style="filled", fontname=FONT_NAME, fontsize=FONT_SIZE, width=NODE_WIDTH, height=NODE_HEIGHT, margin=MARGIN)
    dot.attr("edge", fontname=FONT_NAME, fontsize=EDGE_FONT_SIZE, arrowsize=ARROW_SIZE, penwidth=PEN_WIDTH)

    dot.node("in_up", label="In\n(ch,H×W)", fillcolor=COLOR_INPUT)
    dot.node("interp", label="Interp\n×2", fillcolor=COLOR_UP)
    dot.node("conv", label="Conv\n3×3", fillcolor=COLOR_UP)
    dot.node("out_up", label="Out\n(ch,2H×2W)", fillcolor=COLOR_INPUT)
    dot.edge("in_up", "interp");dot.edge("interp", "conv");dot.edge("conv", "out_up")

    save_pdf = os.path.join(SAVE_DIR, "upsample_block_ultra_narrow_HD")
    dot.render(save_pdf, view=False, format=GRAPH_FORMAT_PDF)
    dot.render(save_pdf, view=False, format=GRAPH_FORMAT_PNG)
    print(f"✅ UpSample 极致窄+超清图已生成: {save_pdf}.pdf + {save_pdf}.png")

# ===================== 一键运行所有绘图 =====================
if __name__ == "__main__":
    create_ultra_narrow_hd_graph()
    draw_res_block_hd()
    draw_attn_block_hd()
    draw_downsample_hd()
    draw_upsample_hd()
    print(f"\n🎉 全部生成完成！文件夹：{SAVE_DIR} | 格式：PDF(无损超清) + PNG(高清) | 宽度：原始1/4 极致窄")