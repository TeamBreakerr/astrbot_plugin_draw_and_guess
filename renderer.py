"""图片渲染：手机白板 + 最终排行榜。所有 PIL 操作都封装在这里。"""
from __future__ import annotations

import asyncio
import io
import os
from dataclasses import dataclass

import aiohttp
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# 优先使用插件自带字体（font.otf/font.ttf，随插件一起打包，跨平台稳定）；
# 其次系统 CJK 字体；最后回退 PIL 默认字体（中文会变方框，但不会崩溃）。
_HERE = os.path.dirname(__file__)
_FONT_CANDIDATES = [
    os.path.join(_HERE, "font.otf"),
    os.path.join(_HERE, "font.ttf"),
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _resolve_font_path() -> str | None:
    for p in _FONT_CANDIDATES:
        if p and os.path.exists(p):
            return p
    return None


def _load_font(size: int) -> ImageFont.ImageFont:
    p = _resolve_font_path()
    if p:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def render_whiteboard(
    label: str = "",
    width: int = 1080,
    height: int = 1920,
    dot_spacing: int = 90,
) -> bytes:
    """生成手机比例（9:16）的白板 PNG。

    点阵作为绘图参考，颜色极浅以免干扰最终图像；右上角标签提示当前轮次。
    返回 PNG 字节流。
    """
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # 极浅的圆点网格作为绘画参考
    dot_color = (228, 232, 240)
    r = 2
    for y in range(dot_spacing, height, dot_spacing):
        for x in range(dot_spacing, width, dot_spacing):
            draw.ellipse((x - r, y - r, x + r, y + r), fill=dot_color)

    # 极细外框，避免 QQ 自动裁边/压缩误判图像内容
    border_color = (220, 225, 235)
    draw.rectangle((0, 0, width - 1, height - 1), outline=border_color, width=2)

    # 角标
    if label:
        font = _load_font(34)
        pad_x, pad_y = 28, 22
        tw = draw.textlength(label, font=font)
        # 右上角浅色描边胶囊
        bbox = (
            width - pad_x - tw - 18,
            pad_y - 6,
            width - pad_x + 6,
            pad_y + 44,
        )
        draw.rounded_rectangle(bbox, radius=16, fill=(245, 247, 252), outline=border_color, width=1)
        draw.text(
            (width - pad_x - tw + 0, pad_y),
            label,
            font=font,
            fill=(140, 150, 170),
        )

    # 左下角小提示
    hint_font = _load_font(26)
    hint = "长按本图 → 编辑 → 涂鸦 → 完成后回复 /draw_submit"
    hw = draw.textlength(hint, font=hint_font)
    if hw < width - 60:
        draw.text((30, height - 60), hint, font=hint_font, fill=(180, 188, 205))

    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


@dataclass
class RankRow:
    rank: int
    name: str
    score: int
    avatar_url: str
    user_id: str


async def _fetch_bytes(url: str, timeout: float = 10.0) -> bytes | None:
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as s:
            async with s.get(url) as r:
                if r.status != 200:
                    return None
                return await r.read()
    except Exception:
        return None


def _circle_avatar(raw: bytes | None, size: int) -> Image.Image:
    """把头像裁成圆形；失败时画一个占位灰圈。"""
    if raw:
        try:
            im = Image.open(io.BytesIO(raw)).convert("RGBA")
        except Exception:
            im = None
    else:
        im = None
    if im is None:
        im = Image.new("RGBA", (size, size), (200, 200, 200, 255))
    im = im.resize((size, size), Image.Resampling.LANCZOS)

    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size, size), fill=255)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(im, (0, 0), mask)
    return out


def _gradient_bg(width: int, height: int, top: tuple, bottom: tuple) -> Image.Image:
    bg = Image.new("RGB", (width, height), top)
    draw = ImageDraw.Draw(bg)
    for y in range(height):
        ratio = y / max(height - 1, 1)
        r = int(top[0] * (1 - ratio) + bottom[0] * ratio)
        g = int(top[1] * (1 - ratio) + bottom[1] * ratio)
        b = int(top[2] * (1 - ratio) + bottom[2] * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    return bg


def _medal_color(rank: int) -> tuple:
    if rank == 1:
        return (255, 196, 36)   # gold
    if rank == 2:
        return (192, 200, 216)  # silver
    if rank == 3:
        return (205, 127, 50)   # bronze
    return (200, 205, 215)


def _draw_rounded_rect(draw: ImageDraw.ImageDraw, xy, radius: int, fill, outline=None, width: int = 1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def _text_center(draw, xy, text, font, fill):
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx, cy = xy
    draw.text((cx - w / 2, cy - h / 2 - bbox[1]), text, font=font, fill=fill)


async def render_scoreboard(
    title: str,
    subtitle: str,
    rows: list[RankRow],
) -> bytes:
    """渲染最终积分排行榜，返回 PNG 字节。"""
    # 异步下载所有头像
    avatar_bytes_list = await asyncio.gather(
        *[_fetch_bytes(r.avatar_url) for r in rows],
        return_exceptions=False,
    )

    # 布局参数
    width = 900
    pad = 32
    header_h = 150
    row_h = 110
    row_gap = 14
    footer_h = 60
    height = header_h + len(rows) * (row_h + row_gap) - row_gap + footer_h + pad * 2

    # 字体
    font_title = _load_font(46)
    font_subtitle = _load_font(20)
    font_rank = _load_font(40)
    font_name = _load_font(28)
    font_score_num = _load_font(36)
    font_score_label = _load_font(16)
    font_footer = _load_font(14)

    bg = _gradient_bg(width, height, (245, 247, 255), (255, 250, 240))
    draw = ImageDraw.Draw(bg, "RGBA")

    # 标题区域
    draw.text((pad + 4, pad), title, font=font_title, fill=(38, 50, 100))
    draw.text(
        (pad + 4, pad + 64),
        subtitle,
        font=font_subtitle,
        fill=(110, 120, 150),
    )

    # 行
    y = pad + header_h
    for i, row in enumerate(rows):
        raw = avatar_bytes_list[i] if i < len(avatar_bytes_list) else None
        # 卡片
        card_x0 = pad
        card_y0 = y
        card_x1 = width - pad
        card_y1 = y + row_h
        # 顶 3 名卡片底色与左侧色条
        card_fill = (255, 255, 255, 235) if row.rank > 3 else (255, 250, 230, 245)
        _draw_rounded_rect(
            draw,
            (card_x0, card_y0, card_x1, card_y1),
            radius=18,
            fill=card_fill,
            outline=(220, 225, 235),
            width=1,
        )
        # 左侧色条（排名牌）
        stripe_w = 12
        _draw_rounded_rect(
            draw,
            (card_x0, card_y0, card_x0 + stripe_w, card_y1),
            radius=6,
            fill=_medal_color(row.rank),
        )

        # 排名数字
        rank_cx = card_x0 + stripe_w + 50
        rank_cy = card_y0 + row_h // 2
        _text_center(draw, (rank_cx, rank_cy), f"#{row.rank}", font_rank, (60, 70, 110))

        # 头像
        avatar_size = row_h - 24
        ax = card_x0 + stripe_w + 110
        ay = card_y0 + 12
        avatar = _circle_avatar(raw, avatar_size)
        # 给头像加一圈描边
        ring = Image.new("RGBA", (avatar_size + 6, avatar_size + 6), (0, 0, 0, 0))
        ImageDraw.Draw(ring).ellipse(
            (0, 0, avatar_size + 6, avatar_size + 6),
            fill=_medal_color(row.rank) + (255,),
        )
        bg.paste(ring, (ax - 3, ay - 3), ring)
        bg.paste(avatar, (ax, ay), avatar)

        # 昵称（裁剪）
        name = row.name
        max_name_w = card_x1 - (ax + avatar_size + 28) - 200
        if max_name_w < 100:
            max_name_w = 100
        # 简单截断 + 省略号
        if draw.textlength(name, font=font_name) > max_name_w:
            while name and draw.textlength(name + "…", font=font_name) > max_name_w:
                name = name[:-1]
            name = name + "…"
        draw.text(
            (ax + avatar_size + 24, card_y0 + row_h // 2 - 18),
            name,
            font=font_name,
            fill=(40, 50, 80),
        )
        # 副信息：QQ 号
        draw.text(
            (ax + avatar_size + 24, card_y0 + row_h // 2 + 16),
            f"ID: {row.user_id}",
            font=font_subtitle,
            fill=(140, 150, 170),
        )

        # 右侧分数
        score_text = str(row.score)
        sw = draw.textlength(score_text, font=font_score_num)
        sx = card_x1 - 32 - sw
        draw.text(
            (sx, card_y0 + row_h // 2 - 30),
            score_text,
            font=font_score_num,
            fill=(220, 80, 60) if row.rank == 1 else (60, 70, 110),
        )
        label = "分"
        lx = card_x1 - 32 - draw.textlength(label, font=font_score_label)
        draw.text(
            (lx, card_y0 + row_h // 2 + 14),
            label,
            font=font_score_label,
            fill=(120, 130, 150),
        )

        y += row_h + row_gap

    # 页脚
    draw.text(
        (pad + 4, height - pad - 20),
        "你画我猜 · 游戏结束",
        font=font_footer,
        fill=(150, 158, 180),
    )

    out = io.BytesIO()
    bg.convert("RGB").save(out, format="PNG", optimize=True)
    return out.getvalue()
