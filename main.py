"""你画我猜（Draw & Guess）AstrBot 插件 — 竞猜模式。

游戏流程：
  1. 在群里 `/draw_create` 开房，创建者自动加入；
  2. 其他玩家 `/draw_join`；可 `/draw_leave` / `/draw_list`；
  3. 房主 `/draw_start <领域>` 开始游戏；
  4. 系统随机分配每个人作画一次的顺序；
  5. 题目由 LLM 生成，私聊发给当前作画者，群里只公布字数；
  6. 作画者 `/draw_submit` 并随消息附上一张图片，进入竞猜状态；
  7. 此时插件激活，捕捉群内所有玩家的发言，第一条 *完全匹配* 答案者得分；
  8. 全部回合结束后渲染图片排行榜，游戏结束。
"""
from __future__ import annotations

import asyncio
import random
import re
import time
from dataclasses import dataclass, field
from enum import Enum

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star, register
from astrbot.core.platform import MessageType as PlatformMessageType
from astrbot.core.platform.message_session import MessageSession

from .renderer import RankRow, render_scoreboard


# ------------------------- 数据模型 -------------------------


class RoomState(str, Enum):
    LOBBY = "lobby"          # 等待玩家加入
    DRAWING = "drawing"      # 当前作画者已收到题目，等待提交画作
    GUESSING = "guessing"    # 画作公布，竞猜中
    ENDED = "ended"          # 游戏结束（瞬态，结算完成后房间被清掉）


@dataclass
class Player:
    user_id: str
    name: str
    avatar_url: str = ""
    score: int = 0


@dataclass
class Room:
    group_origin: str
    group_id: str
    platform_id: str
    owner_id: str
    players: dict[str, Player] = field(default_factory=dict)
    state: RoomState = RoomState.LOBBY
    domain: str = ""
    order: list[str] = field(default_factory=list)
    round_index: int = 0
    current_answer: str = ""
    normalized_answer: str = ""
    round_deadline: float = 0.0
    timeout_task: asyncio.Task | None = None
    created_at: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def current_drawer_id(self) -> str:
        if 0 <= self.round_index < len(self.order):
            return self.order[self.round_index]
        return ""


# ------------------------- 工具函数 -------------------------


# 匹配仅由空白 / 标点拼成的串，避免空字符串误判
_NORMALIZE_STRIP = re.compile(r"[\s　\-—_·,，。.！!?？:：;；'\"“”‘’()（）\[\]【】<>《》/\\]+")


def normalize_answer(s: str) -> str:
    if not s:
        return ""
    return _NORMALIZE_STRIP.sub("", s).strip().lower()


def qq_avatar_url(user_id: str, size: int = 640) -> str:
    return f"https://q1.qlogo.cn/g?b=qq&nk={user_id}&s={size}"


def _extract_image_component(event: AstrMessageEvent) -> Comp.Image | None:
    comps = getattr(event.message_obj, "message", []) or []
    for c in comps:
        if isinstance(c, Comp.Image):
            return c
    return None


def _looks_like_command(text: str) -> bool:
    """命令一般以 / # ! 开头；用来在 GUESSING 状态下跳过命令消息。"""
    if not text:
        return True
    t = text.lstrip()
    return bool(t) and t[0] in ("/", "#", "!", "！", "／")


# ------------------------- 插件本体 -------------------------


@register(
    "astrbot_plugin_draw_and_guess",
    "TeamBreaker",
    "你画我猜竞猜模式：LLM 出题 + 群内作画 + 激活竞猜 + 图片排行榜",
    "1.0.0",
    "https://github.com/TeamBreakerr/astrbot_plugin_draw_and_guess",
)
class DrawAndGuessPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.config = config or {}
        self.rooms: dict[str, Room] = {}

        self.llm_provider_id = str(self.config.get("llm_provider_id", "")).strip()
        self.min_players = max(2, int(self.config.get("min_players", 2)))
        self.max_players = max(self.min_players, int(self.config.get("max_players", 8)))
        self.draw_timeout = int(self.config.get("draw_timeout", 300))
        self.guess_timeout = int(self.config.get("guess_timeout", 180))
        self.score_correct = int(self.config.get("score_correct", 2))
        self.score_drawer = int(self.config.get("score_drawer", 1))
        self.answer_min_len = max(1, int(self.config.get("answer_min_len", 2)))
        self.answer_max_len = max(self.answer_min_len, int(self.config.get("answer_max_len", 6)))

    async def terminate(self) -> None:
        for room in list(self.rooms.values()):
            if room.timeout_task and not room.timeout_task.done():
                room.timeout_task.cancel()
        self.rooms.clear()

    # --------------- 房间 / 玩家辅助 ---------------

    def _get_room(self, event: AstrMessageEvent) -> Room | None:
        return self.rooms.get(event.unified_msg_origin)

    def _require_group(self, event: AstrMessageEvent) -> bool:
        return event.get_message_type() == PlatformMessageType.GROUP_MESSAGE

    def _player_from_event(self, event: AstrMessageEvent) -> Player:
        uid = str(event.get_sender_id())
        name = event.get_sender_name() or f"用户{uid}"
        return Player(user_id=uid, name=name, avatar_url=qq_avatar_url(uid))

    def _cancel_timeout(self, room: Room) -> None:
        if room.timeout_task and not room.timeout_task.done():
            room.timeout_task.cancel()
        room.timeout_task = None

    async def _send_group(self, room: Room, chain: list) -> None:
        try:
            await self.context.send_message(room.group_origin, MessageChain(chain=chain))
        except Exception:
            logger.exception("draw_and_guess: 群消息发送失败 origin=%s", room.group_origin)

    async def _send_private(self, room: Room, user_id: str, text: str) -> bool:
        try:
            session = MessageSession(
                room.platform_id,
                PlatformMessageType.FRIEND_MESSAGE,
                str(user_id),
            )
            return bool(
                await self.context.send_message(session, MessageChain(chain=[Comp.Plain(text)]))
            )
        except Exception:
            logger.exception("draw_and_guess: 私聊发送失败 uid=%s", user_id)
            return False

    # --------------- LLM 出题 ---------------

    async def _generate_topic(self, domain: str) -> str | None:
        provider = None
        if self.llm_provider_id:
            try:
                provider = self.context.get_provider_by_id(self.llm_provider_id)
            except Exception:
                logger.exception(
                    "draw_and_guess: get_provider_by_id(%s) 异常", self.llm_provider_id
                )
        if provider is None:
            try:
                provider = self.context.get_using_provider()
            except Exception:
                provider = None
        if provider is None:
            logger.warning("draw_and_guess: 没有可用的 LLM provider，无法出题")
            return None

        prompt = (
            f"你是“你画我猜”游戏的出题官。请在领域【{domain}】内为这一局生成一个题目。\n"
            "要求：\n"
            f"1. 必须是一个具体的、可被绘制的事物/动作/概念（普通玩家能在白板上画出来）；\n"
            f"2. 字数（汉字数）在 {self.answer_min_len}-{self.answer_max_len} 之间；\n"
            "3. 难度适中，不要过于冷门，但也不要太简单；\n"
            "4. 不要使用标点、括号、引号、英文或解释；\n"
            "5. 只输出题目本身，不要任何前缀、后缀、引言或解释。"
        )
        try:
            resp = await provider.text_chat(
                prompt=prompt,
                session_id=f"draw_and_guess:{int(time.time() * 1000)}",
            )
        except Exception:
            logger.exception("draw_and_guess: LLM 调用异常")
            return None
        text = (getattr(resp, "completion_text", None) or "").strip()
        # 取第一行，去除常见包裹符号
        line = text.splitlines()[0].strip() if text else ""
        line = line.strip("「」“”\"'`*《》()（）[]【】 \t")
        # 移除空白
        line = re.sub(r"\s+", "", line)
        if not line:
            return None
        # 字数校验：超过限制则截断；过短则放行（让游戏继续，避免反复重试）
        if len(line) > self.answer_max_len * 2:
            line = line[: self.answer_max_len * 2]
        return line

    # --------------- 状态转换 ---------------

    async def _begin_round(self, room: Room) -> None:
        """开始当前 round_index 对应玩家的作画轮。"""
        drawer_id = room.current_drawer_id
        if not drawer_id:
            await self._finish_game(room)
            return
        drawer = room.players.get(drawer_id)
        if drawer is None:
            # 作画者中途退出 → 跳过
            room.round_index += 1
            await self._begin_round(room)
            return

        # 出题
        topic = await self._generate_topic(room.domain)
        if not topic:
            await self._send_group(room, [Comp.Plain("❌ LLM 出题失败，本轮跳过。")])
            room.round_index += 1
            await self._begin_round(room)
            return

        room.current_answer = topic
        room.normalized_answer = normalize_answer(topic)
        room.state = RoomState.DRAWING

        # 私聊作画者
        private_ok = await self._send_private(
            room,
            drawer_id,
            f"🎨 你画我猜 · 第 {room.round_index + 1}/{len(room.order)} 轮\n"
            f"群: {room.group_id} · 领域: {room.domain}\n"
            f"你的题目是：【{topic}】\n"
            f"请在 {self.draw_timeout} 秒内回到群里发送 /draw_submit 并附上一张作画图片；\n"
            f"或发送 /draw_skip 跳过本轮（不得分）。\n"
            "⚠️ 不要把题目透露给其他玩家！"
        )
        if not private_ok:
            await self._send_group(
                room,
                [
                    Comp.At(qq=drawer_id),
                    Comp.Plain(
                        " 私聊不可达。请先加机器人为好友后再试。"
                        "本轮已自动跳过。"
                    ),
                ],
            )
            room.round_index += 1
            await self._begin_round(room)
            return

        # 群内宣布
        await self._send_group(
            room,
            [
                Comp.Plain(f"🎮 第 {room.round_index + 1}/{len(room.order)} 轮开始！\n本轮作画者："),
                Comp.At(qq=drawer_id),
                Comp.Plain(
                    f"（{drawer.name}）\n"
                    f"领域：{room.domain}\n"
                    f"答案共 {len(room.current_answer)} 个字\n"
                    f"请作画者私聊查收题目，并在 {self.draw_timeout} 秒内于群内 /draw_submit 提交画作。"
                ),
            ],
        )

        # 启动作画超时
        room.round_deadline = time.time() + self.draw_timeout
        self._cancel_timeout(room)
        room.timeout_task = asyncio.create_task(self._draw_timeout_watcher(room))

    async def _draw_timeout_watcher(self, room: Room) -> None:
        try:
            await asyncio.sleep(self.draw_timeout)
        except asyncio.CancelledError:
            return
        async with room.lock:
            if room.state != RoomState.DRAWING:
                return
            await self._send_group(
                room,
                [Comp.Plain(f"⏰ 作画超时，本轮答案是【{room.current_answer}】。")],
            )
            room.round_index += 1
            await self._begin_round(room)

    async def _guess_timeout_watcher(self, room: Room) -> None:
        try:
            await asyncio.sleep(self.guess_timeout)
        except asyncio.CancelledError:
            return
        async with room.lock:
            if room.state != RoomState.GUESSING:
                return
            await self._send_group(
                room,
                [Comp.Plain(f"⏰ 竞猜超时，本轮无人答对。答案是【{room.current_answer}】。")],
            )
            room.round_index += 1
            await self._begin_round(room)

    async def _finish_game(self, room: Room) -> None:
        room.state = RoomState.ENDED
        self._cancel_timeout(room)

        # 排序：分数高→低；同分按玩家加入顺序
        join_order = {uid: i for i, uid in enumerate(room.players.keys())}
        ranked = sorted(
            room.players.values(),
            key=lambda p: (-p.score, join_order.get(p.user_id, 0)),
        )
        rows = [
            RankRow(
                rank=i + 1,
                name=p.name,
                score=p.score,
                avatar_url=p.avatar_url or qq_avatar_url(p.user_id),
                user_id=p.user_id,
            )
            for i, p in enumerate(ranked)
        ]
        try:
            img_bytes = await render_scoreboard(
                title="你画我猜 · 最终排行榜",
                subtitle=f"领域：{room.domain or '未指定'} · 共 {len(room.order)} 轮 · 玩家 {len(ranked)} 人",
                rows=rows,
            )
        except Exception:
            logger.exception("draw_and_guess: 渲染排行榜失败")
            text_lines = ["🏁 游戏结束 · 最终排行榜"]
            for r in rows:
                text_lines.append(f"#{r.rank}  {r.name}  {r.score} 分")
            await self._send_group(room, [Comp.Plain("\n".join(text_lines))])
        else:
            await self._send_group(
                room,
                [
                    Comp.Plain("🏁 游戏结束！最终排行榜："),
                    Comp.Image.fromBytes(img_bytes),
                ],
            )
        self.rooms.pop(room.group_origin, None)

    # ------------------------- 指令 -------------------------

    @filter.command("draw_create", alias={"创建画猜", "画猜创建", "draw_open"})
    async def cmd_create(self, event: AstrMessageEvent):
        if not self._require_group(event):
            yield event.plain_result("❌ 只能在群聊中创建你画我猜房间。")
            return
        origin = event.unified_msg_origin
        if origin in self.rooms:
            yield event.plain_result("⚠️ 本群已有房间。使用 /draw_list 查看，或 /draw_disband 由房主解散。")
            return

        creator = self._player_from_event(event)
        room = Room(
            group_origin=origin,
            group_id=str(event.get_group_id() or ""),
            platform_id=event.get_platform_id(),
            owner_id=creator.user_id,
            players={creator.user_id: creator},
        )
        self.rooms[origin] = room
        yield event.plain_result(
            "🎨 你画我猜房间已创建！\n"
            f"房主：{creator.name}（{creator.user_id}）\n"
            f"其他玩家发送 /draw_join 加入；房主用 /draw_start <领域> 开始游戏。\n"
            f"当前 1/{self.max_players} 人。"
        )

    @filter.command("draw_join", alias={"加入画猜", "画猜加入"})
    async def cmd_join(self, event: AstrMessageEvent):
        if not self._require_group(event):
            return
        room = self._get_room(event)
        if room is None:
            yield event.plain_result("❌ 本群没有进行中的房间。使用 /draw_create 创建。")
            return
        if room.state != RoomState.LOBBY:
            yield event.plain_result("⚠️ 游戏已开始，不能再加入。")
            return
        if len(room.players) >= self.max_players:
            yield event.plain_result(f"⚠️ 房间已满（{self.max_players} 人）。")
            return
        player = self._player_from_event(event)
        if player.user_id in room.players:
            yield event.plain_result("你已经在房间里了。")
            return
        room.players[player.user_id] = player
        yield event.plain_result(
            f"✅ {player.name} 加入房间。当前 {len(room.players)}/{self.max_players} 人。"
        )

    @filter.command("draw_leave", alias={"退出画猜", "画猜退出"})
    async def cmd_leave(self, event: AstrMessageEvent):
        if not self._require_group(event):
            return
        room = self._get_room(event)
        if room is None:
            return
        uid = str(event.get_sender_id())
        if uid not in room.players:
            yield event.plain_result("你不在房间里。")
            return

        # 大厅阶段可自由退出；房主退出则房间解散
        if room.state == RoomState.LOBBY:
            name = room.players[uid].name
            del room.players[uid]
            if uid == room.owner_id:
                self.rooms.pop(room.group_origin, None)
                yield event.plain_result(f"👋 房主 {name} 退出，房间已解散。")
                return
            yield event.plain_result(f"👋 {name} 退出。当前 {len(room.players)}/{self.max_players} 人。")
            return

        # 游戏中：把玩家移出，分数清零；若是当前作画者，则本轮跳过
        async with room.lock:
            player = room.players.pop(uid, None)
            if uid in room.order:
                # 不动 order 列表，保持轮次稳定；仅作画时识别"玩家已退出"再跳过
                pass
            yield event.plain_result(f"👋 {player.name if player else uid} 中途退出。")
            if uid == room.current_drawer_id and room.state in (RoomState.DRAWING, RoomState.GUESSING):
                self._cancel_timeout(room)
                await self._send_group(
                    room,
                    [Comp.Plain(f"作画者中途退出，本轮取消。答案是【{room.current_answer}】。")],
                )
                room.round_index += 1
                await self._begin_round(room)

    @filter.command("draw_list", alias={"画猜成员", "画猜列表", "draw_room"})
    async def cmd_list(self, event: AstrMessageEvent):
        if not self._require_group(event):
            return
        room = self._get_room(event)
        if room is None:
            yield event.plain_result("❌ 本群没有房间。/draw_create 创建。")
            return
        owner = room.players.get(room.owner_id)
        lines = [
            f"🎨 你画我猜房间状态",
            f"状态：{room.state.value}",
            f"房主：{owner.name if owner else room.owner_id}",
            f"领域：{room.domain or '（未开始）'}",
            f"玩家 {len(room.players)}/{self.max_players}：",
        ]
        for i, p in enumerate(room.players.values(), 1):
            marker = ""
            if room.state != RoomState.LOBBY:
                if room.order and p.user_id == room.current_drawer_id:
                    marker = " 🖌"
                if p.user_id == room.owner_id:
                    marker += " 👑"
            else:
                if p.user_id == room.owner_id:
                    marker = " 👑"
            lines.append(f"  {i}. {p.name}（{p.score}分）{marker}")
        if room.state != RoomState.LOBBY and room.order:
            lines.append(f"进度：第 {room.round_index + 1}/{len(room.order)} 轮")
        yield event.plain_result("\n".join(lines))

    @filter.command("draw_help", alias={"画猜帮助", "画猜说明", "你画我猜帮助"})
    async def cmd_help(self, event: AstrMessageEvent):
        text = (
            "🎨 你画我猜（Draw & Guess · 竞猜模式）\n"
            "————————————————\n"
            "【玩法】\n"
            "1. 任一玩家在群里 /draw_create 开房，开房者自动加入并成为房主。\n"
            "2. 大厅阶段，其他玩家发 /draw_join 加入；可随时 /draw_leave 退出；\n"
            "   /draw_list 查看当前成员；房主可 /draw_disband 解散房间。\n"
            f"3. 人凑齐后（默认 ≥ {self.min_players} 人），房主 /draw_start <领域> 开始游戏。\n"
            "   <领域> 是出题的范围，比如「动物」「日常生活」「电影名」「水果」等。\n"
            "4. 系统随机分配作画顺序，每人轮流作画 1 次。\n"
            "5. 每一轮：\n"
            "   · LLM 根据领域生成一个题目；\n"
            "   · 题目通过【私聊】发给当前作画者（请确保 bot 已加为好友）；\n"
            "   · 群里宣布作画者、领域、答案字数（不公布答案）；\n"
            "   · 作画者在群里发 /draw_submit 并随消息附 1 张作画图片；\n"
            "   · 提交后竞猜激活，机器人开始监听群消息；\n"
            f"   · 第一个【完全匹配】答案的玩家获得 {self.score_correct} 分，"
            f"作画者获得 {self.score_drawer} 分；\n"
            f"   · 作画 {self.draw_timeout}s / 竞猜 {self.guess_timeout}s 超时则跳过该轮。\n"
            "6. 全部回合结束，机器人生成最终积分排行榜图片（头像 + 昵称 + 积分 + 顺位）。\n"
            "\n【指令】\n"
            "  /draw_create               创建房间（创建者自动加入）\n"
            "  /draw_join                 加入当前群的房间（大厅阶段）\n"
            "  /draw_leave                退出房间（房主退出则解散）\n"
            "  /draw_list                 查看房间状态和成员\n"
            "  /draw_disband              房主解散房间\n"
            "  /draw_start <领域>         房主开始游戏，输入领域\n"
            "  /draw_submit               作画者提交画作（须随消息附图）\n"
            "  /draw_skip                 作画者/房主跳过本轮（不得分）\n"
            "  /draw_status               查看当前游戏进度\n"
            "  /draw_help                 显示这段帮助\n"
            "\n【匹配规则】\n"
            "  · 答案去除空白和常见标点后必须完全一致；\n"
            "  · 英文不区分大小写；\n"
            "  · 命令消息（以 / # ! 开头）不会被识别为竞猜内容；\n"
            "  · 作画者本人发言不会被识别为竞猜内容。\n"
            "\n【小贴士】\n"
            "  · 作画者要先把机器人加为 QQ 好友，否则无法收到私聊题目；\n"
            "  · 领域越具体，题目质量越稳定；\n"
            "  · 别把答案告诉别的玩家 😉"
        )
        yield event.plain_result(text)

    @filter.command("draw_disband", alias={"解散画猜", "画猜解散"})
    async def cmd_disband(self, event: AstrMessageEvent):
        if not self._require_group(event):
            return
        room = self._get_room(event)
        if room is None:
            return
        uid = str(event.get_sender_id())
        if uid != room.owner_id:
            yield event.plain_result("⚠️ 仅房主可解散房间。")
            return
        self._cancel_timeout(room)
        self.rooms.pop(room.group_origin, None)
        yield event.plain_result("🛑 房间已解散。")

    @filter.command("draw_start", alias={"开始画猜", "画猜开始"})
    async def cmd_start(self, event: AstrMessageEvent, domain: str = ""):
        if not self._require_group(event):
            return
        room = self._get_room(event)
        if room is None:
            yield event.plain_result("❌ 本群没有房间。/draw_create 创建。")
            return
        uid = str(event.get_sender_id())
        if uid != room.owner_id:
            yield event.plain_result("⚠️ 仅房主可开始游戏。")
            return
        if room.state != RoomState.LOBBY:
            yield event.plain_result("⚠️ 游戏已在进行中。")
            return
        domain = (domain or "").strip()
        if not domain:
            yield event.plain_result("用法：/draw_start <领域>\n例如：/draw_start 动物")
            return
        if len(room.players) < self.min_players:
            yield event.plain_result(
                f"⚠️ 至少需要 {self.min_players} 名玩家才能开始（当前 {len(room.players)} 人）。"
            )
            return

        room.domain = domain
        ids = list(room.players.keys())
        random.shuffle(ids)
        room.order = ids
        room.round_index = 0

        order_names = "→".join(room.players[i].name for i in ids)
        yield event.plain_result(
            f"🚀 游戏开始！\n领域：{domain}\n作画顺序：{order_names}\n马上为第一位作画者出题…"
        )
        # 开始第一轮（异步）
        asyncio.create_task(self._begin_round_with_lock(room))

    async def _begin_round_with_lock(self, room: Room) -> None:
        async with room.lock:
            await self._begin_round(room)

    @filter.command("draw_submit", alias={"提交画作", "画猜提交", "submit_draw"})
    async def cmd_submit(self, event: AstrMessageEvent):
        if not self._require_group(event):
            return
        room = self._get_room(event)
        if room is None:
            return
        uid = str(event.get_sender_id())
        if room.state != RoomState.DRAWING:
            yield event.plain_result("⚠️ 当前不在作画阶段。")
            return
        if uid != room.current_drawer_id:
            yield event.plain_result("⚠️ 只有当前作画者可以提交。")
            return
        img = _extract_image_component(event)
        if img is None:
            yield event.plain_result("⚠️ 请在 /draw_submit 命令同一条消息中附上一张图片。")
            return

        async with room.lock:
            if room.state != RoomState.DRAWING:
                return
            self._cancel_timeout(room)
            room.state = RoomState.GUESSING
            room.round_deadline = time.time() + self.guess_timeout
            room.timeout_task = asyncio.create_task(self._guess_timeout_watcher(room))

            await self._send_group(
                room,
                [
                    Comp.Plain(
                        f"🖼️ 画作已提交！开始竞猜，第一个完全匹配答案的玩家获胜。\n"
                        f"答案共 {len(room.current_answer)} 个字 · 时限 {self.guess_timeout} 秒\n"
                        "直接在群里发言即可（无需 @ 机器人）。"
                    ),
                ],
            )

    @filter.command("draw_skip", alias={"跳过画猜", "画猜跳过"})
    async def cmd_skip(self, event: AstrMessageEvent):
        if not self._require_group(event):
            return
        room = self._get_room(event)
        if room is None:
            return
        uid = str(event.get_sender_id())
        if room.state not in (RoomState.DRAWING, RoomState.GUESSING):
            return
        if uid != room.current_drawer_id and uid != room.owner_id:
            yield event.plain_result("⚠️ 只有当前作画者或房主可跳过本轮。")
            return
        async with room.lock:
            if room.state not in (RoomState.DRAWING, RoomState.GUESSING):
                return
            self._cancel_timeout(room)
            await self._send_group(
                room,
                [Comp.Plain(f"⏭️ 本轮已跳过，答案是【{room.current_answer}】。")],
            )
            room.round_index += 1
            await self._begin_round(room)

    @filter.command("draw_status", alias={"画猜状态"})
    async def cmd_status(self, event: AstrMessageEvent):
        if not self._require_group(event):
            return
        room = self._get_room(event)
        if room is None:
            yield event.plain_result("❌ 本群没有房间。")
            return
        bits = [f"状态：{room.state.value}"]
        if room.state in (RoomState.DRAWING, RoomState.GUESSING) and room.order:
            drawer = room.players.get(room.current_drawer_id)
            bits.append(f"第 {room.round_index + 1}/{len(room.order)} 轮")
            if drawer:
                bits.append(f"作画者：{drawer.name}")
            bits.append(f"答案字数：{len(room.current_answer)}")
            remaining = max(0, int(room.round_deadline - time.time()))
            bits.append(f"剩余 {remaining} 秒")
        yield event.plain_result(" · ".join(bits))

    # ------------------------- 竞猜捕捉 -------------------------

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=50)
    async def on_group_message(self, event: AstrMessageEvent):
        """在 GUESSING 状态下识别玩家发言并匹配答案。"""
        room = self._get_room(event)
        if room is None or room.state != RoomState.GUESSING:
            return
        uid = str(event.get_sender_id())
        if uid not in room.players:
            return
        if uid == room.current_drawer_id:
            return  # 作画者不参与猜测
        text = (event.message_str or "").strip()
        if not text or _looks_like_command(text):
            return

        guess_norm = normalize_answer(text)
        if not guess_norm or guess_norm != room.normalized_answer:
            return

        # 命中：抢一把锁，确保只有第一个生效
        async with room.lock:
            if room.state != RoomState.GUESSING:
                return
            self._cancel_timeout(room)
            guesser = room.players.get(uid)
            drawer = room.players.get(room.current_drawer_id)
            if guesser:
                guesser.score += self.score_correct
            if drawer:
                drawer.score += self.score_drawer
            # 阻止其他插件继续处理该条消息（例如 LLM 自动回复）
            try:
                event.stop_event()
            except Exception:
                pass
            await self._send_group(
                room,
                [
                    Comp.Plain(
                        f"🎯 答对啦！{guesser.name if guesser else uid} +{self.score_correct} 分\n"
                        f"作画者 {drawer.name if drawer else ''} +{self.score_drawer} 分\n"
                        f"答案：【{room.current_answer}】"
                    ),
                ],
            )
            room.round_index += 1
            await self._begin_round(room)
