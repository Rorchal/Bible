"""解析「章→节→段 + offset」式的目录文件。

这是第二步的另一条路：不让模型分段，而是用人已经整理好的目录，
按字符 offset 从原文里精确切片。正文逐字来自原文，不经模型改写。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

OUTLINE_DIR = Path(__file__).resolve().parent.parent / "data" / "00_outlines"

# 目录里常见各种连字符/破折号，统一成 ASCII 再匹配
_DASHES = str.maketrans({"‐": "-", "‑": "-", "‒": "-",
                         "–": "-", "—": "-", "−": "-",
                         "－": "-"})

_CHAPTER = re.compile(r"^##\s+(?!#)(.+?)\s*$", re.M)
_SECTION = re.compile(r"^###\s+(.+?)\s*$", re.M)
_ROW = re.compile(r"^\|\s*([0-9]+-[0-9]+)\s*\|(.*?)\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*$")
_TOTAL = re.compile(r"总字符长度[：:]\s*(\d+)")


@dataclass
class Segment:
    seg_no: str        # 段序号，如 "2-3"
    summary: str       # 语义段落内容简述
    start: int         # 闭区间起点（字符）
    end: int           # 闭区间终点（字符）
    chapter: str = ""  # 所属章标题
    section: str = ""  # 所属节标题
    text: str = ""     # 切片后填入

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass
class Outline:
    title: str
    segments: list[Segment] = field(default_factory=list)
    declared_total: int | None = None
    source_name: str = ""


def _strip_offset_suffix(heading: str) -> str:
    """去掉章标题末尾的「（offset：0-1162）」。"""
    return re.sub(r"[（(]\s*offset[：:].*?[）)]\s*$", "", heading).strip()


def parse_outline(path: Path) -> Outline:
    raw = path.read_text(encoding="utf-8").translate(_DASHES)

    title_match = re.search(r"^#\s+(.+?)\s*$", raw, re.M)
    title = title_match.group(1).strip() if title_match else path.stem

    total_match = _TOTAL.search(raw)
    declared_total = int(total_match.group(1)) if total_match else None

    name_match = re.search(r"文档名称[：:]\s*(\S+)", raw)
    source_name = name_match.group(1).strip() if name_match else ""

    segments: list[Segment] = []
    chapter = section = ""
    for line in raw.split("\n"):
        stripped = line.strip()

        if stripped.startswith("### "):
            section = stripped[4:].strip()
            continue
        if stripped.startswith("## "):
            chapter = _strip_offset_suffix(stripped[3:].strip())
            section = ""
            continue

        row = _ROW.match(stripped)
        if row:
            seg_no, summary, start, end = row.groups()
            segments.append(
                Segment(
                    seg_no=seg_no,
                    summary=summary.strip(),
                    start=int(start),
                    end=int(end),
                    chapter=chapter,
                    section=section,
                )
            )

    return Outline(title, segments, declared_total, source_name)


@dataclass
class Issue:
    level: str   # "错误" | "警告"
    message: str


def validate(outline: Outline, text_length: int | None = None) -> list[Issue]:
    """检查 offset 的连续性、边界和与原文长度的一致性。"""
    issues: list[Issue] = []
    segs = outline.segments

    if not segs:
        return [Issue("错误", "目录里没有解析到任何段落行")]

    for s in segs:
        if s.end < s.start:
            issues.append(Issue("错误", f"{s.seg_no} 终点 {s.end} 小于起点 {s.start}"))

    if segs[0].start != 0:
        issues.append(Issue("警告", f"首段起点是 {segs[0].start}，不是 0"))

    for prev, cur in zip(segs, segs[1:]):
        gap = cur.start - prev.end
        if gap > 1:
            issues.append(
                Issue("错误", f"{prev.seg_no} 与 {cur.seg_no} 之间漏了 {gap - 1} 个字符"
                              f"（{prev.end + 1}–{cur.start - 1}）")
            )
        elif gap < 1:
            issues.append(
                Issue("错误", f"{prev.seg_no} 与 {cur.seg_no} 重叠 {1 - gap} 个字符")
            )

    covered = segs[-1].end - segs[0].start + 1
    if outline.declared_total is not None and covered != outline.declared_total:
        issues.append(
            Issue("警告", f"按闭区间覆盖 {covered} 字，元信息声明 {outline.declared_total} 字，"
                          f"差 {covered - outline.declared_total}")
        )

    if text_length is not None and segs[-1].end >= text_length:
        issues.append(
            Issue("警告", f"末段终点 {segs[-1].end} 超出原文长度 {text_length}"
                          f"（最大下标 {text_length - 1}），末段将被截到原文结尾")
        )

    return issues


def slice_text(outline: Outline, text: str) -> None:
    """按 offset 把原文切进每个 Segment。闭区间，越界自动截断。"""
    for s in outline.segments:
        start = max(0, min(s.start, len(text)))
        end = max(start, min(s.end + 1, len(text)))  # 闭区间 → 切片要 +1
        s.text = text[start:end]
