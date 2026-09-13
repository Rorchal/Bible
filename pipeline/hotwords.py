"""热词表：修正 ASR 的固定错字。

替换是确定性的子串替换，在送模型**之前**先做一遍（让模型读到通顺的文本），
模型输出**之后**再兜一遍（模型偶尔会把正确词又写回错的）。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from pipeline.common import ROOT

HOTWORDS_PATH = ROOT / "data" / "hotwords.txt"

# (正确词, [错写...])
Entry = tuple[str, list[str]]


def load_hotwords(path: Path = HOTWORDS_PATH) -> list[Entry]:
    if not path.exists():
        return []

    entries: list[Entry] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path.name} 第 {lineno} 行缺少 '='：{line!r}")

        correct, _, wrongs_raw = line.partition("=")
        correct = correct.strip()
        wrongs = [w.strip() for w in wrongs_raw.split(",") if w.strip()]
        if not correct or not wrongs:
            raise ValueError(f"{path.name} 第 {lineno} 行格式不对：{line!r}")
        if correct in wrongs:
            raise ValueError(
                f"{path.name} 第 {lineno} 行：正确词「{correct}」出现在错写列表里，会无限自我替换"
            )
        entries.append((correct, wrongs))

    return entries


def _ordered_pairs(entries: list[Entry]) -> list[tuple[str, str]]:
    """(错写, 正确词) 列表，长的错写排前面，避免短的先匹配掉一部分。"""
    pairs = [(wrong, correct) for correct, wrongs in entries for wrong in wrongs]
    return sorted(pairs, key=lambda p: len(p[0]), reverse=True)


def apply_hotwords(text: str, entries: list[Entry]) -> tuple[str, Counter[str]]:
    """返回替换后的文本，以及每个正确词命中了多少次。"""
    hits: Counter[str] = Counter()
    for wrong, correct in _ordered_pairs(entries):
        count = text.count(wrong)
        if count:
            text = text.replace(wrong, correct)
            hits[correct] += count
    return text, hits


def glossary_for_prompt(entries: list[Entry]) -> str:
    """给模型看的术语表。只给正确写法，不给错写——免得反而提示它写错。"""
    if not entries:
        return ""
    terms = "、".join(correct for correct, _ in entries)
    return (
        f"\n\n本系列讲道的固定术语，务必用以下写法，不要改成别的同音词：\n{terms}"
    )
