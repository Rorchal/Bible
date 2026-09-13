"""热词表：修正 ASR 的固定错字。

替换是确定性的子串替换，在送模型**之前**先做一遍（让模型读到通顺的文本），
模型输出**之后**再兜一遍（模型偶尔会把正确词又写回错的）。

有些错写本身是正常词（例：「艺人」是正常词，且「手艺人」里就含「艺人」），
这种要用例外列表把不该动的词护住。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.common import ROOT

HOTWORDS_PATH = ROOT / "data" / "hotwords.txt"


@dataclass
class Entry:
    correct: str
    wrongs: list[str]
    # 含有错写、但不该被替换的词。例：义人 = 艺人 ! 手艺人
    exceptions: list[str] = field(default_factory=list)


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

        correct, _, rest = line.partition("=")
        correct = correct.strip()

        # 「!」之后是例外词
        wrongs_raw, _, exceptions_raw = rest.partition("!")
        wrongs = [w.strip() for w in wrongs_raw.split(",") if w.strip()]
        exceptions = [e.strip() for e in exceptions_raw.split(",") if e.strip()]

        if not correct or not wrongs:
            raise ValueError(f"{path.name} 第 {lineno} 行格式不对：{line!r}")
        if correct in wrongs:
            raise ValueError(
                f"{path.name} 第 {lineno} 行：正确词「{correct}」出现在错写列表里，"
                f"会无限自我替换"
            )
        for exc in exceptions:
            if not any(w in exc for w in wrongs):
                raise ValueError(
                    f"{path.name} 第 {lineno} 行：例外词「{exc}」不含任何错写"
                    f"（{'、'.join(wrongs)}），这条例外不起作用，多半是写错了"
                )

        entries.append(Entry(correct, wrongs, exceptions))

    return entries


def _ordered_pairs(entries: list[Entry]) -> list[tuple[str, str]]:
    """(错写, 正确词) 列表，长的错写排前面，避免短的先匹配掉一部分。"""
    pairs = [(w, e.correct) for e in entries for w in e.wrongs]
    return sorted(pairs, key=lambda p: len(p[0]), reverse=True)


def _all_exceptions(entries: list[Entry]) -> list[str]:
    """所有例外词，长的在前——「演艺人员」要比「艺人」先护住。"""
    seen = {exc for e in entries for exc in e.exceptions}
    return sorted(seen, key=len, reverse=True)


# 占位符用控制字符，正常文本里不可能出现
_SENTINEL = "\x00"


def apply_hotwords(text: str, entries: list[Entry]) -> tuple[str, Counter[str]]:
    """返回替换后的文本，以及每个正确词命中了多少次。

    例外词先换成占位符护住，替换完再还原。
    """
    exceptions = _all_exceptions(entries)
    guarded: list[str] = []
    for exc in exceptions:
        if exc in text:
            token = f"{_SENTINEL}{len(guarded)}{_SENTINEL}"
            text = text.replace(exc, token)
            guarded.append(exc)

    hits: Counter[str] = Counter()
    for wrong, correct in _ordered_pairs(entries):
        count = text.count(wrong)
        if count:
            text = text.replace(wrong, correct)
            hits[correct] += count

    for i, exc in enumerate(guarded):
        text = text.replace(f"{_SENTINEL}{i}{_SENTINEL}", exc)

    return text, hits


def glossary_for_prompt(entries: list[Entry]) -> str:
    """给模型看的术语表。只给正确写法，不给错写——免得反而提示它写错。"""
    if not entries:
        return ""
    terms = "、".join(e.correct for e in entries)
    return f"\n\n本系列讲道的固定术语，务必用以下写法，不要改成别的同音词：\n{terms}"
