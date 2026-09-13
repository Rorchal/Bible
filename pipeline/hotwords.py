"""热词表：修正 ASR 的固定错字。

两种规则：

  正确词 = 错写      确定性替换。用于错写本身不是正常词的情况（尽钱、零界）。
  正确词 ? 易混词    交给模型按上下文判断。用于易混词本身是常用词的情况
                     （一人、二人）——盲替换会毁掉「因一人的悖逆」这类经文。

两种都可以用 ! 跟一串例外：
  确定性规则的例外会被占位符护住，不参与替换；
  判断型规则的例外作为反例写进提示词，告诉模型这些说法本身是对的。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.common import ROOT

HOTWORDS_PATH = ROOT / "data" / "hotwords.txt"

REPLACE = "replace"  # = 确定性替换
JUDGE = "judge"  # ? 交给模型判断


@dataclass
class Entry:
    correct: str
    wrongs: list[str]
    mode: str = REPLACE
    exceptions: list[str] = field(default_factory=list)


def load_hotwords(path: Path = HOTWORDS_PATH) -> list[Entry]:
    if not path.exists():
        return []

    entries: list[Entry] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # 取最先出现的那个分隔符
        candidates = [(line.find(c), c) for c in ("=", "?") if c in line]
        if not candidates:
            raise ValueError(f"{path.name} 第 {lineno} 行缺少 '=' 或 '?'：{line!r}")
        sep_at, sep = min(candidates)

        correct = line[:sep_at].strip()
        wrongs_raw, _, exceptions_raw = line[sep_at + 1 :].partition("!")
        wrongs = [w.strip() for w in wrongs_raw.split(",") if w.strip()]
        exceptions = [e.strip() for e in exceptions_raw.split(",") if e.strip()]

        if not correct or not wrongs:
            raise ValueError(f"{path.name} 第 {lineno} 行格式不对：{line!r}")
        if correct in wrongs:
            raise ValueError(
                f"{path.name} 第 {lineno} 行：正确词「{correct}」出现在对照列表里"
            )
        for exc in exceptions:
            if not any(w in exc for w in wrongs):
                raise ValueError(
                    f"{path.name} 第 {lineno} 行：例外词「{exc}」不含任何对照词"
                    f"（{'、'.join(wrongs)}），这条例外不起作用，多半是写错了"
                )

        entries.append(
            Entry(correct, wrongs, REPLACE if sep == "=" else JUDGE, exceptions)
        )

    return entries


def _ordered_pairs(entries: list[Entry]) -> list[tuple[str, str]]:
    """(错写, 正确词)，长的错写排前面。只取确定性替换的规则。"""
    pairs = [
        (w, e.correct) for e in entries if e.mode == REPLACE for w in e.wrongs
    ]
    return sorted(pairs, key=lambda p: len(p[0]), reverse=True)


def _replace_exceptions(entries: list[Entry]) -> list[str]:
    """确定性规则的例外词，长的在前——「演艺人员」要比「艺人」先护住。"""
    seen = {exc for e in entries if e.mode == REPLACE for exc in e.exceptions}
    return sorted(seen, key=len, reverse=True)


# 占位符用控制字符，正常文本里不可能出现
_SENTINEL = "\x00"


def apply_hotwords(text: str, entries: list[Entry]) -> tuple[str, Counter[str]]:
    """只执行确定性替换。判断型规则不碰文本，它们只进提示词。"""
    guarded: list[str] = []
    for exc in _replace_exceptions(entries):
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
    """给模型看的术语表 + 易混词判断指引。

    确定性规则只给正确写法，不给错写——免得反而提示模型往错的写。
    判断型规则必须把两边都给出来，模型才知道要在哪两个词之间抉择。
    """
    if not entries:
        return ""

    blocks: list[str] = []

    terms = list(dict.fromkeys(e.correct for e in entries))
    blocks.append(
        "本系列讲道的固定术语，务必用以下写法，不要改成别的同音词：\n"
        + "、".join(terms)
    )

    judged = [e for e in entries if e.mode == JUDGE]
    if judged:
        lines = [
            "以下几组词发音相近，语音识别常认错。"
            "请**逐处根据上下文判断**该用哪个，不要机械替换："
        ]
        for e in judged:
            forms = "」「".join(e.wrongs)
            lines.append(f"- 「{forms}」有时是「{e.correct}」的误识")
            if e.exceptions:
                lines.append(
                    f"  但这些说法本身就是对的，不要改：{'、'.join(e.exceptions)}"
                )
        blocks.append("\n".join(lines))

    return "\n\n" + "\n\n".join(blocks)
