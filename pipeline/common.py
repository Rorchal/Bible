"""输入解析与文本分窗 —— 第二步和第三步共用。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "01_raw"
REFINED_DIR = ROOT / "data" / "02_refined"
CHUNK_DIR = ROOT / "data" / "03_chunks"
INDEX_DIR = ROOT / "data" / "04_index"

SUPPORTED_SUFFIXES = {".txt", ".srt", ".vtt", ".md"}


@dataclass
class Cue:
    """字幕里的一条，start 是秒。"""

    start: float
    text: str


@dataclass
class Transcript:
    """一篇讲道的原始转录。"""

    doc_id: str
    source_path: Path
    text: str
    cues: list[Cue] = field(default_factory=list)

    @property
    def has_timestamps(self) -> bool:
        return bool(self.cues)

    @property
    def char_count(self) -> int:
        return len(self.text)


def hhmmss(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def parse_timecode(raw: str) -> float:
    """把 00:12:30,500 或 00:12:30.500 或 12:30.500 转成秒。"""
    raw = raw.strip().replace(",", ".")
    parts = raw.split(":")
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = "0", parts[0], parts[1]
    else:
        raise ValueError(f"看不懂的时间码: {raw!r}")
    return int(h) * 3600 + int(m) * 60 + float(s)


_CUE_TIME = re.compile(
    r"(\d{1,2}:\d{2}(?::\d{2})?[.,]\d{1,3}|\d{1,2}:\d{2}:\d{2})\s*-->\s*"
    r"(\d{1,2}:\d{2}(?::\d{2})?[.,]\d{1,3}|\d{1,2}:\d{2}:\d{2})"
)
# 字幕里常见的内联标签：<c>、<00:00:01.000>、{\an8} 之类
_INLINE_TAG = re.compile(r"<[^>]*>|\{\\[^}]*\}")


def parse_subtitles(raw: str) -> list[Cue]:
    """解析 SRT / VTT。两种格式的块结构一样，只是 VTT 有个文件头。"""
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", raw.replace("\r\n", "\n").strip()):
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        if lines[0].upper().startswith("WEBVTT"):
            continue
        match = None
        body_start = 0
        for i, line in enumerate(lines[:3]):
            match = _CUE_TIME.search(line)
            if match:
                body_start = i + 1
                break
        if not match:
            continue
        body = " ".join(_INLINE_TAG.sub("", ln).strip() for ln in lines[body_start:])
        body = re.sub(r"\s+", " ", body).strip()
        if body:
            cues.append(Cue(start=parse_timecode(match.group(1)), text=body))
    return dedupe_cues(cues)


def dedupe_cues(cues: list[Cue]) -> list[Cue]:
    """滚动字幕会让同一句在相邻 cue 里重复出现，去掉这种重复。"""
    out: list[Cue] = []
    for cue in cues:
        if out and (cue.text == out[-1].text or cue.text.startswith(out[-1].text)):
            # 后一条包含前一条 —— 是滚动，用长的那条替换
            out[-1] = Cue(start=out[-1].start, text=cue.text)
            continue
        out.append(cue)
    return out


def normalize_plain_text(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("　", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_transcript(path: Path) -> Transcript:
    raw = path.read_text(encoding="utf-8")
    doc_id = path.stem

    if path.suffix.lower() in {".srt", ".vtt"}:
        cues = parse_subtitles(raw)
        if cues:
            return Transcript(doc_id, path, " ".join(c.text for c in cues), cues)
        # 解析不出 cue 就退化成纯文本处理
    return Transcript(doc_id, path, normalize_plain_text(raw), [])


def iter_transcripts(directory: Path = RAW_DIR) -> list[Transcript]:
    paths = sorted(
        p
        for p in directory.iterdir()
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_SUFFIXES
        and p.name.lower() != "readme.md"
    )
    return [load_transcript(p) for p in paths]


def with_time_markers(transcript: Transcript, every_seconds: float = 60.0) -> str:
    """给带时间轴的转录每隔约一分钟插一个 [hh:mm:ss] 标记。

    模型据此把时间带进每一段，将来检索命中时能回放到录音的对应位置。
    """
    if not transcript.has_timestamps:
        return transcript.text

    pieces: list[str] = []
    last_marked = -every_seconds
    for cue in transcript.cues:
        if cue.start - last_marked >= every_seconds:
            pieces.append(f"[{hhmmss(cue.start)}]")
            last_marked = cue.start
        pieces.append(cue.text)
    return " ".join(pieces)


# 中文断句：在句末标点后切，标点跟着前一句走
_SENTENCE_END = re.compile(r"(?<=[。！？!?；;…])(?![」』”’\)）】》])")


def split_sentences(text: str) -> list[str]:
    """按句末标点切句。ASR 无标点的长文会退化成按换行/长度切。"""
    parts = [s for s in _SENTENCE_END.split(text) if s.strip()]
    if len(parts) > 1:
        return parts

    # 没有句末标点（典型的 ASR 直出）—— 退而按换行切，再按长度兜底
    parts = [ln for ln in re.split(r"\n+", text) if ln.strip()]
    out: list[str] = []
    for part in parts:
        while len(part) > 200:
            cut = part.rfind("，", 0, 200)
            cut = cut + 1 if cut > 80 else 200
            out.append(part[:cut])
            part = part[cut:]
        if part.strip():
            out.append(part)
    return out or [text]


def window_text(text: str, max_chars: int, overlap_chars: int = 0) -> list[str]:
    """按句子边界切成不超过 max_chars 的窗口，可带重叠。"""
    if len(text) <= max_chars:
        return [text]

    sentences: list[str] = []
    for sentence in split_sentences(text):
        # 无标点的 ASR 长句可能一句就超过上限，硬切兜底，否则窗口会超限
        while len(sentence) > max_chars:
            sentences.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if sentence:
            sentences.append(sentence)

    windows: list[str] = []
    current: list[str] = []
    size = 0

    for sentence in sentences:
        if size + len(sentence) > max_chars and current:
            windows.append("".join(current))
            if overlap_chars > 0:
                tail: list[str] = []
                tail_size = 0
                for prev in reversed(current):
                    if tail_size >= overlap_chars:
                        break
                    tail.insert(0, prev)
                    tail_size += len(prev)
                current, size = tail, tail_size
            else:
                current, size = [], 0
        current.append(sentence)
        size += len(sentence)

    if current:
        windows.append("".join(current))
    return windows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
