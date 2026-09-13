"""按目录文件的 offset，把原始转录切成带段落信息的文本。

    python -m pipeline.apply_outline data/01_raw/某讲道.txt

目录默认按同名 .md 在 data/00_outlines/ 里找，也可以用 --outline 指定。

正文**逐字来自原文**，只做热词的确定性替换，不经模型改写、不做摘要。
产出：
    data/02_refined/<名字>.md    带完整层级的可读文稿，供人工校对
    data/02_refined/<名字>.json  与第二步同构，第三步可直接消费
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from pipeline.common import REFINED_DIR, load_transcript, write_json
from pipeline.outline import LINE
from pipeline.hotwords import apply_hotwords, load_hotwords
from pipeline.outline import OUTLINE_DIR, parse_outline, slice_text, validate

# 常见中文书卷名 + 章:节，用于从标题和简述里挑出经文引用
_BOOKS = (
    "创世记|出埃及记|利未记|民数记|申命记|约书亚记|士师记|路得记|撒母耳记[上下]?|"
    "列王纪[上下]?|历代志[上下]?|以斯拉记|尼希米记|以斯帖记|约伯记|诗篇|箴言|传道书|"
    "雅歌|以赛亚书|耶利米书|耶利米哀歌|以西结书|但以理书|何西阿书|约珥书|阿摩司书|"
    "俄巴底亚书|约拿书|弥迦书|那鸿书|哈巴谷书|西番雅书|哈该书|撒迦利亚书|玛拉基书|"
    "马太福音|马可福音|路加福音|约翰福音|使徒行传|罗马书|哥林多[前后]书|加拉太书|"
    "以弗所书|腓立比书|歌罗西书|帖撒罗尼迦[前后]书|提摩太[前后]书|提多书|腓利门书|"
    "希伯来书|雅各书|彼得[前后]书|约翰[一二三]书|犹大书|启示录|伯|诗|箴|太|可|路|约|徒|罗"
)
_SCRIPTURE = re.compile(rf"({_BOOKS})\s*(\d+)[:：](\d+(?:\s*[-–]\s*\d+)?)")


def find_scriptures(*texts: str) -> list[str]:
    found: list[str] = []
    for text in texts:
        for book, chapter, verses in _SCRIPTURE.findall(text):
            ref = f"{book} {chapter}:{verses.replace(' ', '')}"
            if ref not in found:
                found.append(ref)
    return found


def default_outline_path(source: Path) -> Path:
    return OUTLINE_DIR / f"{source.stem}.md"


def to_markdown(outline, doc_id: str, source: Path, raw_len: int) -> str:
    total = sum(len(s.text) for s in outline.segments)
    lines = [
        f"# {outline.title}",
        "",
        f"- **来源**：`{source}`（{raw_len} 字）",
        f"- **目录**：{len(outline.segments)} 段（按{outline.unit_name}定位），切片共 {total} 字",
        "",
        "---",
        "",
    ]
    chapter = section = None
    for s in outline.segments:
        if s.chapter != chapter:
            chapter = s.chapter
            lines += [f"## {chapter}", ""]
        if s.section != section:
            section = s.section
            lines += [f"### {section}", ""]
        lines += [
            f"**[{s.seg_no}]** `{outline.unit_name} {s.start}–{s.end}`（{len(s.text)} 字）",
            "",
            f"> {s.summary}",
            "",
            s.text,
            "",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="按目录 offset 把原始转录切成带段落信息的文本"
    )
    parser.add_argument("source", type=Path, help="原始转录 .txt/.srt/.vtt")
    parser.add_argument("--outline", type=Path, default=None, help="目录 .md")
    parser.add_argument(
        "--no-hotwords", action="store_true", help="不做热词替换，纯粹按 offset 切"
    )
    parser.add_argument(
        "--strict", action="store_true", help="有任何警告也退出，不只是错误"
    )
    args = parser.parse_args(argv)

    if not args.source.exists():
        print(f"找不到原文：{args.source}", file=sys.stderr)
        return 1

    outline_path = args.outline or default_outline_path(args.source)
    if not outline_path.exists():
        print(f"找不到目录：{outline_path}", file=sys.stderr)
        return 1

    outline = parse_outline(outline_path)

    if args.source.suffix.lower() in {".txt", ".md"}:
        # offset 指向原始文件，不能用规范化过的文本，否则会整体错位
        text = args.source.read_text(encoding="utf-8").lstrip("\ufeff")
        transcript = None
    else:
        transcript = load_transcript(args.source)
        text = transcript.text

    line_count = len(text.rstrip("\n").split("\n"))
    extent = line_count if outline.unit == LINE else len(text)

    print(f"目录：{outline_path.name}  {len(outline.segments)} 段  单位：{outline.unit_name}")
    print(f"原文：{args.source.name}  {len(text)} 字 / {line_count} 行")

    issues = validate(outline, extent)
    errors = [i for i in issues if i.level == "错误"]
    for issue in issues:
        print(f"  [{issue.level}] {issue.message}")
    if errors:
        print("\n有错误，未生成文件。先修目录里的 offset。", file=sys.stderr)
        return 1
    if issues and args.strict:
        print("\n--strict：有警告，未生成文件。", file=sys.stderr)
        return 1

    hotwords = [] if args.no_hotwords else load_hotwords()
    if hotwords:
        text, hits = apply_hotwords(text, hotwords)
        if hits:
            print("  热词修正：" + "、".join(f"{w}×{n}" for w, n in hits.most_common()))

    slice_text(outline, text)

    empty = [s.seg_no for s in outline.segments if not s.text.strip()]
    if empty:
        print(f"  [警告] 这些段切出来是空的：{'、'.join(empty)}")

    doc_id = args.source.stem
    REFINED_DIR.mkdir(parents=True, exist_ok=True)
    (REFINED_DIR / f"{doc_id}.md").write_text(
        to_markdown(outline, doc_id, args.source, len(text)), encoding="utf-8"
    )

    covered = sum(len(s.text) for s in outline.segments)
    covered_units = (
        sum(s.end - s.start + 1 for s in outline.segments)
        if outline.unit == LINE
        else covered
    )
    write_json(
        REFINED_DIR / f"{doc_id}.json",
        {
            "title": outline.title,
            "theme": "",
            "main_scripture": "",
            "segments": [
                {
                    "segment_id": f"{doc_id}::{s.seg_no}",
                    "heading": s.section or s.chapter,
                    "summary": s.summary,
                    "scripture_refs": find_scriptures(s.section, s.summary),
                    "start_time": None,
                    "text": s.text,
                    "chapter": s.chapter,
                    "offset": [s.start, s.end],
                }
                for s in outline.segments
            ],
            "source": {
                "doc_id": doc_id,
                "path": str(args.source),
                "outline": str(outline_path),
                "unit": outline.unit,
                "has_timestamps": bool(transcript and transcript.has_timestamps),
                "raw_chars": len(text),
                "refined_chars": covered,
                "retention": round(covered / max(len(text), 1), 3),
                "segmented_by": "outline",
            },
        },
    )

    print(
        f"\n✓ {len(outline.segments)} 段，覆盖 {covered_units}/{extent} {outline.unit_name}"
        f"（{covered_units / max(extent, 1):.1%}），正文共 {covered} 字"
        f"\n  → data/02_refined/{doc_id}.md"
        f"\n  → data/02_refined/{doc_id}.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
