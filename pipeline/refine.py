"""第二步：把 ASR 转录稿清洗成结构化、分段的讲道文稿。

    python -m pipeline.refine                    # 处理 data/01_raw 下所有新文件
    python -m pipeline.refine --force            # 重新处理（覆盖已有结果）
    python -m pipeline.refine --dry-run          # 只看会处理什么，不调 API
    python -m pipeline.refine --only 罗马书       # 只处理文件名含该关键字的

输出 data/02_refined/<doc_id>.json（给下一步用）
   和 data/02_refined/<doc_id>.md（给你人工校对）
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import anthropic

from pipeline.common import (
    REFINED_DIR,
    Transcript,
    iter_transcripts,
    window_text,
    with_time_markers,
    write_json,
)
from pipeline.hotwords import Entry, apply_hotwords, glossary_for_prompt, load_hotwords
from pipeline.prompts import (
    OVERVIEW_SCHEMA,
    OVERVIEW_SYSTEM,
    REFINE_SCHEMA,
    REFINE_SYSTEM,
    SEGMENTS_ONLY_SCHEMA,
    build_refine_user_prompt,
)

MODEL = "claude-opus-5"

# 单次调用喂进去的字数上限。中文约 1 字 ≈ 1 token，输出要与输入等长，
# 留足输出空间。一小时讲道约 1 万字，通常一次就能吃下。
WINDOW_CHARS = 12000
WINDOW_OVERLAP = 0  # 整理任务不要重叠，否则内容会重复出现两次
MAX_TOKENS = 64000


def _call(
    client: anthropic.Anthropic,
    *,
    system: str,
    user: str,
    schema: dict,
    effort: str,
) -> dict:
    """一次结构化调用。system 做了缓存，多文件连跑时省钱。"""
    with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ],
        messages=[{"role": "user", "content": user}],
        thinking={"type": "adaptive"},
        output_config={
            "effort": effort,
            "format": {"type": "json_schema", "schema": schema},
        },
    ) as stream:
        message = stream.get_final_message()

    if message.stop_reason == "refusal":
        detail = getattr(message.stop_details, "explanation", None) or ""
        raise RuntimeError(f"模型拒绝处理这段内容。{detail}")
    if message.stop_reason == "max_tokens":
        raise RuntimeError(
            f"输出被 max_tokens={MAX_TOKENS} 截断了。"
            f"把 WINDOW_CHARS 调小再试（当前 {WINDOW_CHARS}）。"
        )

    text = next((b.text for b in message.content if b.type == "text"), None)
    if text is None:
        raise RuntimeError("响应里没有文本块，无法解析 JSON。")
    return json.loads(text)


def refine_transcript(
    client: anthropic.Anthropic,
    transcript: Transcript,
    effort: str,
    hotwords: list[Entry],
) -> dict:
    source = with_time_markers(transcript)

    # 送模型之前先把已知错字改掉，模型读到的就是通顺文本
    source, pre_hits = apply_hotwords(source, hotwords)
    glossary = glossary_for_prompt(hotwords)

    windows = window_text(source, WINDOW_CHARS, WINDOW_OVERLAP)

    if len(windows) == 1:
        print(f"    一次处理（{transcript.char_count} 字）…", flush=True)
        result = _call(
            client,
            system=REFINE_SYSTEM,
            user=build_refine_user_prompt(
                windows[0], transcript.doc_id, glossary=glossary
            ),
            schema=REFINE_SCHEMA,
            effort=effort,
        )
    else:
        print(
            f"    分 {len(windows)} 批处理（{transcript.char_count} 字）…", flush=True
        )
        segments: list[dict] = []
        for i, window in enumerate(windows):
            print(f"      第 {i + 1}/{len(windows)} 批（{len(window)} 字）", flush=True)
            part = _call(
                client,
                system=REFINE_SYSTEM,
                user=build_refine_user_prompt(
                    window, transcript.doc_id, i, len(windows), glossary
                ),
                schema=SEGMENTS_ONLY_SCHEMA,
                effort=effort,
            )
            segments.extend(part["segments"])

        outline = "\n".join(
            f"{i + 1}. {s['heading']}：{s['summary']}" for i, s in enumerate(segments)
        )
        print("      归纳总标题…", flush=True)
        overview = _call(
            client,
            system=OVERVIEW_SYSTEM,
            user=f"讲道文件名：{transcript.doc_id}\n\n各段提纲：\n{outline}",
            schema=OVERVIEW_SCHEMA,
            effort="low",
        )
        result = {**overview, "segments": segments}

    post_hits: Counter[str] = Counter()
    for i, segment in enumerate(result["segments"]):
        segment["segment_id"] = f"{transcript.doc_id}::{i:02d}"
        # 模型偶尔会把正确词又写回错的，输出后再兜一遍
        for field in ("heading", "summary", "text"):
            segment[field], hits = apply_hotwords(segment[field], hotwords)
            post_hits += hits
    for field in ("title", "theme"):
        result[field], hits = apply_hotwords(result[field], hotwords)
        post_hits += hits

    refined_chars = sum(len(s["text"]) for s in result["segments"])
    result["source"] = {
        "doc_id": transcript.doc_id,
        "path": _display_path(transcript.source_path),
        "has_timestamps": transcript.has_timestamps,
        "raw_chars": transcript.char_count,
        "refined_chars": refined_chars,
        "retention": round(refined_chars / max(transcript.char_count, 1), 3),
    }
    result["hotword_hits"] = {
        "pre": dict(pre_hits),   # 原文里改掉的
        "post": dict(post_hits),  # 模型输出里又兜回来的
    }
    return result


def _display_path(path: Path) -> str:
    """能相对项目根就相对，不能就给绝对路径 —— 不因此让整个任务失败。"""
    root = REFINED_DIR.parent.parent
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def to_markdown(result: dict) -> str:
    src = result["source"]
    lines = [
        f"# {result['title']}",
        "",
        f"- **主旨**：{result['theme']}",
        f"- **主要经文**：{result['main_scripture'] or '（未标注）'}",
        f"- **来源**：`{src['path']}`",
        f"- **字数**：原文 {src['raw_chars']} → 整理后 {src['refined_chars']}"
        f"（保留率 {src['retention']:.0%}）",
        "",
        "---",
        "",
    ]
    for segment in result["segments"]:
        stamp = f" `[{segment['start_time']}]`" if segment.get("start_time") else ""
        lines.append(f"## {segment['heading']}{stamp}")
        lines.append("")
        if segment["scripture_refs"]:
            lines.append(f"> 经文：{'、'.join(segment['scripture_refs'])}")
            lines.append("")
        lines.append(segment["text"])
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="第二步：清洗 + 分段提炼")
    parser.add_argument("--force", action="store_true", help="重新处理已有结果的文件")
    parser.add_argument("--dry-run", action="store_true", help="只列出待处理文件")
    parser.add_argument("--only", default="", help="只处理文件名含该关键字的")
    parser.add_argument(
        "--effort",
        default="medium",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="模型投入程度，默认 medium",
    )
    args = parser.parse_args(argv)

    try:
        hotwords = load_hotwords()
    except ValueError as exc:
        print(f"热词表有问题：{exc}", file=sys.stderr)
        return 1
    if hotwords:
        print(f"热词表：{len(hotwords)} 条 —— {'、'.join(c for c, _ in hotwords)}")

    transcripts = iter_transcripts()
    if args.only:
        transcripts = [t for t in transcripts if args.only in t.doc_id]

    if not transcripts:
        print("data/01_raw 下没有找到转录文件。把 .txt/.srt/.vtt 放进去再运行。")
        return 1

    pending = [
        t
        for t in transcripts
        if args.force or not (REFINED_DIR / f"{t.doc_id}.json").exists()
    ]
    skipped = len(transcripts) - len(pending)
    if skipped:
        print(f"跳过 {skipped} 个已处理的文件（要重做加 --force）")

    if args.dry_run:
        for t in pending:
            stamps = "有时间轴" if t.has_timestamps else "无时间轴"
            marked = with_time_markers(t)
            _, hits = apply_hotwords(marked, hotwords)
            batches = len(window_text(marked, WINDOW_CHARS))
            line = f"  {t.doc_id}  {t.char_count} 字  {stamps}  分 {batches} 批"
            if hits:
                line += "  热词命中：" + "、".join(
                    f"{w}×{n}" for w, n in hits.most_common()
                )
            print(line)
        print(f"\n共 {len(pending)} 个文件待处理。去掉 --dry-run 开始。")
        return 0

    if not pending:
        print("没有待处理的文件。")
        return 0

    client = anthropic.Anthropic()
    failures = 0

    for t in pending:
        print(f"\n[{t.doc_id}]")
        try:
            result = refine_transcript(client, t, args.effort, hotwords)
        except anthropic.RateLimitError as exc:
            print(f"    限流：{exc}", file=sys.stderr)
            failures += 1
            continue
        except anthropic.APIStatusError as exc:
            print(f"    API 报错 {exc.status_code}：{exc}", file=sys.stderr)
            failures += 1
            continue
        except anthropic.APIConnectionError as exc:
            print(f"    连接失败：{exc}", file=sys.stderr)
            failures += 1
            continue
        except (RuntimeError, json.JSONDecodeError) as exc:
            print(f"    处理失败：{exc}", file=sys.stderr)
            failures += 1
            continue

        write_json(REFINED_DIR / f"{t.doc_id}.json", result)
        (REFINED_DIR / f"{t.doc_id}.md").write_text(
            to_markdown(result), encoding="utf-8"
        )

        src = result["source"]
        print(
            f"    ✓ {len(result['segments'])} 段，"
            f"保留率 {src['retention']:.0%} → data/02_refined/{t.doc_id}.md"
        )
        hw = result["hotword_hits"]
        if hw["pre"] or hw["post"]:
            parts = [f"{w}×{n}" for w, n in sorted(hw["pre"].items())]
            note = "    热词修正：" + "、".join(parts) if parts else "    热词修正："
            if hw["post"]:
                note += "（模型输出里又兜回 " + "、".join(
                    f"{w}×{n}" for w, n in sorted(hw["post"].items())
                ) + "）"
            print(note)

        if src["retention"] < 0.7:
            print(
                "    ⚠ 保留率偏低，可能被摘要了。打开 .md 核对一下，"
                "必要时用 --force --effort high 重跑。"
            )

    print(f"\n完成 {len(pending) - failures}/{len(pending)} 个。")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
