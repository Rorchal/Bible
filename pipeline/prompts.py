"""第二步的提示词。

核心原则：这一步是**清洗 + 结构化**，不是摘要。
讲道的举例、比喻、应用正是 RAG 检索最有价值的部分，一摘要就全丢了。
段落的 summary 只进 metadata，text 必须保留全部内容。
"""

REFINE_SYSTEM = """\
你是中文讲道录音转录稿的整理编辑。输入是语音识别（ASR）直出的文本，\
可能没有标点、有错别字、有口语赘词。你要把它整理成可供检索的结构化文稿。

# 最重要的一条规则

**这是整理，不是摘要。** 输出的 text 字段必须覆盖输入的全部内容，\
长度应与输入相当（去掉赘词后通常是输入的 85%–100%）。讲员举的例子、\
打的比方、讲的故事、重复强调的话、生活应用——全部保留。\
绝不要因为"不重要"而删掉任何一段实质内容，也绝不要添加原文没有的解释、\
评论或神学观点。你在誊清，不在改写。

# 具体要做的整理

1. **加标点、分自然段。** 按语义断句，用规范的中文标点。
2. **去口语赘词。** 删掉"嗯""啊""那个""就是说""对不对""是不是"这类\
无实义的填充词，以及语音重复（"我们我们今天"→"我们今天"）。
3. **修口误自纠。** 讲员说错又立刻改正的，只保留改正后的说法。\
   （例："在马太福音——不对，是马可福音第五章" → "在马可福音第五章"）
4. **改 ASR 错别字**，尤其是圣经专名。用和合本的通行译名：\
   亚伯拉罕、以撒、雅各、摩西、大卫、所罗门、以赛亚、耶利米、以西结、\
   彼得、保罗、约翰、提摩太、腓利门、以弗所、哥林多、加拉太、腓立比、\
   歌罗西、帖撒罗尼迦、希伯来、雅各书、启示录……\
   拿不准的专名保持原样，不要猜着改。
5. **标出经文引用。** 正文里的引用规范成「书卷 章:节」，\
   例如"创世记 22:1-19""罗马书 8:28"。同时汇总到 scripture_refs 字段。

# 分段

按**讲道的论述结构**切分，不要按长度机械切。一段 = 讲员讲完一个完整的点。\
典型的一篇讲道有 4–12 段：开场引入、几个分论点、应用、结语。\
每段给一个具体的小标题（不要用"第一部分""引言"这种空标题，\
要写"亚伯拉罕在摩利亚山上的顺服"这种有信息量的）。

# 时间戳

如果输入里有 [hh:mm:ss] 标记，把每段开头最近的那个标记填进该段的 start_time。\
标记本身不要出现在 text 里。输入没有标记就填 null。
"""

REFINE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "这篇讲道的标题，从内容归纳，15 字以内",
        },
        "theme": {
            "type": "string",
            "description": "全篇主旨，1-2 句话",
        },
        "main_scripture": {
            "type": "string",
            "description": "全篇的主要经文出处，如「创世记 22:1-19」；没有明确主经文则填空字符串",
        },
        "segments": {
            "type": "array",
            "description": "按论述结构切分的段落，覆盖输入全文",
            "items": {
                "type": "object",
                "properties": {
                    "heading": {
                        "type": "string",
                        "description": "有信息量的小标题，20 字以内",
                    },
                    "summary": {
                        "type": "string",
                        "description": "这一段讲了什么，1-2 句。只进 metadata，不替代 text",
                    },
                    "scripture_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "本段涉及的经文，规范为「书卷 章:节」",
                    },
                    "start_time": {
                        "type": ["string", "null"],
                        "description": "本段开头的时间戳 hh:mm:ss；输入无时间标记则为 null",
                    },
                    "text": {
                        "type": "string",
                        "description": "整理后的完整正文。必须覆盖本段全部内容，不得摘要",
                    },
                },
                "required": [
                    "heading",
                    "summary",
                    "scripture_refs",
                    "start_time",
                    "text",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "theme", "main_scripture", "segments"],
    "additionalProperties": False,
}

# 长文分窗时用的 schema：只出 segments，标题主旨最后统一生成
SEGMENTS_ONLY_SCHEMA = {
    "type": "object",
    "properties": {"segments": REFINE_SCHEMA["properties"]["segments"]},
    "required": ["segments"],
    "additionalProperties": False,
}

OVERVIEW_SYSTEM = """\
你在为一篇中文讲道稿拟总标题和主旨。输入是这篇讲道各段的小标题与摘要。\
标题要具体、有信息量，不要用"神的爱""信心的功课"这种放之四海皆准的空话。\
"""

OVERVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "theme": {"type": "string"},
        "main_scripture": {"type": "string"},
    },
    "required": ["title", "theme", "main_scripture"],
    "additionalProperties": False,
}


def build_refine_user_prompt(
    text: str, doc_id: str, window_index: int = 0, window_total: int = 1
) -> str:
    header = f"讲道文件名：{doc_id}"
    if window_total > 1:
        header += (
            f"\n这是全文的第 {window_index + 1} / {window_total} 段"
            f"（长文分批处理）。只整理下面给你的这一批内容，"
            f"不要补写前后文，也不要因为它在中间就省略开头或结尾。"
        )
    return f"{header}\n\n以下是需要整理的转录原文：\n\n<转录原文>\n{text}\n</转录原文>"
