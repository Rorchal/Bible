#!/usr/bin/env bash
# 在你自己的服务器上把这个流水线跑起来。
#
#   curl -fsSL https://raw.githubusercontent.com/Rorchal/Bible/claude/bible-audio-rag-system-44nysl/scripts/bootstrap.sh | bash
#
# 或者 clone 下来之后： bash scripts/bootstrap.sh

set -euo pipefail

BRANCH="claude/bible-audio-rag-system-44nysl"
REPO="https://github.com/Rorchal/Bible"
DIR="${BIBLE_DIR:-$HOME/Bible}"

echo "==> 准备目录 $DIR"
if [ -d "$DIR/.git" ]; then
  git -C "$DIR" fetch origin "$BRANCH"
  git -C "$DIR" checkout "$BRANCH"
  git -C "$DIR" pull origin "$BRANCH"
else
  git clone -b "$BRANCH" "$REPO" "$DIR"
fi

cd "$DIR"

echo "==> 建虚拟环境"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

cat <<'TIP'

装好了。接下来：

  cd ~/Bible
  source .venv/bin/activate

  export ANTHROPIC_API_KEY=sk-ant-...        # 你的 key

  # 把转录文本放进 data/01_raw/（.txt / .srt / .vtt）
  # 然后先空跑看看，不花钱：
  python -m pipeline.refine --dry-run

  # 确认无误再正式跑：
  python -m pipeline.refine

产出在 data/02_refined/：.json 给下一步用，.md 给你人工校对。

TIP
