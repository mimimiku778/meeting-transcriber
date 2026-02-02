#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================"
echo "  Meeting Transcriber インストーラー"
echo "========================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. macOS確認
if [[ "$(uname)" != "Darwin" ]]; then
    echo -e "${RED}エラー: macOS専用${NC}"
    exit 1
fi

# 2. Homebrew確認
echo "1. Homebrew確認..."
if ! command -v brew &> /dev/null; then
    echo -e "${RED}エラー: Homebrewが必要${NC}"
    echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    exit 1
fi
echo -e "   ${GREEN}✓${NC}"

# 3. ffmpeg
echo "2. ffmpeg..."
if ! command -v ffmpeg &> /dev/null; then
    brew install ffmpeg
fi
echo -e "   ${GREEN}✓${NC}"

# 4. Python venv + パッケージ
echo "3. Python環境..."
[ ! -d ".venv" ] && python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -e .
echo -e "   ${GREEN}✓${NC}"

# 5. HF_TOKEN (.env)
echo ""
echo "4. HF_TOKEN..."
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
fi
if [ -z "$HF_TOKEN" ]; then
    read -p "   トークンを入力 (空欄でスキップ): " HF_TOKEN_INPUT
    [ -n "$HF_TOKEN_INPUT" ] && HF_TOKEN="$HF_TOKEN_INPUT"
fi
if [ -n "$HF_TOKEN" ]; then
    echo "HF_TOKEN=$HF_TOKEN" > "$SCRIPT_DIR/.env"
    echo -e "   ${GREEN}✓${NC} .envに保存"
else
    echo -e "   ${YELLOW}⚠ 未設定（後で .env に HF_TOKEN=xxx を記載）${NC}"
fi

# 6. Claude Code MCP
echo ""
echo "5. Claude Code..."
if command -v claude &> /dev/null; then
    claude mcp remove meeting-transcriber -s user 2>/dev/null || true
    ENV_ARGS=(-e PYTHONPATH="$SCRIPT_DIR/src")
    [ -n "$HF_TOKEN" ] && ENV_ARGS+=(-e HF_TOKEN="$HF_TOKEN")
    claude mcp add meeting-transcriber -s user "${ENV_ARGS[@]}" \
        -- "$SCRIPT_DIR/.venv/bin/python" -m meeting_transcriber.server
    echo -e "   ${GREEN}✓${NC}"
else
    echo -e "   ${YELLOW}⚠ スキップ${NC}"
fi

# 7. Claude Desktop
echo ""
echo "6. Claude Desktop..."
DESKTOP_CONFIG="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
if [ -f "$DESKTOP_CONFIG" ]; then
    python3 << EOF
import json
from pathlib import Path
p = Path("$DESKTOP_CONFIG")
c = json.loads(p.read_text())
c.setdefault("mcpServers", {})["meeting-transcriber"] = {
    "command": "$SCRIPT_DIR/.venv/bin/python",
    "args": ["-m", "meeting_transcriber.server"],
    "env": {"PYTHONPATH": "$SCRIPT_DIR/src", "HF_TOKEN": "${HF_TOKEN:-}"}
}
p.write_text(json.dumps(c, indent=2, ensure_ascii=False))
EOF
    echo -e "   ${GREEN}✓${NC}"
else
    echo -e "   ${YELLOW}⚠ スキップ${NC}"
fi

# 8. スキル
echo ""
echo "7. Claude Codeスキル..."
mkdir -p "$HOME/.claude/commands"
cp -f "$SCRIPT_DIR/skills/transcribe-meeting.md" "$HOME/.claude/commands/"
echo -e "   ${GREEN}✓${NC}"

# 9. CLI
echo ""
echo "8. CLIコマンド..."
mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/transcribe" << EOF
#!/bin/bash
PYTHONPATH="$SCRIPT_DIR/src" \\
PYTHONWARNINGS="ignore" \\
exec "$SCRIPT_DIR/.venv/bin/python" -m meeting_transcriber.cli "\$@"
EOF
chmod +x "$HOME/.local/bin/transcribe"

[[ ":$PATH:" != *":$HOME/.local/bin:"* ]] && echo -e "   ${YELLOW}⚠ PATHに追加: export PATH=\"\$HOME/.local/bin:\$PATH\"${NC}"
echo -e "   ${GREEN}✓${NC}"

# 完了
echo ""
echo "========================================"
echo -e "  ${GREEN}完了${NC}"
echo "========================================"
echo ""
echo "使い方:"
echo "  transcribe /path/to/video.mov"
echo "  /transcribe-meeting /path/to/video.mov  (Claude Code)"
echo ""
[ -z "$HF_TOKEN" ] && echo -e "${YELLOW}HF_TOKENを設定: export HF_TOKEN=\"hf_xxx\"${NC}" && echo ""
echo "権限が必要:"
echo "  https://huggingface.co/pyannote/speaker-diarization-3.1"
echo "  https://huggingface.co/pyannote/segmentation-3.0"
echo "  https://huggingface.co/pyannote/speaker-diarization-community-1"
