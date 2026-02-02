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

# 5. Claude Code MCP
echo ""
echo "4. Claude Code..."
if command -v claude &> /dev/null; then
    claude mcp remove meeting-transcriber -s user 2>/dev/null || true
    claude mcp add meeting-transcriber -s user \
        -e PYTHONPATH="$SCRIPT_DIR/src" \
        -- "$SCRIPT_DIR/.venv/bin/python" -m meeting_transcriber.server
    echo -e "   ${GREEN}✓${NC}"
else
    echo -e "   ${YELLOW}⚠ スキップ${NC}"
fi

# 6. Claude Desktop
echo ""
echo "5. Claude Desktop..."
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
    "env": {"PYTHONPATH": "$SCRIPT_DIR/src"}
}
p.write_text(json.dumps(c, indent=2, ensure_ascii=False))
EOF
    echo -e "   ${GREEN}✓${NC}"
else
    echo -e "   ${YELLOW}⚠ スキップ${NC}"
fi

# 7. スキル
echo ""
echo "6. Claude Codeスキル..."
mkdir -p "$HOME/.claude/commands"
cp -f "$SCRIPT_DIR/skills/transcribe-meeting.md" "$HOME/.claude/commands/"
echo -e "   ${GREEN}✓${NC}"

# 8. CLI
echo ""
echo "7. CLIコマンド..."
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
