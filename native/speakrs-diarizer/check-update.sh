#!/usr/bin/env bash
# speakrs の最新版を crates.io で確認する。
# 新しい版が出ていたら Cargo.toml の version を上げて cargo build --release で更新。
set -euo pipefail
cd "$(dirname "$0")"

echo "■ Cargo.toml の指定:"
grep -E '^speakrs' Cargo.toml | sed 's/^/  /'

echo "■ ビルド済み(ロック)バージョン:"
if [ -f Cargo.lock ]; then
  awk '/name = "speakrs"$/{getline; print "  " $0}' Cargo.lock
else
  echo "  (Cargo.lock なし = 未ビルド)"
fi

echo "■ crates.io 最新版:"
if command -v cargo >/dev/null; then
  cargo search speakrs --limit 1 | sed 's/^/  /'
else
  # cargo が無い場合は crates.io API にフォールバック
  curl -s https://crates.io/api/v1/crates/speakrs \
    | grep -o '"max_stable_version":"[^"]*"' | sed 's/^/  /'
fi

echo
echo "更新する場合: Cargo.toml の version を上げて再ビルド →"
echo "  PKG_CONFIG_PATH=/opt/homebrew/opt/openblas/lib/pkgconfig \\"
echo "  LIBRARY_PATH=/opt/homebrew/opt/openblas/lib cargo build --release"
