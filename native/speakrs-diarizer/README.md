# speakrs-diarizer

[speakrs](https://github.com/avencera/speakrs)（Rust / Apple Silicon CoreML）を使った高速話者識別の
薄い CLI ラッパー。pyannote community-1 同等精度（VoxConverse 7.1% DER）を、CPU の pyannote より
**桁違いに高速**に実行する。実測: 4分音声を **2.1秒**（≈113x realtime）、51分会議で約30秒。

Python 側 `meeting_transcriber/diarization_speakrs.py` が本バイナリを subprocess で呼び出す。

## 入出力

```
speakrs-diarizer <16khz-mono.wav> [coreml|coreml-fast|cpu]   # 既定: coreml
# stdout に RTTM: SPEAKER audio 1 <start> <dur> <NA> <NA> SPEAKER_xx <NA> <NA>
```

## ビルド

OpenBLAS が必要（speakrs の LAPACK バックエンド。`openblas-static` は gfortran 必須なので
prebuilt の `openblas-system` を使う）。

```bash
brew install openblas      # 初回のみ
cd native/speakrs-diarizer
PKG_CONFIG_PATH=/opt/homebrew/opt/openblas/lib/pkgconfig \
LIBRARY_PATH=/opt/homebrew/opt/openblas/lib \
cargo build --release
# => target/release/speakrs-diarizer
```

初回実行時にモデル（約300MB）が `~/.cache/huggingface/hub/models--avencera--speakrs-models`
へ自動DLされる（以降はオフライン可）。

## 最新版の確認・更新

```bash
./check-update.sh          # crates.io の最新 speakrs バージョンを表示
```

新しい版が出ていたら `Cargo.toml` の `speakrs = { version = "x.y", ... }` を上げて再ビルド。

## 環境変数（Python ラッパー側）

- `MEETING_SPEAKRS_BIN` — バイナリのパスを上書き（既定: 本ディレクトリの target/release）
- `MEETING_SPEAKRS_MODE` — `coreml`（既定）/ `coreml-fast`（2秒窓・高速だが境界粗い）/ `cpu`
