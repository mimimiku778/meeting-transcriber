//! speakrs-diarizer: meeting-transcriber 用の薄い CLI ラッパー。
//!
//! 入力: 16kHz mono WAV のパス。出力: RTTM を stdout に出す。
//! Apple Silicon では CoreML バックエンドで pyannote community-1 同等精度・桁違いに高速。
//!
//! usage: speakrs-diarizer <16khz-mono.wav> [coreml|coreml-fast|cpu]

use std::env;
use std::process::exit;

use speakrs::{ExecutionMode, OwnedDiarizationPipeline};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: speakrs-diarizer <16khz-mono.wav> [coreml|coreml-fast|cpu]");
        exit(2);
    }
    let wav_path = &args[1];
    let mode = args.get(2).map(String::as_str).unwrap_or("coreml");
    let exec = match mode {
        "coreml" => ExecutionMode::CoreMl,
        "coreml-fast" => ExecutionMode::CoreMlFast,
        "cpu" => ExecutionMode::Cpu,
        other => {
            eprintln!("unknown mode: {other}");
            exit(2);
        }
    };

    // WAV を Vec<f32> へ。Python 側で 16kHz mono を保証して渡すが、
    // 念のためステレオはダウンミックス、整数 PCM は正規化する。
    let mut reader = match hound::WavReader::open(wav_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("wav open error: {e}");
            exit(1);
        }
    };
    let spec = reader.spec();
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(Result::ok).collect(),
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .filter_map(Result::ok)
                .map(|v| v as f32 / max)
                .collect()
        }
    };
    let mono: Vec<f32> = if spec.channels > 1 {
        let ch = spec.channels as usize;
        raw.chunks(ch).map(|c| c.iter().sum::<f32>() / ch as f32).collect()
    } else {
        raw
    };

    let mut pipeline = match OwnedDiarizationPipeline::from_pretrained(exec) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("pipeline load error: {e}");
            exit(1);
        }
    };
    let result = match pipeline.run(&mono) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("diarization error: {e}");
            exit(1);
        }
    };
    print!("{}", result.rttm("audio"));
}
