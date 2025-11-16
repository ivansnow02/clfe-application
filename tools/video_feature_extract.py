import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import whisper
from MSA_FET import FeatureExtractionTool
from transformers import AutoModel, AutoTokenizer, pipeline


_TEXT_ENCODER_CACHE: Dict[str, Tuple[Any, Any]] = {}
_TRANSLATOR_CACHE: Dict[str, Any] = {}
_WHISPER_CACHE: Dict[str, Any] = {}


def load_config(config_path: str):
    """Load YAML config for MMSA-FET and return as dict."""
    import yaml

    cfg_path = Path(config_path)
    with cfg_path.open() as f:
        config = yaml.safe_load(f)
    return config


def _get_language_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract language config from the unified config, with sensible defaults."""
    lang_cfg = config.get("language", {})

    whisper_cfg = lang_cfg.get("whisper", {})
    translation_cfg = lang_cfg.get("translation", {})
    encoder_cfg = lang_cfg.get("encoder", {})

    return {
        "whisper_model": whisper_cfg.get("model", "small"),
        "transcript_language": whisper_cfg.get("language", "zh"),
        "translator_model": translation_cfg.get("model", "Helsinki-NLP/opus-mt-en-zh"),
        "enable_translation": translation_cfg.get("enable", True),
        "bert_model": encoder_cfg.get("model", "bert-base-chinese"),
        "pooling": encoder_cfg.get("pooling", "cls"),
        "max_length": encoder_cfg.get("max_length", 512),
        "device": encoder_cfg.get("device", "auto"),
    }


def _contains_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _chunk_text(text: str, chunk_size: int = 400) -> List[str]:
    normalized = text.strip()
    if not normalized:
        return []
    return [
        normalized[i : i + chunk_size] for i in range(0, len(normalized), chunk_size)
    ]


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_whisper(model_name: str):
    if model_name not in _WHISPER_CACHE:
        print(f"[INFO] Loading Whisper model: {model_name}")
        _WHISPER_CACHE[model_name] = whisper.load_model(model_name)
    return _WHISPER_CACHE[model_name]


def _transcribe_with_whisper(
    video_path: Path,
    model_name: str,
    language: str,
    temperature: float = 0.0,
) -> Tuple[str, List[Dict[str, float]]]:
    model = _load_whisper(model_name)
    result = model.transcribe(
        str(video_path),
        language=language,
        task="transcribe",
        temperature=temperature,
        verbose=False,
    )
    text = (result.get("text") or "").strip()
    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "id": seg.get("id"),
            "start": seg.get("start"),
            "end": seg.get("end"),
            "text": (seg.get("text") or "").strip(),
        })
    return text, segments


def _get_translator(model_name: str, device: str):
    if model_name not in _TRANSLATOR_CACHE:
        print(f"[INFO] Loading translation model: {model_name}")
        device_id = 0 if device == "cuda" else -1
        _TRANSLATOR_CACHE[model_name] = pipeline(
            "translation",
            model=model_name,
            device=device_id,
        )
    return _TRANSLATOR_CACHE[model_name]


def _translate_to_chinese(
    text: str, model_name: str, device: str
) -> Tuple[str, Optional[str]]:
    chunks = _chunk_text(text)
    if not chunks:
        return "", None
    translator = _get_translator(model_name, device)
    translated_chunks: List[str] = []
    for chunk in chunks:
        output = translator(chunk, max_length=512)
        translated_chunks.append(output[0]["translation_text"].strip())
    translated = "\n".join(translated_chunks).strip()
    return translated, model_name


def _load_text_encoder(model_name: str, device: str) -> Tuple[Any, Any]:
    if model_name not in _TEXT_ENCODER_CACHE:
        print(f"[INFO] Loading BERT model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        if device == "cuda":
            model.to(device)
        _TEXT_ENCODER_CACHE[model_name] = (tokenizer, model)
    return _TEXT_ENCODER_CACHE[model_name]


def _compute_language_embedding(
    text: str,
    model_name: str,
    pooling: str,
    device: str,
    max_length: int,
) -> List[float]:
    tokenizer, encoder = _load_text_encoder(model_name, device)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = encoder(**inputs)
    hidden = outputs.last_hidden_state
    if pooling == "mean":
        mask = inputs["attention_mask"].unsqueeze(-1)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        pooled = summed / counts
    else:
        pooled = hidden[:, 0, :]
    return pooled.squeeze(0).detach().cpu().tolist()


def extract_single_video(
    video_path: str,
    out_path: str,
    config_path: str,
    transcript_path: Optional[str] = None,
    whisper_model: Optional[str] = None,
    transcript_language: Optional[str] = None,
    ensure_chinese: Optional[bool] = None,
    translator_model: Optional[str] = None,
    bert_model: Optional[str] = None,
    pooling: Optional[str] = None,
    max_length: Optional[int] = None,
    skip_language: bool = False,
):
    """Extract multimodal features with language from config or CLI overrides."""
    video_path_path = Path(video_path)
    out_path_path = Path(out_path)
    out_path_path.parent.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)

    # Load language config from file, then override with CLI args if provided
    lang_config = _get_language_config(config)
    if whisper_model is not None:
        lang_config["whisper_model"] = whisper_model
    if transcript_language is not None:
        lang_config["transcript_language"] = transcript_language
    if translator_model is not None:
        lang_config["translator_model"] = translator_model
    if bert_model is not None:
        lang_config["bert_model"] = bert_model
    if pooling is not None:
        lang_config["pooling"] = pooling
    if max_length is not None:
        lang_config["max_length"] = max_length
    if ensure_chinese is not None:
        lang_config["enable_translation"] = ensure_chinese

    fet = FeatureExtractionTool(config=config)

    print(f"[INFO] Extracting audio/visual features for: {video_path_path}")
    feature = fet.run_single(str(video_path_path))

    if not skip_language:
        device = lang_config.get("device", "auto")
        if device == "auto":
            device = _get_device()

        transcript, segments = _transcribe_with_whisper(
            video_path_path,
            lang_config["whisper_model"],
            lang_config["transcript_language"],
        )
        normalized_text = transcript
        translator_used = None

        if (
            lang_config["enable_translation"]
            and normalized_text
            and not _contains_chinese(normalized_text)
        ):
            normalized_text, translator_used = _translate_to_chinese(
                normalized_text, lang_config["translator_model"], device
            )

        if transcript_path:
            transcript_file = Path(transcript_path)
            transcript_file.parent.mkdir(parents=True, exist_ok=True)
            transcript_file.write_text(normalized_text or "", encoding="utf-8")
            print(f"[INFO] Saved transcript to: {transcript_file}")

        if normalized_text:
            embedding = _compute_language_embedding(
                normalized_text,
                model_name=lang_config["bert_model"],
                pooling=lang_config["pooling"],
                device=device,
                max_length=lang_config["max_length"],
            )
            feature["language"] = {
                "transcript": normalized_text,
                "raw_transcript": transcript,
                "segments": segments,
                "whisper": {
                    "model": lang_config["whisper_model"],
                    "language": lang_config["transcript_language"],
                },
                "encoder": {
                    "model": lang_config["bert_model"],
                    "pooling": lang_config["pooling"],
                    "max_length": lang_config["max_length"],
                    "embedding": embedding,
                },
            }
            if translator_used:
                feature["language"]["translator"] = translator_used
        else:
            print("[WARN] Transcript is empty. Skipping language embedding.")

    print("[INFO] Extraction done. Keys in feature dict:", feature.keys())

    with out_path_path.open("wb") as f:
        pickle.dump(feature, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[INFO] Saved features to: {out_path_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract multimodal features (audio+video+language) for a single video",
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file (e.g. input.mp4)",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Path to output pickle file (e.g. output/feature.pkl)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to unified YAML config file (includes audio/video/language settings)",
    )
    parser.add_argument(
        "--transcript",
        type=str,
        default=None,
        help="Optional path to save the final Chinese transcript (.txt)",
    )
    parser.add_argument(
        "--skip-language",
        action="store_true",
        help="Disable Whisper/BERT language extraction",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default=None,
        help="Override Whisper model from config (tiny/base/small/medium/large-v2)",
    )
    parser.add_argument(
        "--transcript-language",
        type=str,
        default=None,
        help="Override language code from config",
    )
    parser.add_argument(
        "--bert-model",
        type=str,
        default=None,
        help="Override BERT model from config",
    )
    parser.add_argument(
        "--language-pooling",
        type=str,
        choices=["cls", "mean"],
        default=None,
        help="Override pooling strategy from config",
    )
    parser.add_argument(
        "--max-transcript-length",
        type=int,
        default=None,
        help="Override max token length from config",
    )
    parser.add_argument(
        "--translator-model",
        type=str,
        default=None,
        help="Override translation model from config",
    )
    parser.add_argument(
        "--no-ensure-chinese",
        dest="ensure_chinese",
        action="store_false",
        default=None,
        help="Override: do not enforce Chinese output",
    )

    args = parser.parse_args()
    extract_single_video(
        video_path=args.video,
        out_path=args.out,
        config_path=args.config,
        transcript_path=args.transcript,
        whisper_model=args.whisper_model,
        transcript_language=args.transcript_language,
        ensure_chinese=args.ensure_chinese,
        translator_model=args.translator_model,
        bert_model=args.bert_model,
        pooling=args.language_pooling,
        max_length=args.max_transcript_length,
        skip_language=args.skip_language,
    )


if __name__ == "__main__":
    main()
