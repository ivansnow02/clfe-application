import os
from pathlib import Path
from typing import Tuple

import gradio as gr
import numpy as np
import torch

from src.clfe.cli import load_trained_model
from src.clfe.fet_client import fetch_features_npz


def _pad_truncate(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad or truncate a (L, D) array to target_len along L with zeros."""
    if arr.shape[0] == target_len:
        return arr
    if arr.shape[0] > target_len:
        return arr[:target_len]
    pad_len = target_len - arr.shape[0]
    pad = np.zeros((pad_len, arr.shape[1]), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


def _prepare_text_tensor(text: str, tokenizer, max_len: int) -> torch.Tensor:
    """Tokenize text to (1,3,S) tensor expected by model (ids, mask, segment)."""
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].squeeze(0)  # (S,)
    attention_mask = encoded["attention_mask"].squeeze(0)  # (S,)
    # Some tokenizers may not provide token_type_ids (e.g. roberta); create zeros
    segment_ids = encoded.get("token_type_ids", torch.zeros_like(input_ids))
    if segment_ids.dim() > 1:
        segment_ids = segment_ids.squeeze(0)  # (S,)
    stacked = torch.stack([
        input_ids.long(),
        attention_mask.float(),
        segment_ids.long(),
    ])  # (3,S)
    return stacked.unsqueeze(0)  # (1,3,S)


def _prepare_modal_tensor(
    arr: np.ndarray, target_len: int, device: torch.device
) -> torch.Tensor:
    arr_proc = _pad_truncate(arr, target_len)
    return torch.tensor(arr_proc, dtype=torch.float32, device=device).unsqueeze(0)


def build_predict_fn():
    """Build prediction function with cached model + config loaded once."""
    config_path = os.environ.get("CLFE_CONFIG", "notebook_configs/sims.yaml")
    ckpt_path = os.environ.get(
        "CLFE_CKPT", "ckpt/ALMT_Demo_SIMS/best_binary_epoch_5.pth"
    )
    tools_server = os.environ.get("TOOLS_SERVER", "http://localhost:8000")
    # Feature extraction config (path inside tools container)
    fet_config = os.environ.get(
        "FET_CONFIG",
        "configs/mmsa_fet_av_language.yaml",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, args = load_trained_model(config_path, ckpt_path, device)
    # type: ignore for dynamic attribute from wrapped module
    tokenizer = model.bertmodel.tokenizer  # type: ignore[attr-defined]
    max_text_len = getattr(args.model, "l_input_length", 50)
    a_input_len = args.model.a_input_length
    v_input_len = args.model.v_input_length

    def predict(video_file) -> Tuple[str, float, dict]:
        if video_file is None:
            return "未上传视频", 0.0, {}
        # Gradio supplies a temp file path already; ensure Path
        if isinstance(video_file, dict) and "name" in video_file:
            # Newer Gradio returns dict; use 'name'
            video_path = Path(video_file["name"])  # type: ignore
        else:
            video_path = Path(str(video_file))
        if not video_path.exists():
            return f"视频文件不存在: {video_path}", 0.0, {}

        # Call tools feature extraction service with local file upload
        try:
            feats = fetch_features_npz(tools_server, str(video_path), fet_config)
        except Exception as e:
            return f"特征服务调用失败: {e}", 0.0, {}

        audio = feats.get("audio")
        video = feats.get("video")
        language = feats.get("language", {})
        transcript = language.get("transcript") or language.get("raw_transcript") or ""

        if transcript:
            text_tensor = _prepare_text_tensor(transcript, tokenizer, max_text_len).to(
                device
            )
        else:
            # Fallback: empty string (will be mostly padding)
            text_tensor = _prepare_text_tensor("", tokenizer, max_text_len).to(device)

        if audio is None or video is None:
            return "特征缺失(audio/video)", 0.0, {"error": "服务未返回音频或视频特征"}
        audio_tensor = _prepare_modal_tensor(audio, a_input_len, device)
        video_tensor = _prepare_modal_tensor(video, v_input_len, device)

        with torch.no_grad():
            output = model(video_tensor, audio_tensor, text_tensor)
        score = float(output.squeeze().cpu().item())

        info = {
            "transcript": transcript,
            "audio_shape": list(audio_tensor.shape),
            "video_shape": list(video_tensor.shape),
            "text_shape": list(text_tensor.shape),
            "raw_score": score,
        }
        return transcript or "(无字幕)", score, info

    return predict


def launch():
    predict_fn = build_predict_fn()
    with gr.Blocks(title="CLFE 单视频推理") as demo:
        gr.Markdown(
            "## 🎬 CLFE 单视频情感/回归推理\n上传一个视频，后端自动提取多模态特征并输出模型分数。"
        )
        with gr.Row():
            video_input = gr.Video(
                label="上传视频", sources=["upload"], elem_id="video"
            )
        run_btn = gr.Button("开始推理")
        transcript_out = gr.Textbox(label="中文字幕", interactive=False)
        score_out = gr.Number(label="模型分数", precision=4)
        info_out = gr.JSON(label="详细信息")

        def _run(video_file):
            return predict_fn(video_file)

        run_btn.click(
            _run, inputs=[video_input], outputs=[transcript_out, score_out, info_out]
        )
    demo.launch(
        server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_PORT", 7860))
    )


if __name__ == "__main__":
    launch()
