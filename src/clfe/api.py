import os
import shutil
import tempfile
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from huggingface_hub import hf_hub_download

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


ml_models: Dict[str, Any] = {}
semaphore: asyncio.Semaphore = None  # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load configuration
    config_path = os.environ.get("CLFE_CONFIG", "configs/sims.yaml")
    ckpt_path = os.environ.get("CLFE_CKPT", "ckpt/best_binary_epoch_5.pth")

    if not os.path.exists(ckpt_path):
        print(
            f"Checkpoint not found at {ckpt_path}. Attempting download from Hugging Face..."
        )
        try:
            repo_id = "IvanSnow02/clfe"
            filename = "best_binary_epoch_5.pth"
            local_dir = os.path.dirname(ckpt_path)
            os.makedirs(local_dir, exist_ok=True)
            ckpt_path = hf_hub_download(
                repo_id=repo_id, filename=filename, local_dir=local_dir
            )
            print(f"Downloaded checkpoint to {ckpt_path}")
        except Exception as e:
            print(f"Failed to download checkpoint: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {ckpt_path} with config {config_path}...")
    try:
        model, args = load_trained_model(config_path, ckpt_path, device)
        ml_models["model"] = model
        ml_models["args"] = args
        # type: ignore for dynamic attribute from wrapped module
        ml_models["tokenizer"] = model.bertmodel.tokenizer  # type: ignore[attr-defined]
        ml_models["device"] = device
        ml_models["max_text_len"] = getattr(args.model, "l_input_length", 50)
        ml_models["a_input_len"] = args.model.a_input_length
        ml_models["v_input_len"] = args.model.v_input_length
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

    # Initialize semaphore for concurrency control
    max_concurrent = int(os.environ.get("MAX_CONCURRENT_TASKS", "1"))
    global semaphore
    semaphore = asyncio.Semaphore(max_concurrent)
    print(f"Concurrency limit set to {max_concurrent}")

    yield

    ml_models.clear()


app = FastAPI(lifespan=lifespan)


def process_prediction(video_path: Path):
    """Synchronous function to handle the heavy lifting of prediction."""
    tools_server = os.environ.get("TOOLS_SERVER", "http://localhost:8000")
    fet_config = os.environ.get("FET_CONFIG", "configs/mmsa_fet_av_language.yaml")

    model = ml_models["model"]
    tokenizer = ml_models["tokenizer"]
    device = ml_models["device"]
    max_text_len = ml_models["max_text_len"]
    a_input_len = ml_models["a_input_len"]
    v_input_len = ml_models["v_input_len"]

    # 1. Feature Extraction
    try:
        feats = fetch_features_npz(tools_server, str(video_path), fet_config)
    except Exception as e:
        raise RuntimeError(f"Feature extraction failed: {e}")

    audio = feats.get("audio")
    video = feats.get("video")
    language = feats.get("language", {})
    transcript = language.get("transcript") or language.get("raw_transcript") or ""

    if audio is None or video is None:
        raise RuntimeError("Service returned missing audio or video features")

    # 2. Prepare Tensors
    if transcript:
        text_tensor = _prepare_text_tensor(transcript, tokenizer, max_text_len).to(
            device
        )
    else:
        text_tensor = _prepare_text_tensor("", tokenizer, max_text_len).to(device)

    audio_tensor = _prepare_modal_tensor(audio, a_input_len, device)
    video_tensor = _prepare_modal_tensor(video, v_input_len, device)

    # 3. Model Inference
    with torch.no_grad():
        output = model(video_tensor, audio_tensor, text_tensor)
    score = float(output.squeeze().cpu().item())

    return {
        "transcript": transcript or "(无字幕)",
        "audio_shape": list(audio_tensor.shape),
        "video_shape": list(video_tensor.shape),
        "text_shape": list(text_tensor.shape),
        "raw_score": score,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not semaphore:
        raise HTTPException(status_code=503, detail="Server not initialized")

    # Use semaphore to limit concurrent processing
    async with semaphore:
        # Create a temporary file to save the upload
        filename = file.filename or "temp_video"
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(filename).suffix
        ) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)

        try:
            # Run the blocking prediction logic in a thread pool
            result = await run_in_threadpool(process_prediction, tmp_path)
            return result
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
        finally:
            # Clean up temp file
            if tmp_path.exists():
                os.unlink(tmp_path)


if __name__ == "__main__":
    port = int(os.environ.get("API_PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
