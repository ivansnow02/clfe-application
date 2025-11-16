import io
import json
import tempfile
from typing import Optional
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from pydantic import BaseModel

from video_feature_extract import extract_features_dict

app = FastAPI(title="CLFE Feature Extraction Service", version="1.0")


# 保留旧的 ExtractRequest 用于向后兼容（如有需要）
class ExtractRequest(BaseModel):
    video_path: str
    config_path: str
    transcript_path: Optional[str] = None
    whisper_model: Optional[str] = None
    transcript_language: Optional[str] = None
    ensure_chinese: Optional[bool] = None
    translator_model: Optional[str] = None
    bert_model: Optional[str] = None
    pooling: Optional[str] = None
    max_length: Optional[int] = None
    skip_language: bool = False


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/extract/json")
async def extract_json(
    video: UploadFile = File(...),
    config_path: str = Form("configs/mmsa_fet_av_language.yaml"),
    whisper_model: Optional[str] = Form(None),
    transcript_language: Optional[str] = Form(None),
    ensure_chinese: Optional[bool] = Form(None),
    translator_model: Optional[str] = Form(None),
    bert_model: Optional[str] = Form(None),
    pooling: Optional[str] = Form(None),
    max_length: Optional[int] = Form(None),
    skip_language: bool = Form(False),
):
    """接收上传的视频文件,提取特征并返回 JSON。"""
    temp_video_path = None
    try:
        # 保存上传的视频到临时文件
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(video.filename or "video.mp4").suffix
        ) as tmp:
            content = await video.read()
            tmp.write(content)
            temp_video_path = tmp.name

        feats = extract_features_dict(
            video_path=temp_video_path,
            config_path=config_path,
            transcript_path=None,
            whisper_model=whisper_model,
            transcript_language=transcript_language,
            ensure_chinese=ensure_chinese,
            translator_model=translator_model,
            bert_model=bert_model,
            pooling=pooling,
            max_length=max_length,
            skip_language=skip_language,
        )
        # Convert numpy arrays to lists for JSON
        out = {}
        for k, v in feats.items():
            if isinstance(v, np.ndarray):
                out[k] = {
                    "shape": v.shape,
                    "dtype": str(v.dtype),
                    "data": v.tolist(),
                }
            else:
                out[k] = v
        return out
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        print(f"[ERROR] /extract/json failed:\n{tb}", flush=True)
        raise HTTPException(status_code=500, detail=f"{e}\n{tb}")
    finally:
        # 清理临时文件
        if temp_video_path and Path(temp_video_path).exists():
            Path(temp_video_path).unlink()


@app.post("/extract/npz")
async def extract_npz(
    video: UploadFile = File(...),
    config_path: str = Form("configs/mmsa_fet_av_language.yaml"),
    compress: bool = Query(True),
    whisper_model: Optional[str] = Form(None),
    transcript_language: Optional[str] = Form(None),
    ensure_chinese: Optional[str] = Form(None),
    translator_model: Optional[str] = Form(None),
    bert_model: Optional[str] = Form(None),
    pooling: Optional[str] = Form(None),
    max_length: Optional[int] = Form(None),
    skip_language: bool = Form(False),
):
    """接收上传的视频文件,提取特征并返回 NPZ 格式。"""
    temp_video_path = None
    try:
        # 保存上传的视频到临时文件
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(video.filename or "video.mp4").suffix
        ) as tmp:
            content = await video.read()
            tmp.write(content)
            temp_video_path = tmp.name

        feats = extract_features_dict(
            video_path=temp_video_path,
            config_path=config_path,
            transcript_path=None,
            whisper_model=whisper_model,
            transcript_language=transcript_language,
            ensure_chinese=ensure_chinese
            if ensure_chinese is None
            else ensure_chinese.lower() == "true",
            translator_model=translator_model,
            bert_model=bert_model,
            pooling=pooling,
            max_length=max_length,
            skip_language=skip_language,
        )
        # Prepare NPZ payload - validate audio/video exist
        # MMSA-FET 返回的 key 是 'vision' 不是 'video'
        audio = feats.get("audio")
        video_feat = feats.get("vision")
        if video_feat is None:
            video_feat = feats.get("video")
        if audio is None or video_feat is None:
            raise ValueError(
                f"Missing audio or video features in extraction result. Available keys: {list(feats.keys())}"
            )
        language = feats.get("language")
        lang_json = (
            json.dumps(language, ensure_ascii=False).encode("utf-8")
            if language
            else b"{}"
        )
        lang_arr = np.frombuffer(lang_json, dtype=np.uint8)
        buf = io.BytesIO()
        if compress:
            np.savez_compressed(
                buf, audio=audio, video=video_feat, language_json=lang_arr
            )
        else:
            np.savez(buf, audio=audio, video=video_feat, language_json=lang_arr)
        buf.seek(0)
        from fastapi.responses import Response

        return Response(
            content=buf.read(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename=features.npz"},
        )
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        print(f"[ERROR] /extract/npz failed:\n{tb}", flush=True)
        raise HTTPException(status_code=500, detail=f"{e}\n{tb}")
    finally:
        # 清理临时文件
        if temp_video_path and Path(temp_video_path).exists():
            Path(temp_video_path).unlink()
