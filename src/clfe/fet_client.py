import json
from io import BytesIO
from pathlib import Path
from typing import Dict, Any

import numpy as np
import requests


def fetch_features_npz(
    server_url: str,
    video_path: str,
    config_path: str = "configs/mmsa_fet_av_language.yaml",
) -> Dict[str, Any]:
    """Call tools API to extract features and return dict with arrays and language metadata.

    server_url: like http://tools:8000 or http://localhost:8000
    video_path: local path to video file (will be uploaded)
    config_path: config path inside tools container
    """
    url = server_url.rstrip("/") + "/extract/npz?compress=1"

    with open(video_path, "rb") as f:
        files = {"video": (Path(video_path).name, f, "video/mp4")}
        data = {"config_path": config_path, "skip_language": "false"}
        r = requests.post(url, files=files, data=data, timeout=3600)

    r.raise_for_status()
    buf = BytesIO(r.content)
    with np.load(buf) as npz:
        audio = npz["audio"]
        video = npz["video"]
        lang_json = bytes(npz["language_json"]).decode("utf-8")
        language = json.loads(lang_json)
    return {"audio": audio, "video": video, "language": language}


def fetch_features_json(
    server_url: str,
    video_path: str,
    config_path: str = "configs/mmsa_fet_av_language.yaml",
) -> Dict[str, Any]:
    """Call tools API to extract features and return dict with JSON-serialized arrays.

    server_url: like http://tools:8000 or http://localhost:8000
    video_path: local path to video file (will be uploaded)
    config_path: config path inside tools container
    """
    url = server_url.rstrip("/") + "/extract/json"

    with open(video_path, "rb") as f:
        files = {"video": (Path(video_path).name, f, "video/mp4")}
        data = {"config_path": config_path, "skip_language": "false"}
        r = requests.post(url, files=files, data=data, timeout=3600)

    r.raise_for_status()
    data = r.json()
    # Convert lists back to numpy
    for k, v in list(data.items()):
        if isinstance(v, dict) and {"shape", "dtype", "data"}.issubset(v.keys()):
            arr = np.array(v["data"], dtype=v["dtype"]).reshape(v["shape"])
            data[k] = arr
    return data
