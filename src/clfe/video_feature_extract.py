import argparse
import pickle
from pathlib import Path

from MSA_FET import FeatureExtractionTool


def load_config(config_path: str):
    """Load YAML config for MMSA-FET and return as dict."""
    import yaml

    cfg_path = Path(config_path)
    with cfg_path.open() as f:
        config = yaml.safe_load(f)
    return config


def extract_single_video(video_path: str, out_path: str, config_path: str):
    video_path = Path(video_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    fet = FeatureExtractionTool(config=config)

    print(f"[INFO] Extracting features for: {video_path}")
    feature = fet.run_single(str(video_path))

    print("[INFO] Extraction done. Keys in feature dict:", feature.keys())

    with out_path.open("wb") as f:
        pickle.dump(feature, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[INFO] Saved features to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract audio+video features for a single video using MMSA-FET and YAML config",
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
        help="Path to YAML config file for MMSA-FET",
    )

    args = parser.parse_args()
    extract_single_video(args.video, args.out, args.config)


if __name__ == "__main__":
    main()
