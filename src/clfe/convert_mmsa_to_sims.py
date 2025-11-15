#!/usr/bin/env python3
"""
将 MMSA-FET 提取的特征转换为 SIMS 模型输入格式
"""

import pickle
import argparse
import numpy as np
from pathlib import Path


def pad_or_truncate(features, target_length):
    """
    填充或截断特征到目标长度（与SIMS数据处理一致）

    - 截断：均匀采样
    - 填充：重复最后一帧

    Args:
        features: (seq_len, feat_dim) numpy array
        target_length: 目标序列长度

    Returns:
        (target_length, feat_dim) numpy array
    """
    current_length = features.shape[0]

    if current_length == target_length:
        return features
    elif current_length > target_length:
        # 截断：均匀采样
        indices = np.linspace(0, current_length - 1, target_length, dtype=int)
        return features[indices]
    else:
        # 填充：重复最后一帧
        pad_length = target_length - current_length
        last_frame = np.repeat(features[-1:], pad_length, axis=0)
        return np.vstack([features, last_frame])


def align_video_features(video_feat, target_dim=177):
    """
    对齐视频特征维度

    MMSA-FET OpenFace 输出 171 维，SIMS 需要 177 维
    可能是不同的 OpenFace 配置（landmark/AU 组合不同）

    Args:
        video_feat: (seq_len, 171) numpy array
        target_dim: 目标特征维度

    Returns:
        (seq_len, target_dim) numpy array
    """
    current_dim = video_feat.shape[1]

    if current_dim == target_dim:
        return video_feat
    elif current_dim < target_dim:
        # 补零到目标维度
        pad_width = ((0, 0), (0, target_dim - current_dim))
        return np.pad(video_feat, pad_width, mode="constant", constant_values=0)
    else:
        # 截断到目标维度（保留前 target_dim 维）
        return video_feat[:, :target_dim]


def convert_mmsa_to_sims(
    mmsa_pkl_path,
    output_path,
    audio_target_len=925,
    video_target_len=232,
    video_target_dim=177,
):
    """
    转换 MMSA-FET 特征到 SIMS 格式

    Args:
        mmsa_pkl_path: MMSA-FET 输出的 pkl 文件路径
        output_path: 转换后的输出路径
        audio_target_len: 音频目标序列长度
        video_target_len: 视频目标序列长度
        video_target_dim: 视频目标特征维度
    """
    # 加载 MMSA-FET 特征
    with open(mmsa_pkl_path, "rb") as f:
        mmsa_data = pickle.load(f)

    print(f"[INFO] Loaded MMSA-FET features from: {mmsa_pkl_path}")
    print(f"  Audio shape: {mmsa_data['audio'].shape}")
    print(f"  Video shape: {mmsa_data['video'].shape}")

    # 处理音频特征
    audio_feat = mmsa_data["audio"]  # (430, 25)
    audio_aligned = pad_or_truncate(audio_feat, audio_target_len)  # (925, 25)

    # 处理视频特征
    video_feat = mmsa_data["video"]  # (45, 171)
    video_dim_aligned = align_video_features(video_feat, video_target_dim)  # (45, 177)
    video_aligned = pad_or_truncate(video_dim_aligned, video_target_len)  # (232, 177)

    # 构建输出字典（保持与 SIMS 数据加载器兼容的格式）
    sims_data = {
        "audio": audio_aligned.astype(np.float32),
        "vision": video_aligned.astype(
            np.float32
        ),  # 注意：SIMS 用 'vision' 而不是 'video'
    }

    print(f"[INFO] Converted features:")
    print(f"  Audio: {audio_feat.shape} -> {audio_aligned.shape}")
    print(f"  Video: {video_feat.shape} -> {video_aligned.shape}")

    # 保存转换后的特征
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(sims_data, f)

    print(f"[INFO] Saved SIMS-compatible features to: {output_path}")

    return sims_data


def main():
    parser = argparse.ArgumentParser(
        description="Convert MMSA-FET features to SIMS model input format"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input MMSA-FET pickle file path"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output SIMS-format pickle file path"
    )
    parser.add_argument(
        "--audio-len",
        type=int,
        default=925,
        help="Target audio sequence length (default: 925)",
    )
    parser.add_argument(
        "--video-len",
        type=int,
        default=232,
        help="Target video sequence length (default: 232)",
    )
    parser.add_argument(
        "--video-dim",
        type=int,
        default=177,
        help="Target video feature dimension (default: 177)",
    )

    args = parser.parse_args()

    convert_mmsa_to_sims(
        mmsa_pkl_path=args.input,
        output_path=args.output,
        audio_target_len=args.audio_len,
        video_target_len=args.video_len,
        video_target_dim=args.video_dim,
    )


if __name__ == "__main__":
    main()
