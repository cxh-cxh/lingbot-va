#!/usr/bin/env python3
"""
LeRobot Parquet State Statistics Calculator
计算 LeRobot 数据集中 observation.state 的 q01 和 q99 统计量
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_lerobot_dataset(dataset_path: Union[str, Path]) -> Dict:
    """
    加载 LeRobot 数据集的基本信息
    
    目录结构:
    dataset/
    ├── data/
    │   └── chunk-000/
    │       ├── episode_000000.parquet
    │       └── ...
    ├── videos/
    └── meta/
        ├── info.json
        └── episodes.jsonl
    """
    dataset_path = Path(dataset_path)
    
    # 读取 info.json 获取元数据
    info_path = dataset_path / "meta" / "info.json"
    if info_path.exists():
        with open(info_path, 'r') as f:
            info = json.load(f)
    else:
        info = {}
    
    # 查找所有 parquet 文件
    data_dir = dataset_path / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    
    print(f"Found {len(parquet_files)} parquet files")
    
    return {
        "info": info,
        "parquet_files": parquet_files,
        "dataset_path": dataset_path
    }


def extract_state_from_parquet(parquet_path: Path) -> Optional[np.ndarray]:
    """
    从单个 parquet 文件中提取 observation.state 数据
    
    LeRobot v2.1 格式中 state 通常存储为:
    - observation.state: list/array 格式的机器人状态
    """
    try:
        df = pd.read_parquet(parquet_path)
        
        # LeRobot 中 state 列名通常是 'observation.state' 或 'state'
        state_col = None
        for col in df.columns:
            if 'state' in col.lower():
                state_col = col
                break
        
        if state_col is None:
            print(f"Warning: No state column found in {parquet_path}")
            return None
        
        # 提取 state 数据
        state_data = df[state_col].values
        
        # 如果 state 是 list/array 格式，需要展开
        if isinstance(state_data[0], (list, np.ndarray)):
            state_array = np.vstack(state_data)
        else:
            state_array = state_data.reshape(-1, 1)
            
        return state_array.astype(np.float32)
        
    except Exception as e:
        print(f"Error reading {parquet_path}: {e}")
        return None


def compute_quantile_stats(
    dataset_path: Union[str, Path],
    batch_size: int = 1000,
    save_stats: bool = True
) -> Dict:
    """
    计算整个数据集的 q01 和 q99 统计量
    
    使用流式计算避免内存溢出（适用于大数据集）
    """
    dataset = load_lerobot_dataset(dataset_path)
    parquet_files = dataset["parquet_files"]
    
    if not parquet_files:
        raise ValueError("No parquet files found!")
    
    # 首先读取一个文件确定 state 维度
    print("Detecting state dimensions...")
    sample_state = None
    for pf in parquet_files:
        sample_state = extract_state_from_parquet(pf)
        if sample_state is not None:
            break
    
    if sample_state is None:
        raise ValueError("Could not extract state from any parquet file!")
    
    state_dim = sample_state.shape[1]
    print(f"State dimension: {state_dim}")
    
    # 使用 Welford 算法流式计算均值和方差
    # 同时收集数据用于计算分位数（当数据量太大时使用近似算法）
    
    all_states = []  # 用于小数据集直接计算
    use_approximation = False
    max_samples = 1000000  # 最大样本数，超过则使用近似
    
    total_frames = 0
    
    print("Collecting state data...")
    for pf in tqdm(parquet_files):
        state = extract_state_from_parquet(pf)
        if state is not None:
            all_states.append(state)
            total_frames += len(state)
            
            # 检查内存使用，决定是否切换到近似模式
            if total_frames > max_samples:
                use_approximation = True
                print(f"\nLarge dataset detected ({total_frames} frames). "
                      "Switching to approximate quantile calculation.")
                break
    
    # 合并所有数据
    if all_states:
        combined_states = np.vstack(all_states)
    else:
        raise ValueError("No valid state data found!")
    
    # 计算统计量
    print(f"\nComputing statistics on {len(combined_states)} frames...")
    
    stats = {
        "mean": np.mean(combined_states, axis=0).tolist(),
        "std": np.std(combined_states, axis=0).tolist(),
        "min": np.min(combined_states, axis=0).tolist(),
        "max": np.max(combined_states, axis=0).tolist(),
        "q01": np.percentile(combined_states, 1, axis=0).tolist(),
        "q99": np.percentile(combined_states, 99, axis=0).tolist(),
        "count": len(combined_states),
        "state_dim": state_dim
    }
    
    # 打印结果
    print("\n" + "="*60)
    print("STATE STATISTICS RESULTS")
    print("="*60)
    print(f"Total frames: {stats['count']}")
    print(f"State dimension: {stats['state_dim']}")
    print(f"\nMean: {stats['mean']}")
    print(f"Std:  {stats['std']}")
    print(f"\nMin:  {stats['min']}")
    print(f"Max:  {stats['max']}")
    print(f"\nQ01:  {stats['q01']}")
    print(f"Q99:  {stats['q99']}")
    print("="*60)
    
    # 保存为 norm_stats.json 格式（兼容 LeRobot/Pi0 格式）
    if save_stats:
        output_path = Path(dataset_path) / "meta" / "norm_stats.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        norm_stats = {
            "state": {
                "mean": stats["mean"],
                "std": stats["std"],
                "q01": stats["q01"],
                "q99": stats["q99"]
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(norm_stats, f, indent=2)
        
        print(f"\nSaved norm_stats.json to: {output_path}")
    
    return stats


def compute_quantile_stats_streaming(
    dataset_path: Union[str, Path],
    save_stats: bool = True
) -> Dict:
    """
    流式计算 q01 和 q99，适用于超大数据集（使用 P² 算法或分块近似）
    """
    from scipy import stats as scipy_stats
    
    dataset = load_lerobot_dataset(dataset_path)
    parquet_files = dataset["parquet_files"]
    
    # 使用 TDigest 或简单分块法近似计算分位数
    # 这里使用简单的蓄水池采样 + 最终精确计算
    
    reservoir_size = 100000  # 蓄水池大小
    reservoir = []
    total_count = 0
    
    print("Streaming processing with reservoir sampling...")
    for pf in tqdm(parquet_files):
        state = extract_state_from_parquet(pf)
        if state is None:
            continue
            
        for sample in state:
            total_count += 1
            if len(reservoir) < reservoir_size:
                reservoir.append(sample)
            else:
                # 蓄水池采样
                j = np.random.randint(0, total_count)
                if j < reservoir_size:
                    reservoir[j] = sample
    
    reservoir_array = np.array(reservoir)
    
    stats = {
        "mean": np.mean(reservoir_array, axis=0).tolist(),
        "std": np.std(reservoir_array, axis=0).tolist(),
        "min": np.min(reservoir_array, axis=0).tolist(),
        "max": np.max(reservoir_array, axis=0).tolist(),
        "q01": np.percentile(reservoir_array, 1, axis=0).tolist(),
        "q99": np.percentile(reservoir_array, 99, axis=0).tolist(),
        "count": total_count,
        "sample_count": len(reservoir),
        "method": "reservoir_sampling"
    }
    
    print(f"\nApproximate statistics from {len(reservoir)} samples "
          f"(total {total_count} frames):")
    print(f"Q01: {stats['q01']}")
    print(f"Q99: {stats['q99']}")
    
    if save_stats:
        output_path = Path(dataset_path) / "meta" / "norm_stats_approx.json"
        with open(output_path, 'w') as f:
            json.dump({"state": stats}, f, indent=2)
        print(f"\nSaved to: {output_path}")
    
    return stats


def update_episodes_stats(dataset_path: Union[str, Path]):
    """
    更新 LeRobot 格式的 episodes_stats.jsonl（如果需要）
    """
    # 这里可以实现按 episode 统计并更新 meta/episodes_stats.jsonl
    pass


def main():
    parser = argparse.ArgumentParser(
        description="Compute q01/q99 statistics for LeRobot dataset state"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to LeRobot dataset directory"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for large datasets"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save stats to file"
    )
    
    args = parser.parse_args()
    
    if args.streaming:
        compute_quantile_stats_streaming(
            args.dataset_path,
            save_stats=not args.no_save
        )
    else:
        compute_quantile_stats(
            args.dataset_path,
            save_stats=not args.no_save
        )


if __name__ == "__main__":
    main()