import os
import cv2
import torch
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import sys
from decord import VideoReader, cpu
import av
import torch.nn.functional as F

from wan.modules.vae2_2 import Wan2_2_VAE
from wan.modules.t5 import T5EncoderModel


class MultiCamLatentExtractor:
    """多摄像头视频潜在特征提取器"""
    
    def __init__(
        self,
        vae_checkpoint_path: str,
        t5_checkpoint_path: str,
        t5_tokenizer_path: str,
        text_len: int = 256,
        target_h: int = 256,
        target_w: int = 256,
        target_fps: int = 10,
        device: str = "cuda"
    ):
        """
        初始化提取器
        
        Args:
            vae_checkpoint_path: VAE 权重文件路径
            t5_checkpoint_path: T5 文本编码器权重路径
            t5_tokenizer_path: T5 tokenizer 路径（本地目录或 "google/umt5-xxl"）
            text_len: 文本序列最大长度（推荐 256）
            target_h: 目标高度（必须是 32 的倍数）
            target_w: 目标宽度（必须是 32 的倍数）
            target_fps: 目标帧率
            device: 计算设备
        """
        self.target_h = target_h
        self.target_w = target_w
        self.target_fps = target_fps
        self.device = device
        
        # 验证分辨率
        assert target_h % 32 == 0, f"Height must be multiple of 32, got {target_h}"
        assert target_w % 32 == 0, f"Width must be multiple of 32, got {target_w}"
        
        # ========== 加载 VAE ==========
        print(f"Loading VAE from {vae_checkpoint_path}...")
        self.vae = Wan2_2_VAE(
            vae_pth=vae_checkpoint_path,
            device=device
        )
        self.vae.model.eval()
        print("✅ VAE loaded successfully!")
        
        # ========== 加载 T5 文本编码器 ==========
        print(f"Loading T5 encoder from {t5_checkpoint_path}...")
        print(f"Tokenizer path: {t5_tokenizer_path}, text_len: {text_len}")
        
        # ✅ 修复 1: 传入 tokenizer_path 参数
        self.t5_encoder = T5EncoderModel(
            text_len=text_len,
            dtype=torch.bfloat16,
            device=device,
            checkpoint_path=t5_checkpoint_path,
            tokenizer_path=t5_tokenizer_path,  # ✅ 关键：传入 tokenizer 路径
            shard_fn=None
        )
        print("✅ T5 encoder loaded successfully!")
        
    
    def extract_frames_from_video(
        self,
        video_path: str,
        start_frame: int = 0,
        end_frame: int = None
    ) -> tuple:
        """
        使用 PyAV 从 AV1 视频中提取并预处理帧
        """
        try:
            container = av.open(str(video_path))
            # 显式指定使用第一个视频流
            stream = container.streams.video[0]
            # 某些 AV1 视频可能没有设置正确的 thread_count，手动设置以加速
            stream.thread_type = "AUTO" 
        except Exception as e:
            print(f"    ❌ Cannot open video with PyAV: {e}")
            return [], [], 0

        ori_fps = float(stream.average_rate)
        
        # 计算采样间隔
        skip_interval = int(ori_fps / self.target_fps)
        if skip_interval < 1:
            skip_interval = 1
        
        frames = []
        frame_ids = []
        
        # 计数器
        current_idx = 0
        
        # 逐帧解码
        for frame in container.decode(video=0):
            # 如果设置了结束帧且已到达，停止解码
            if end_frame is not None and current_idx >= end_frame:
                break
                
            # 检查是否在目标范围内且符合采样间隔
            if current_idx >= start_frame:
                relative_idx = current_idx - start_frame
                if relative_idx % skip_interval == 0:
                    # 转换为 RGB (PyAV 默认可能是 yuv420p)
                    img = frame.to_image() # 得到 PIL.Image 对象
                    
                    # Resize
                    img = img.resize((self.target_w, self.target_h), Image.Resampling.BILINEAR)
                    
                    # 转为 numpy 数组 (H, W, 3)
                    frames.append(np.array(img))
                    frame_ids.append(relative_idx)
                    
            current_idx += 1
        
        container.close()

        # 调整帧数满足 Wan2.2 的 (N-1) % 4 == 0 要求
        n = len(frames)
        if n == 0:
            return frames, frame_ids, ori_fps
        
        remainder = (n - 1) % 4
        if remainder != 0:
            frames = frames[:-remainder]
            frame_ids = frame_ids[:-remainder]
            print(f"    Trimmed from {n} to {len(frames)} frames to satisfy (N-1)%4==0")
        
        return frames, frame_ids, ori_fps
    
    def encode_to_latent(self, frames: List[np.ndarray]) -> tuple:
        """
        将帧编码为 latent 特征
        
        Args:
            frames: 帧列表，每个元素 shape [H, W, C]
            
        Returns:
            latent: latent 特征 [N, C]
            latent_num_frames: 时间维度帧数
            latent_h: 潜在空间高度
            latent_w: 潜在空间宽度
        """
        if len(frames) == 0:
            raise ValueError("No frames to encode")
        
        # 转换为 tensor: [T, H, W, C] -> [1, C, T, H, W]
        frames_array = np.stack(frames)  # [T, H, W, C]
        frames_tensor = torch.from_numpy(frames_array).float() / 255.0  # [0, 1]
        frames_tensor = frames_tensor * 2 - 1  # [-1, 1]
        frames_tensor = frames_tensor.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, T, H, W]
        
        # 移动到设备
        frames_tensor = frames_tensor.to(self.device)
        
        # 编码
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                latent = self.vae.model.encode(frames_tensor, self.vae.scale)
        
        # 获取 latent 维度信息
        latent_num_frames = latent.shape[2]
        latent_h = latent.shape[3]
        latent_w = latent.shape[4]
        latent_channels = latent.shape[1]
        
        # 展平: [1, C, T, H, W] -> [T*H*W, C]
        latent = latent.squeeze(0)  # [C, T, H, W]
        latent = latent.permute(1, 2, 3, 0)  # [T, H, W, C]
        latent = latent.reshape(-1, latent_channels)  # [T*H*W, C]
        latent = latent.to(torch.bfloat16).cpu()
        
        return latent, latent_num_frames, latent_h, latent_w
    
    def encode_text(self, text: str) -> torch.Tensor:
        """
        使用 T5EncoderModel 提取文本嵌入
        
        Args:
            text: 文本描述
            
        Returns:
            text_emb: 文本嵌入 [L, D], dtype=bfloat16
        """
        if not text or text.strip() == "":
            # 返回空嵌入
            return torch.zeros(1, 4096, dtype=torch.bfloat16)  # umt5-xxl hidden size = 4096
        
        try:
            # ✅ 修复 2: T5EncoderModel 使用 __call__ 方法，返回 List[Tensor]
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    result = self.t5_encoder([text], device=self.device)
            
            # result 是 list, 取第一个元素 [actual_len, 4096]
            text_emb = result[0]
            text_emb = text_emb.to(torch.bfloat16).cpu()
            
            if text_emb.shape[0] < self.t5_encoder.text_len:
            # 在维度1（seq_len）右侧补零
                pad_size = self.t5_encoder.text_len - text_emb.shape[0]
                text_emb = F.pad(
                    text_emb, 
                    (0, 0, 0, pad_size),  # (左, 右, 上, 下) - 在seq_len维度的右侧补
                    value=0
                )
            
            return text_emb
            
        except Exception as e:
            print(f"⚠️  Text encoding failed: {e}, returning zero embedding")
            return torch.zeros(1, 4096, dtype=torch.bfloat16)
    
    def process_episode(
        self,
        episode_data: Dict[str, Any],
        dataset_dir: Path,
        output_dir: Path,
        camera_name: str = "cam_high"
    ) -> Optional[List[Dict[str, Any]]]:
        """
        处理单个 episode 的单个摄像头视角
        
        Args:
            episode_data: episode 元数据（从 episodes.jsonl 读取）
            dataset_dir: 数据集根目录
            output_dir: 输出目录
            camera_name: 摄像头名称
            
        Returns:
            latent_dict: 包含所有字段的字典，或 None（如果处理失败）
        """
        episode_index = episode_data['episode_index']
        action_configs = episode_data.get('action_config', [])
        
        if not action_configs:
            print(f"⚠️  Episode {episode_index}: No action_config found, skipping...")
            return None
        
        # 构建视频路径
        video_rel_path = f"chunk-000/observation.images.{camera_name}/episode_{episode_index:06d}.mp4"
        video_path = dataset_dir / "videos" / video_rel_path
        
        if not video_path.exists():
            print(f"⚠️  Video not found: {video_path}, skipping...")
            return None
        
        print(f"\n{'='*60}")
        print(f"Processing Episode {episode_index} - Camera: {camera_name}")
        print(f"Video: {video_path}")
        print(f"{'='*60}")
        
        all_latent_dicts = []
        
        # 处理每个 action segment
        for seg_idx, action_config in enumerate(action_configs):
            start_frame = action_config.get('start_frame', 0)
            end_frame = action_config.get('end_frame', episode_data.get('length'))
            action_text = action_config.get('action_text', '')
            
            print(f"\n  Segment {seg_idx + 1}/{len(action_configs)}:")
            print(f"    Frames: {start_frame} - {end_frame}")
            if len(action_text) > 80:
                print(f"    Text: {action_text[:80]}...")
            else:
                print(f"    Text: {action_text}")
            
            try:
                # 1. 提取帧
                frames, frame_ids, ori_fps = self.extract_frames_from_video(
                    str(video_path), start_frame, end_frame
                )
                
                if len(frames) == 0:
                    print(f"    ⚠️  No frames extracted, skipping segment...")
                    continue
                
                print(f"    Extracted {len(frames)} frames")
                
                # 2. 编码为 latent
                latent, latent_num_frames, latent_h, latent_w = self.encode_to_latent(frames)
                print(f"    Latent shape: {latent.shape} (frames: {latent_num_frames})")
                
                # 3. 编码文本
                text_emb = self.encode_text(action_text)
                print(f"    Text embedding shape: {text_emb.shape}")
                
                # 4. 构建输出字典
                latent_dict = {
                    'latent': latent,  # [N, C]
                    'latent_num_frames': latent_num_frames,
                    'latent_height': latent_h,
                    'latent_width': latent_w,
                    'video_num_frames': len(frames),
                    'video_height': self.target_h,
                    'video_width': self.target_w,
                    'text_emb': text_emb,
                    'text': action_text,
                    'frame_ids': frame_ids,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'fps': self.target_fps,
                    'ori_fps': ori_fps,
                }
                
                # 5. 保存
                output_path = output_dir / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(latent_dict, output_path)
                print(f"    ✅ Saved to {output_path}")
                
                all_latent_dicts.append(latent_dict)
                
            except Exception as e:
                print(f"    ❌ Error processing segment: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return all_latent_dicts if all_latent_dicts else None


def process_dataset(
    dataset_dir: str,
    vae_checkpoint_path: str,
    t5_checkpoint_path: str,
    t5_tokenizer_path: str,
    text_len: int = 256,
    output_dir: str = None,
    target_h: int = 256,
    target_w: int = 256,
    target_fps: int = 10,
    cameras: List[str] = None
):
    """
    处理整个数据集（多摄像头）
    
    Args:
        dataset_dir: 数据集目录（包含 videos/ 和 meta/）
        vae_checkpoint_path: VAE 权重路径
        t5_checkpoint_path: T5 文本编码器权重路径
        t5_tokenizer_path: T5 tokenizer 路径
        text_len: 文本序列最大长度
        output_dir: 输出目录（默认为 dataset_dir/latents）
        target_h: 目标高度
        target_w: 目标宽度
        target_fps: 目标帧率
        cameras: 摄像头列表
    """
    dataset_dir = Path(dataset_dir)
    if output_dir is None:
        output_dir = dataset_dir / "latents"
    else:
        output_dir = Path(output_dir)
    
    # 默认摄像头列表
    if cameras is None:
        cameras = [
            "cam_high",
            "cam_left_wrist",
            "cam_right_wrist"
        ]
    
    # 读取 episodes.jsonl
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise ValueError(f"episodes.jsonl not found at {episodes_path}")
    
    episodes = []
    with open(episodes_path, 'r') as f:
        for line in f:
            episodes.append(json.loads(line))
    
    print(f"📚 Found {len(episodes)} episodes")
    print(f"📹 Cameras: {cameras}")
    print(f"🎯 Target resolution: {target_w}x{target_h}@{target_fps}fps")
    print(f"💾 Output directory: {output_dir}")
    
    # ✅ 关键：如果 tokenizer 路径是相对路径，转换为绝对路径
    if t5_tokenizer_path and not os.path.isabs(t5_tokenizer_path):
        t5_tokenizer_path = os.path.abspath(t5_tokenizer_path)
        print(f"🔧 Converted tokenizer path to absolute: {t5_tokenizer_path}")
    
    # 初始化提取器
    extractor = MultiCamLatentExtractor(
        vae_checkpoint_path=vae_checkpoint_path,
        t5_checkpoint_path=t5_checkpoint_path,
        t5_tokenizer_path=t5_tokenizer_path,  # ✅ 现在正确传递
        text_len=text_len,
        target_h=target_h,
        target_w=target_w,
        target_fps=target_fps
    )
    
    # 处理每个 episode 和每个摄像头
    total_segments = 0
    for i, episode in enumerate(episodes):
        print(f"\n{'#'*60}")
        print(f"# Episode {i + 1}/{len(episodes)}")
        print(f"{'#'*60}")
        
        for camera in cameras:
            try:
                result = extractor.process_episode(
                    episode_data=episode,
                    dataset_dir=dataset_dir,
                    output_dir=output_dir / "chunk-000" / f"observation.images.{camera}",
                    camera_name=camera
                )
                
                if result:
                    total_segments += len(result)
                
            except Exception as e:
                print(f"❌ Error processing {camera}: {e}")
                continue
    
    print(f"\n{'='*60}")
    print(f"✅ All episodes processed!")
    print(f"📊 Total segments saved: {total_segments}")
    print(f"💾 Latents directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract multi-camera video latents using Wan2.2 VAE")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--vae_checkpoint", type=str, required=True, help="VAE checkpoint path")
    parser.add_argument("--t5_checkpoint", type=str, required=True, help="T5 encoder checkpoint path")
    parser.add_argument("--t5_tokenizer", type=str, default="google/umt5-xxl", help="T5 tokenizer path (local dir or 'google/umt5-xxl')")
    parser.add_argument("--text_len", type=int, default=512, help="Max text sequence length (default: 512)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--height", type=int, default=256, help="Target height")
    parser.add_argument("--width", type=int, default=256, help="Target width")
    parser.add_argument("--fps", type=int, default=10, help="Target FPS")
    parser.add_argument("--cameras", type=str, nargs='+', default=None, 
                       help="Camera names (default: cam_high cam_left_wrist cam_right_wrist)")
    
    args = parser.parse_args()
    
    process_dataset(
        dataset_dir=args.dataset_dir,
        vae_checkpoint_path=args.vae_checkpoint,
        t5_checkpoint_path=args.t5_checkpoint,
        t5_tokenizer_path=args.t5_tokenizer,
        text_len=args.text_len,
        output_dir=args.output_dir,
        target_h=args.height,
        target_w=args.width,
        target_fps=args.fps,
        cameras=args.cameras
    )