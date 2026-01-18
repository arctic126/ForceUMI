#!/usr/bin/env python3
"""
Episode Trimmer - 播放并裁剪采集的episode数据

功能:
- 播放HDF5格式的episode数据
- 用户可以通过按键标记裁剪点
- 按下裁剪键后，当前帧之后的所有数据将被删除

用法:
    python trim_episode.py <episode_path.hdf5>
    python trim_episode.py <session_directory>  # 处理整个session目录
    
键盘控制:
    空格         - 播放/暂停
    左/右方向键  - 逐帧后退/前进
    上/下方向键  - 加速/减速播放
    T/C          - 在当前帧裁剪 (保留当前帧及之前的数据)
    S            - 保存裁剪后的数据
    N            - 跳过当前episode，处理下一个
    R            - 重置裁剪点（恢复到原始状态）
    H            - 显示/隐藏帮助
    Q/ESC        - 退出
"""

import sys
import cv2
import h5py
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import deque
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


class EpisodeTrimmer:
    """
    Episode裁剪工具
    
    加载episode数据，提供播放和裁剪功能
    """
    
    def __init__(self, episode_path: str):
        """
        初始化裁剪器
        
        Args:
            episode_path: HDF5 episode文件路径
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.episode_path = Path(episode_path)
        
        # 数据
        self.images = None
        self.states = None
        self.actions = None
        self.forces = None
        self.timestamps = None
        self.metadata = {}
        
        # 额外时间戳
        self.timestamps_camera = None
        self.timestamps_pose = None
        self.timestamps_force = None
        
        # 原始帧数
        self.original_frames = 0
        
        # 裁剪点（None表示未裁剪）
        self.trim_frame = None
        
        # 播放状态
        self.current_frame = 0
        self.total_frames = 0
        self.is_playing = False
        self.playback_speed = 1.0
        self.target_fps = 30
        self.last_frame_time = 0
        
        # UI状态
        self.running = True
        self.show_help = True
        self.modified = False
        
        # 显示设置
        self.image_display_size = (960, 720)
        self.plot_size = (800, 400)
        self.plot_history = 300
        
        # 数据缓冲区
        self.force_buffer = deque(maxlen=self.plot_history)
        self.torque_buffer = deque(maxlen=self.plot_history)
        
        # 加载数据
        self._load_episode()
    
    def _load_episode(self) -> bool:
        """加载episode数据"""
        try:
            if not self.episode_path.exists():
                self.logger.error(f"文件不存在: {self.episode_path}")
                return False
            
            self.logger.info(f"加载episode: {self.episode_path}")
            
            with h5py.File(self.episode_path, 'r') as f:
                # 加载主要数据集
                if 'image' in f:
                    self.images = f['image'][:]
                if 'state' in f:
                    self.states = f['state'][:]
                if 'action' in f:
                    self.actions = f['action'][:]
                if 'force' in f:
                    self.forces = f['force'][:]
                if 'timestamp' in f:
                    self.timestamps = f['timestamp'][:]
                
                # 加载额外时间戳
                if 'timestamp_camera' in f:
                    self.timestamps_camera = f['timestamp_camera'][:]
                if 'timestamp_pose' in f:
                    self.timestamps_pose = f['timestamp_pose'][:]
                if 'timestamp_force' in f:
                    self.timestamps_force = f['timestamp_force'][:]
                
                # 加载元数据
                for key in f.attrs:
                    self.metadata[key] = f.attrs[key]
            
            # 确定总帧数
            if self.images is not None:
                self.total_frames = len(self.images)
            elif self.states is not None:
                self.total_frames = len(self.states)
            elif self.forces is not None:
                self.total_frames = len(self.forces)
            else:
                self.logger.error("文件中没有数据")
                return False
            
            self.original_frames = self.total_frames
            
            # 获取FPS
            if 'fps' in self.metadata:
                self.target_fps = float(self.metadata['fps'])
            elif 'camera_fps' in self.metadata:
                self.target_fps = float(self.metadata['camera_fps'])
            
            self.logger.info(f"已加载: {self.total_frames}帧, {self.target_fps} FPS")
            return True
            
        except Exception as e:
            self.logger.error(f"加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def set_trim_point(self, frame_idx: int):
        """
        设置裁剪点
        
        Args:
            frame_idx: 裁剪帧索引（保留此帧及之前的数据）
        """
        if frame_idx < 0 or frame_idx >= self.original_frames:
            self.logger.warning(f"无效的裁剪点: {frame_idx}")
            return
        
        self.trim_frame = frame_idx
        self.total_frames = frame_idx + 1  # 保留0到frame_idx
        self.modified = True
        
        # 如果当前帧超过裁剪点，跳回裁剪点
        if self.current_frame > frame_idx:
            self.current_frame = frame_idx
        
        self.logger.info(f"裁剪点设置在帧 {frame_idx}，保留 {self.total_frames} 帧")
    
    def reset_trim(self):
        """重置裁剪点"""
        self.trim_frame = None
        self.total_frames = self.original_frames
        self.modified = False
        self.logger.info("裁剪点已重置")
    
    def save(self, output_path: Optional[str] = None) -> bool:
        """
        保存裁剪后的数据（直接覆盖原文件）
        
        Args:
            output_path: 输出路径，None则覆盖原文件
            
        Returns:
            是否保存成功
        """
        if self.trim_frame is None:
            self.logger.warning("没有设置裁剪点，无需保存")
            return False
        
        try:
            output_path = Path(output_path) if output_path else self.episode_path
            
            # 裁剪后的帧数
            end_idx = self.trim_frame + 1
            
            # 准备裁剪后的数据
            trimmed_data = {}
            
            if self.images is not None:
                trimmed_data['image'] = self.images[:end_idx]
            if self.states is not None:
                trimmed_data['state'] = self.states[:end_idx]
            if self.actions is not None:
                trimmed_data['action'] = self.actions[:end_idx]
            if self.forces is not None:
                trimmed_data['force'] = self.forces[:end_idx]
            if self.timestamps is not None:
                trimmed_data['timestamp'] = self.timestamps[:end_idx]
            if self.timestamps_camera is not None:
                trimmed_data['timestamp_camera'] = self.timestamps_camera[:end_idx]
            if self.timestamps_pose is not None:
                trimmed_data['timestamp_pose'] = self.timestamps_pose[:end_idx]
            if self.timestamps_force is not None:
                trimmed_data['timestamp_force'] = self.timestamps_force[:end_idx]
            
            # 更新元数据
            trimmed_metadata = dict(self.metadata)
            trimmed_metadata['num_frames'] = end_idx
            trimmed_metadata['trimmed'] = True
            trimmed_metadata['original_frames'] = self.original_frames
            trimmed_metadata['trim_timestamp'] = datetime.now().isoformat()
            
            # 重新计算持续时间
            if self.timestamps is not None:
                trimmed_metadata['duration'] = float(self.timestamps[end_idx-1] - self.timestamps[0])
            
            # 保存到HDF5
            with h5py.File(output_path, 'w') as f:
                # 保存数据集
                for key, data in trimmed_data.items():
                    if key == 'image':
                        f.create_dataset(
                            key, data=data,
                            compression='gzip', compression_opts=4,
                            chunks=(1, *data.shape[1:])
                        )
                    else:
                        f.create_dataset(
                            key, data=data,
                            compression='gzip', compression_opts=4
                        )
                
                # 保存元数据
                for key, value in trimmed_metadata.items():
                    f.attrs[key] = value
            
            self.logger.info(f"裁剪后的数据已保存到: {output_path}")
            self.logger.info(f"帧数: {self.original_frames} -> {end_idx}")
            
            # 更新内部状态
            self.modified = False
            
            return True
            
        except Exception as e:
            self.logger.error(f"保存失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_frame(self, frame_idx: Optional[int] = None) -> Dict[str, Any]:
        """获取指定帧的数据"""
        if frame_idx is None:
            frame_idx = self.current_frame
        
        frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        
        frame_data = {'frame_idx': frame_idx}
        
        if self.images is not None:
            frame_data['image'] = self.images[frame_idx]
        else:
            frame_data['image'] = None
        
        if self.states is not None:
            frame_data['state'] = self.states[frame_idx]
        else:
            frame_data['state'] = None
        
        if self.actions is not None:
            frame_data['action'] = self.actions[frame_idx]
        else:
            frame_data['action'] = None
        
        if self.forces is not None:
            frame_data['force'] = self.forces[frame_idx]
        else:
            frame_data['force'] = None
        
        if self.timestamps is not None:
            frame_data['timestamp'] = self.timestamps[frame_idx]
        else:
            frame_data['timestamp'] = None
        
        return frame_data
    
    def play(self):
        """开始播放"""
        self.is_playing = True
        self.last_frame_time = cv2.getTickCount() / cv2.getTickFrequency()
    
    def pause(self):
        """暂停播放"""
        self.is_playing = False
    
    def toggle_play_pause(self):
        """切换播放/暂停"""
        if self.is_playing:
            self.pause()
        else:
            self.play()
    
    def seek(self, frame_idx: int):
        """跳转到指定帧"""
        self.current_frame = max(0, min(frame_idx, self.total_frames - 1))
        self.last_frame_time = cv2.getTickCount() / cv2.getTickFrequency()
    
    def seek_relative(self, delta: int):
        """相对跳转"""
        self.seek(self.current_frame + delta)
    
    def set_speed(self, speed: float):
        """设置播放速度"""
        self.playback_speed = max(0.1, min(speed, 10.0))
    
    def update(self) -> Optional[Dict[str, Any]]:
        """更新播放状态，返回需要显示的帧"""
        if not self.is_playing:
            return None
        
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        elapsed = current_time - self.last_frame_time
        
        frame_interval = (1.0 / self.target_fps) / self.playback_speed
        
        if elapsed >= frame_interval:
            self.last_frame_time = current_time
            
            frame_data = self.get_frame(self.current_frame)
            self.current_frame += 1
            
            if self.current_frame >= self.total_frames:
                self.pause()
                self.current_frame = self.total_frames - 1
            
            return frame_data
        
        return None


class TrimmerWindow:
    """裁剪窗口"""
    
    def __init__(self, episode_path: str):
        """初始化窗口"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.trimmer = EpisodeTrimmer(episode_path)
        
        # 显示设置
        self.image_display_size = (960, 720)
        self.plot_size = (800, 400)
        self.plot_history = 300
        
        # 数据缓冲
        self.force_buffer = deque(maxlen=self.plot_history)
        self.torque_buffer = deque(maxlen=self.plot_history)
        
        # UI状态
        self.running = True
        self.show_help = True
        self.skip_current = False
        
        # 初始化窗口
        self._init_windows()
        
        # 显示第一帧
        self._display_frame(self.trimmer.get_frame(0))
    
    def _init_windows(self):
        """初始化OpenCV窗口"""
        cv2.namedWindow('Trimmer - Image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Trimmer - Image', self.image_display_size[0], self.image_display_size[1])
        
        # 可选：力传感器窗口
        if self.trimmer.forces is not None:
            cv2.namedWindow('Trimmer - Force', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Trimmer - Force', self.plot_size[0], self.plot_size[1])
    
    def _display_frame(self, frame_data: Dict[str, Any]):
        """显示一帧数据"""
        # 更新力数据缓冲
        if frame_data.get('force') is not None:
            force = frame_data['force']
            self.force_buffer.append(force[:3])
            self.torque_buffer.append(force[3:])
        
        # 显示图像
        self._update_image_display(frame_data)
        
        # 显示力数据
        if self.trimmer.forces is not None:
            self._update_force_plot()
    
    def _update_image_display(self, frame_data: Dict[str, Any]):
        """更新图像显示"""
        image = frame_data.get('image')
        
        if image is not None:
            if len(image.shape) == 3 and image.shape[2] == 3:
                display_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                display_image = image.copy()
            display_image = cv2.resize(display_image, self.image_display_size)
        else:
            display_image = np.zeros((self.image_display_size[1], self.image_display_size[0], 3), dtype=np.uint8)
            cv2.putText(display_image, "No Image Data", (50, display_image.shape[0]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        
        # 添加覆盖层
        self._add_overlay(display_image, frame_data)
        
        cv2.imshow('Trimmer - Image', display_image)
    
    def _add_overlay(self, image: np.ndarray, frame_data: Dict[str, Any]):
        """添加信息覆盖层"""
        frame_idx = frame_data['frame_idx']
        
        # 状态文字
        status = "PLAYING" if self.trimmer.is_playing else "PAUSED"
        status_color = (0, 255, 0) if self.trimmer.is_playing else (0, 165, 255)
        
        # 顶部覆盖层
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (image.shape[1], 150), (0, 0, 0), -1)
        image[:] = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
        
        # 帧信息
        cv2.putText(image, f"Frame: {frame_idx + 1}/{self.trimmer.total_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(image, f"Original: {self.trimmer.original_frames} frames", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(image, f"Speed: {self.trimmer.playback_speed:.1f}x", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 时间戳
        if frame_data.get('timestamp') is not None:
            cv2.putText(image, f"Time: {frame_data['timestamp']:.3f}s", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 状态指示
        cv2.putText(image, status, (image.shape[1] - 150, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
        
        # 裁剪状态
        if self.trimmer.trim_frame is not None:
            trim_text = f"TRIM @ Frame {self.trimmer.trim_frame + 1}"
            cv2.putText(image, trim_text, (image.shape[1] - 280, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image, "Press S to save", (image.shape[1] - 200, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
        
        # 修改标志
        if self.trimmer.modified:
            cv2.putText(image, "[MODIFIED]", (image.shape[1] - 150, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        
        # 进度条
        progress_y = 148
        progress_width = image.shape[1] - 20
        progress = frame_idx / max(1, self.trimmer.original_frames - 1)
        
        # 背景
        cv2.rectangle(image, (10, progress_y), (10 + progress_width, progress_y + 8), (50, 50, 50), -1)
        
        # 有效区域（如果有裁剪点）
        if self.trimmer.trim_frame is not None:
            trim_progress = (self.trimmer.trim_frame + 1) / self.trimmer.original_frames
            cv2.rectangle(image, (10, progress_y), 
                         (10 + int(progress_width * trim_progress), progress_y + 8), (0, 100, 0), -1)
        
        # 当前位置
        cv2.rectangle(image, (10, progress_y), 
                     (10 + int(progress_width * progress), progress_y + 8), (0, 255, 0), -1)
        
        # 裁剪点标记
        if self.trimmer.trim_frame is not None:
            trim_x = 10 + int(progress_width * (self.trimmer.trim_frame / max(1, self.trimmer.original_frames - 1)))
            cv2.line(image, (trim_x, progress_y - 5), (trim_x, progress_y + 13), (0, 0, 255), 3)
        
        # 帮助信息
        if self.show_help:
            help_y = image.shape[0] - 10
            help_texts = [
                "Space: Play/Pause  |  Left/Right: Step  |  Up/Down: Speed",
                "T/C: Trim at current frame  |  S: Save  |  R: Reset  |  N: Next  |  Q: Quit"
            ]
            for i, text in enumerate(reversed(help_texts)):
                y_pos = help_y - i * 25
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(image, (5, y_pos - text_height - 5), (text_width + 10, y_pos + 5), (0, 0, 0), -1)
                cv2.putText(image, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _update_force_plot(self):
        """更新力传感器图表"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
        fig.patch.set_facecolor('#1a1a1a')
        
        for ax in [ax1, ax2]:
            ax.set_facecolor('#2d2d2d')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        if len(self.force_buffer) > 0:
            forces = np.array(self.force_buffer)
            torques = np.array(self.torque_buffer)
            
            ax1.plot(forces[:, 0], 'r-', label='Fx', linewidth=1.5)
            ax1.plot(forces[:, 1], 'g-', label='Fy', linewidth=1.5)
            ax1.plot(forces[:, 2], 'b-', label='Fz', linewidth=1.5)
            ax1.set_ylabel('Force (N)', fontsize=10, color='white')
            ax1.legend(loc='upper right', fontsize=8, facecolor='#2d2d2d', labelcolor='white')
            ax1.grid(True, alpha=0.3)
            ax1.set_title('Force', fontsize=11, color='white')
            
            ax2.plot(torques[:, 0], 'r-', label='Mx', linewidth=1.5)
            ax2.plot(torques[:, 1], 'g-', label='My', linewidth=1.5)
            ax2.plot(torques[:, 2], 'b-', label='Mz', linewidth=1.5)
            ax2.set_ylabel('Torque (Nm)', fontsize=10, color='white')
            ax2.set_xlabel('Frame', fontsize=10, color='white')
            ax2.legend(loc='upper right', fontsize=8, facecolor='#2d2d2d', labelcolor='white')
            ax2.grid(True, alpha=0.3)
            ax2.set_title('Torque', fontsize=11, color='white')
        
        plt.tight_layout()
        
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        plot_img = np.asarray(buf)
        plot_img = cv2.cvtColor(plot_img[:, :, :3], cv2.COLOR_RGB2BGR)
        
        plt.close(fig)
        
        cv2.imshow('Trimmer - Force', plot_img)
    
    def _handle_key(self, key: int) -> bool:
        """处理键盘输入，返回是否继续运行"""
        if key == -1:
            return True
        
        # 空格: 播放/暂停
        if key == ord(' '):
            self.trimmer.toggle_play_pause()
        
        # 左方向键: 后退一帧
        elif key == 81 or key == 2 or key == ord('a'):
            self.trimmer.pause()
            self.trimmer.seek_relative(-1)
            self._display_frame(self.trimmer.get_frame())
        
        # 右方向键: 前进一帧
        elif key == 83 or key == 3 or key == ord('d'):
            self.trimmer.pause()
            self.trimmer.seek_relative(1)
            self._display_frame(self.trimmer.get_frame())
        
        # 上方向键: 加速
        elif key == 82 or key == 0 or key == ord('w'):
            self.trimmer.set_speed(self.trimmer.playback_speed + 0.5)
            self.logger.info(f"速度: {self.trimmer.playback_speed}x")
        
        # 下方向键: 减速
        elif key == 84 or key == 1 or key == ord('s') and (cv2.getWindowProperty('Trimmer - Image', cv2.WND_PROP_VISIBLE) < 1 or False):
            self.trimmer.set_speed(self.trimmer.playback_speed - 0.5)
            self.logger.info(f"速度: {self.trimmer.playback_speed}x")
        
        # T/C: 设置裁剪点
        elif key == ord('t') or key == ord('c'):
            self.trimmer.pause()
            self.trimmer.set_trim_point(self.trimmer.current_frame)
            self._display_frame(self.trimmer.get_frame())
            self.logger.info(f"裁剪点设置在帧 {self.trimmer.current_frame + 1}")
        
        # S: 保存
        elif key == ord('s'):
            if self.trimmer.trim_frame is not None:
                self.trimmer.pause()
                self.logger.info("正在保存...")
                if self.trimmer.save():
                    self.logger.info("保存成功!")
                    # 保存成功后不再标记为modified
                else:
                    self.logger.error("保存失败!")
            else:
                self.logger.warning("请先设置裁剪点 (按 T 或 C)")
        
        # R: 重置裁剪点
        elif key == ord('r'):
            self.trimmer.reset_trim()
            self.force_buffer.clear()
            self.torque_buffer.clear()
            self.trimmer.seek(0)
            self._display_frame(self.trimmer.get_frame())
        
        # N: 跳过当前，处理下一个
        elif key == ord('n'):
            self.skip_current = True
            self.running = False
        
        # H: 切换帮助显示
        elif key == ord('h'):
            self.show_help = not self.show_help
            self._display_frame(self.trimmer.get_frame())
        
        # Home: 跳到开头
        elif key == 80:
            self.trimmer.seek(0)
            self.force_buffer.clear()
            self.torque_buffer.clear()
            self._display_frame(self.trimmer.get_frame())
        
        # End: 跳到结尾
        elif key == 87:
            self.trimmer.seek(self.trimmer.total_frames - 1)
            self._display_frame(self.trimmer.get_frame())
        
        # Q/ESC: 退出
        elif key == ord('q') or key == 27:
            if self.trimmer.modified:
                self.logger.warning("有未保存的修改! 按 S 保存或再按 Q 强制退出")
                # 再给一次机会
                cv2.waitKey(100)
            self.running = False
            return False
        
        return True
    
    def run(self) -> bool:
        """
        运行裁剪窗口
        
        Returns:
            是否跳过当前（True=跳过，False=退出）
        """
        self.logger.info("开始裁剪...")
        self.logger.info(f"Episode: {self.trimmer.episode_path}")
        self.logger.info(f"帧数: {self.trimmer.total_frames}")
        
        self.trimmer.play()
        
        while self.running:
            frame_data = self.trimmer.update()
            if frame_data is not None:
                self._display_frame(frame_data)
            
            key = cv2.waitKey(1) & 0xFF
            if not self._handle_key(key):
                break
        
        cv2.destroyAllWindows()
        
        return self.skip_current


def find_episodes(path: str) -> List[Path]:
    """查找所有episode文件"""
    path = Path(path)
    
    if path.is_file() and path.suffix == '.hdf5':
        return [path]
    
    if path.is_dir():
        episodes = sorted(path.glob('episode*.hdf5'))
        return episodes
    
    return []


def main():
    """主函数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n用法:")
        print("  python trim_episode.py <episode.hdf5>")
        print("  python trim_episode.py <session_directory>")
        print("\n示例:")
        print("  python trim_episode.py data/session_20251202_170443/episode0.hdf5")
        print("  python trim_episode.py data/session_20251202_170443/")
        print("\n注意: 保存时会直接覆盖原文件")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # 查找所有episode
    episodes = find_episodes(input_path)
    
    if not episodes:
        logger.error(f"未找到episode文件: {input_path}")
        sys.exit(1)
    
    logger.info(f"找到 {len(episodes)} 个episode文件")
    
    # 打印操作说明
    print("\n" + "=" * 60)
    print("                   Episode 裁剪工具")
    print("=" * 60)
    print("\n键盘控制:")
    print("  空格          播放/暂停")
    print("  左/右方向键   逐帧后退/前进")
    print("  上/下方向键   加速/减速播放")
    print("  T 或 C        在当前帧设置裁剪点 (删除之后的数据)")
    print("  S             保存裁剪后的数据 (直接覆盖原文件)")
    print("  R             重置裁剪点")
    print("  N             跳过当前episode，处理下一个")
    print("  H             显示/隐藏帮助")
    print("  Q 或 ESC      退出")
    print("\n" + "=" * 60 + "\n")
    
    # 处理每个episode
    for i, episode_path in enumerate(episodes):
        logger.info("=" * 60)
        logger.info(f"处理 [{i+1}/{len(episodes)}]: {episode_path.name}")
        
        try:
            window = TrimmerWindow(str(episode_path))
            skip = window.run()
            
            if not skip and i < len(episodes) - 1:
                # 用户退出，不继续处理
                break
                
        except Exception as e:
            logger.error(f"处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info("裁剪完成!")


if __name__ == '__main__':
    main()

