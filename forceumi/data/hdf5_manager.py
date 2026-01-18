"""
HDF5 File Manager

Handles reading and writing episode data to HDF5 files.
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging


class HDF5Manager:
    """Manager for HDF5 file operations"""
    
    def __init__(
        self,
        compression: str = "gzip",
        compression_level: int = 4
    ):
        """
        Initialize HDF5 manager
        
        Args:
            compression: Compression algorithm (gzip, lzf, None)
            compression_level: Compression level (0-9 for gzip)
        """
        self.compression = compression
        self.compression_level = compression_level
        self.logger = logging.getLogger("forceumi.data.HDF5Manager")
    
    def save_episode(
        self,
        filepath: str,
        data: Dict[str, Any],
        overwrite: bool = False
    ) -> bool:
        """
        Save episode data to HDF5 file
        
        Args:
            filepath: Path to save HDF5 file
            data: Dictionary containing episode data
            overwrite: Whether to overwrite existing file
            
        Returns:
            bool: True if save successful
        """
        try:
            filepath = Path(filepath)
            
            # Check if file exists
            if filepath.exists() and not overwrite:
                self.logger.error(f"File already exists: {filepath}")
                return False
            
            # Create parent directory if needed
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to HDF5
            with h5py.File(filepath, "w") as f:
                # Detect if multi-camera or single camera format
                camera_names = data.get("metadata", {}).get("camera_names", [])
                is_multi_camera = len(camera_names) > 1
                
                # Save camera images
                if is_multi_camera:
                    # Multi-camera: save as image_{camera_name}
                    for camera_name in camera_names:
                        img_key = f"images_{camera_name}"
                        if img_key in data and len(data[img_key]) > 0:
                            arr = np.array(data[img_key])
                            f.create_dataset(
                                f"image_{camera_name}",
                                data=arr,
                                compression=self.compression,
                                compression_opts=self.compression_level,
                                chunks=(1, *arr.shape[1:])
                            )
                else:
                    # Single camera: save as "image" for backward compatibility
                    if len(camera_names) == 1:
                        camera_name = camera_names[0]
                        img_key = f"images_{camera_name}"
                        if img_key in data and len(data[img_key]) > 0:
                            arr = np.array(data[img_key])
                            f.create_dataset(
                                "image",
                                data=arr,
                                compression=self.compression,
                                compression_opts=self.compression_level,
                                chunks=(1, *arr.shape[1:])
                            )
                
                # Save other main datasets
                main_keys = ["state", "action", "force", "timestamp"]
                for key in main_keys:
                    if key in data and len(data[key]) > 0:
                        arr = data[key]
                        f.create_dataset(
                            key,
                            data=arr,
                            compression=self.compression,
                            compression_opts=self.compression_level
                        )
                
                # Save per-camera timestamps
                if is_multi_camera:
                    # Multi-camera: save as timestamp_camera_{camera_name}
                    for camera_name in camera_names:
                        ts_key = f"timestamps_camera_{camera_name}"
                        if ts_key in data and len(data[ts_key]) > 0:
                            f.create_dataset(
                                f"timestamp_camera_{camera_name}",
                                data=np.array(data[ts_key]),
                                compression=self.compression,
                                compression_opts=self.compression_level
                            )
                else:
                    # Single camera: save as "timestamp_camera" for backward compatibility
                    if len(camera_names) == 1:
                        camera_name = camera_names[0]
                        ts_key = f"timestamps_camera_{camera_name}"
                        if ts_key in data and len(data[ts_key]) > 0:
                            f.create_dataset(
                                "timestamp_camera",
                                data=np.array(data[ts_key]),
                                compression=self.compression,
                                compression_opts=self.compression_level
                            )
                
                # Save other sensor timestamps (v0.3.1+)
                for ts_key in ["timestamp_pose", "timestamp_force"]:
                    if ts_key in data and len(data[ts_key]) > 0:
                        f.create_dataset(
                            ts_key,
                            data=data[ts_key],
                            compression=self.compression,
                            compression_opts=self.compression_level
                        )
                
                # Save metadata as attributes
                if "metadata" in data:
                    for key, value in data["metadata"].items():
                        # Convert list to string for HDF5 attributes
                        if isinstance(value, list):
                            value = str(value)
                        f.attrs[key] = value
            
            self.logger.info(f"Episode saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save episode: {e}")
            return False
    
    def load_episode(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Load episode data from HDF5 file
        
        Args:
            filepath: Path to HDF5 file
            
        Returns:
            dict: Episode data or None if load failed
        """
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                self.logger.error(f"File not found: {filepath}")
                return None
            
            data = {}
            
            with h5py.File(filepath, "r") as f:
                # Load metadata first to detect format
                metadata = {}
                for key in f.attrs:
                    metadata[key] = f.attrs[key]
                data["metadata"] = metadata
                
                # Detect camera format
                camera_names = []
                
                # Check for multi-camera format (image_{camera_name})
                for key in f.keys():
                    if key.startswith("image_"):
                        camera_name = key[6:]  # Remove "image_" prefix
                        camera_names.append(camera_name)
                
                # Check for legacy single camera format
                if not camera_names and "image" in f:
                    # Legacy format: single camera named "camera"
                    data["image"] = f["image"][:]
                    camera_names = ["camera"]
                
                # Load multi-camera images
                if camera_names and "image" not in data:
                    for camera_name in camera_names:
                        img_key = f"image_{camera_name}"
                        if img_key in f:
                            data[f"images_{camera_name}"] = f[img_key][:]
                    
                    # Store camera names in metadata
                    data["metadata"]["camera_names"] = camera_names
                
                # Load other main datasets
                for key in ["state", "action", "force", "timestamp"]:
                    if key in f:
                        data[key] = f[key][:]
                    else:
                        data[key] = np.array([])
                
                # Load per-camera timestamps
                if len(camera_names) > 1:
                    # Multi-camera format
                    for camera_name in camera_names:
                        ts_key = f"timestamp_camera_{camera_name}"
                        if ts_key in f:
                            data[f"timestamps_camera_{camera_name}"] = f[ts_key][:]
                elif len(camera_names) == 1:
                    # Single camera: check both formats
                    camera_name = camera_names[0]
                    
                    # Try new format first
                    ts_key_new = f"timestamp_camera_{camera_name}"
                    if ts_key_new in f:
                        data[f"timestamps_camera_{camera_name}"] = f[ts_key_new][:]
                    # Fall back to legacy format
                    elif "timestamp_camera" in f:
                        data[f"timestamps_camera_{camera_name}"] = f["timestamp_camera"][:]
                
                # Load other sensor timestamps (v0.3.1+)
                for ts_key in ["timestamp_pose", "timestamp_force"]:
                    if ts_key in f:
                        data[ts_key] = f[ts_key][:]
            
            self.logger.info(f"Episode loaded from {filepath}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load episode: {e}")
            return None
    
    def get_episode_info(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Get episode metadata without loading full data
        
        Args:
            filepath: Path to HDF5 file
            
        Returns:
            dict: Episode metadata or None if load failed
        """
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                self.logger.error(f"File not found: {filepath}")
                return None
            
            info = {}
            
            with h5py.File(filepath, "r") as f:
                # Get metadata
                for key in f.attrs:
                    info[key] = f.attrs[key]
                
                # Detect camera format
                camera_names = []
                for key in f.keys():
                    if key.startswith("image_"):
                        camera_name = key[6:]  # Remove "image_" prefix
                        camera_names.append(camera_name)
                        info[f"{key}_shape"] = f[key].shape
                
                # Legacy single camera format
                if not camera_names and "image" in f:
                    info["image_shape"] = f["image"].shape
                    camera_names = ["camera"]
                
                info["camera_names"] = camera_names
                info["num_cameras"] = len(camera_names)
                
                # Get other dataset shapes
                for key in ["state", "action", "force", "timestamp"]:
                    if key in f:
                        info[f"{key}_shape"] = f[key].shape
                
                # Get timestamp shapes
                for key in f.keys():
                    if key.startswith("timestamp_"):
                        info[f"{key}_shape"] = f[key].shape
            
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to get episode info: {e}")
            return None
    
    @staticmethod
    def generate_filename(prefix: str = "episode", extension: str = ".hdf5") -> str:
        """
        Generate filename with timestamp
        
        Args:
            prefix: Filename prefix
            extension: File extension
            
        Returns:
            str: Generated filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}{extension}"

