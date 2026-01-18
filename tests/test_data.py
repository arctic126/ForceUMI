"""
Tests for data management modules
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from forceumi.data import Episode, HDF5Manager


class TestEpisode:
    """Test Episode data container"""
    
    def test_init(self):
        """Test episode initialization"""
        episode = Episode()
        assert len(episode) == 0
        assert episode.start_time is not None
        assert episode.end_time is None
    
    def test_add_frame_single_camera_legacy(self):
        """Test adding frames with single camera (legacy API)"""
        episode = Episode()
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        state = np.random.randn(7).astype(np.float32)
        action = np.random.randn(7).astype(np.float32)
        force = np.random.randn(6).astype(np.float32)
        
        # Use legacy single image parameter
        episode.add_frame(image=image, state=state, action=action, force=force,
                         timestamp_camera=1.0, timestamp_pose=1.01, timestamp_force=1.02)
        
        assert len(episode) == 1
        assert "camera" in episode.images
        assert len(episode.images["camera"]) == 1
        assert len(episode.states) == 1
        assert len(episode.actions) == 1
        assert len(episode.forces) == 1
        assert len(episode.timestamps) == 1
        assert "camera" in episode.timestamps_cameras
        assert len(episode.timestamps_cameras["camera"]) == 1
        assert len(episode.timestamps_pose) == 1
        assert len(episode.timestamps_force) == 1
    
    def test_add_frame_multi_camera(self):
        """Test adding frames with multiple cameras"""
        episode = Episode()
        
        images = {
            "wrist_camera": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "static_camera": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        }
        timestamps_cameras = {
            "wrist_camera": 1.0,
            "static_camera": 1.001,
        }
        state = np.random.randn(7).astype(np.float32)
        action = np.random.randn(7).astype(np.float32)
        force = np.random.randn(6).astype(np.float32)
        
        episode.add_frame(
            images=images, 
            state=state, 
            action=action, 
            force=force,
            timestamps_cameras=timestamps_cameras,
            timestamp_pose=1.01, 
            timestamp_force=1.02
        )
        
        assert len(episode) == 1
        assert "wrist_camera" in episode.images
        assert "static_camera" in episode.images
        assert len(episode.images["wrist_camera"]) == 1
        assert len(episode.images["static_camera"]) == 1
        assert len(episode.timestamps_cameras["wrist_camera"]) == 1
        assert len(episode.timestamps_cameras["static_camera"]) == 1
    
    def test_finalize(self):
        """Test episode finalization"""
        episode = Episode()
        
        for _ in range(10):
            episode.add_frame(
                image=np.zeros((480, 640, 3), dtype=np.uint8),
                state=np.zeros(7, dtype=np.float32),
                action=np.zeros(7, dtype=np.float32),
                force=np.zeros(6, dtype=np.float32),
            )
        
        episode.finalize()
        
        assert episode.end_time is not None
        assert "duration" in episode.metadata
        assert "num_frames" in episode.metadata
        assert "fps" in episode.metadata
        assert episode.metadata["num_frames"] == 10
    
    def test_to_dict_multi_camera(self):
        """Test conversion to dictionary with multi-camera"""
        episode = Episode()
        
        images = {
            "wrist_camera": np.zeros((480, 640, 3), dtype=np.uint8),
            "static_camera": np.zeros((480, 640, 3), dtype=np.uint8),
        }
        
        episode.add_frame(
            images=images,
            state=np.zeros(7, dtype=np.float32),
            action=np.zeros(7, dtype=np.float32),
            force=np.zeros(6, dtype=np.float32),
        )
        
        data = episode.to_dict()
        
        assert "images_wrist_camera" in data
        assert "images_static_camera" in data
        assert "timestamps_camera_wrist_camera" in data
        assert "timestamps_camera_static_camera" in data
        assert "state" in data
        assert "action" in data
        assert "force" in data
        assert "timestamp" in data
        assert "metadata" in data
        assert "camera_names" in data["metadata"]
        assert "wrist_camera" in data["metadata"]["camera_names"]
        assert "static_camera" in data["metadata"]["camera_names"]


class TestHDF5Manager:
    """Test HDF5 file manager"""
    
    def test_init(self):
        """Test manager initialization"""
        manager = HDF5Manager()
        assert manager.compression == "gzip"
        assert manager.compression_level == 4
    
    def test_save_and_load(self):
        """Test saving and loading episodes"""
        manager = HDF5Manager()
        
        # Create test data
        data = {
            "image": np.random.randint(0, 255, (10, 480, 640, 3), dtype=np.uint8),
            "state": np.random.randn(10, 7).astype(np.float32),
            "action": np.random.randn(10, 7).astype(np.float32),
            "force": np.random.randn(10, 6).astype(np.float32),
            "timestamp": np.arange(10, dtype=np.float64),
            "metadata": {
                "task": "test",
                "duration": 1.0,
                "fps": 10.0,
            }
        }
        
        # Save to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_episode.hdf5"
            
            success = manager.save_episode(str(filepath), data)
            assert success
            assert filepath.exists()
            
            # Load and verify
            loaded_data = manager.load_episode(str(filepath))
            assert loaded_data is not None
            
            assert np.array_equal(loaded_data["image"], data["image"])
            assert np.allclose(loaded_data["state"], data["state"])
            assert np.allclose(loaded_data["action"], data["action"])
            assert np.allclose(loaded_data["force"], data["force"])
            assert np.allclose(loaded_data["timestamp"], data["timestamp"])
            assert loaded_data["metadata"]["task"] == "test"
    
    def test_generate_filename(self):
        """Test filename generation"""
        filename = HDF5Manager.generate_filename()
        assert filename.startswith("episode_")
        assert filename.endswith(".hdf5")
        
        filename = HDF5Manager.generate_filename(prefix="test", extension=".h5")
        assert filename.startswith("test_")
        assert filename.endswith(".h5")
    
    def test_save_and_load_multi_camera(self):
        """Test saving and loading multi-camera episodes"""
        manager = HDF5Manager()
        
        # Create test data with multiple cameras
        data = {
            "images_wrist_camera": [
                np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)
            ],
            "images_static_camera": [
                np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)
            ],
            "timestamps_camera_wrist_camera": [i * 0.033 for i in range(10)],
            "timestamps_camera_static_camera": [i * 0.033 + 0.001 for i in range(10)],
            "state": np.random.randn(10, 7).astype(np.float32),
            "action": np.random.randn(10, 7).astype(np.float32),
            "force": np.random.randn(10, 6).astype(np.float32),
            "timestamp": np.arange(10, dtype=np.float64) * 0.033,
            "metadata": {
                "task": "test_multi_camera",
                "duration": 1.0,
                "fps": 30.0,
                "camera_names": ["wrist_camera", "static_camera"],
            }
        }
        
        # Save to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_multicam_episode.hdf5"
            
            success = manager.save_episode(str(filepath), data)
            assert success
            assert filepath.exists()
            
            # Load and verify
            loaded_data = manager.load_episode(str(filepath))
            assert loaded_data is not None
            
            # Check multi-camera images
            assert "images_wrist_camera" in loaded_data
            assert "images_static_camera" in loaded_data
            assert np.array_equal(loaded_data["images_wrist_camera"], np.array(data["images_wrist_camera"]))
            assert np.array_equal(loaded_data["images_static_camera"], np.array(data["images_static_camera"]))
            
            # Check timestamps
            assert "timestamps_camera_wrist_camera" in loaded_data
            assert "timestamps_camera_static_camera" in loaded_data
            
            # Check metadata
            assert "camera_names" in loaded_data["metadata"]
    
    def test_single_camera_backward_compatibility(self):
        """Test backward compatibility with single camera format"""
        manager = HDF5Manager()
        
        # Create single camera data (should save as 'image' not 'image_camera')
        data = {
            "images_camera": [
                np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)
            ],
            "timestamps_camera_camera": [i * 0.033 for i in range(10)],
            "state": np.random.randn(10, 7).astype(np.float32),
            "action": np.random.randn(10, 7).astype(np.float32),
            "force": np.random.randn(10, 6).astype(np.float32),
            "timestamp": np.arange(10, dtype=np.float64) * 0.033,
            "metadata": {
                "task": "test_single_camera",
                "camera_names": ["camera"],  # Single camera
            }
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_single_cam.hdf5"
            
            # Save
            success = manager.save_episode(str(filepath), data)
            assert success
            
            # Load and verify it uses legacy format names
            loaded_data = manager.load_episode(str(filepath))
            assert loaded_data is not None
            
            # Should be loaded as legacy format
            assert "image" in loaded_data or "images_camera" in loaded_data

