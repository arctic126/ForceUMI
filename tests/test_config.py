"""
Tests for configuration management
"""

import pytest
import tempfile
from pathlib import Path

from forceumi.config import Config


class TestConfig:
    """Test Config manager"""
    
    def test_init(self):
        """Test configuration initialization"""
        config = Config()
        assert config.config is not None
        assert "devices" in config.config
        assert "data" in config.config
        assert "collector" in config.config
        assert "gui" in config.config
    
    def test_get(self):
        """Test getting configuration values"""
        config = Config()
        
        # Get nested value (multi-camera format)
        cameras = config.get("devices.cameras")
        assert cameras is not None
        assert isinstance(cameras, list)
        assert len(cameras) >= 1
        assert cameras[0]["width"] == 640
        
        # Get with default
        nonexistent = config.get("nonexistent.key", "default")
        assert nonexistent == "default"
    
    def test_set(self):
        """Test setting configuration values"""
        config = Config()
        
        # Set nested value
        config.set("devices.camera.width", 1280)
        assert config.get("devices.camera.width") == 1280
        
        # Set new value
        config.set("new.nested.value", 42)
        assert config.get("new.nested.value") == 42
    
    def test_save_and_load_yaml(self):
        """Test saving and loading YAML configuration"""
        config = Config()
        config.set("devices.camera.width", 1280)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "config.yaml"
            
            # Save
            success = config.save(str(filepath))
            assert success
            assert filepath.exists()
            
            # Load
            new_config = Config()
            success = new_config.load(str(filepath))
            assert success
            assert new_config.get("devices.camera.width") == 1280
    
    def test_save_and_load_json(self):
        """Test saving and loading JSON configuration"""
        config = Config()
        config.set("devices.camera.fps", 60)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "config.json"
            
            # Save
            success = config.save(str(filepath))
            assert success
            assert filepath.exists()
            
            # Load
            new_config = Config()
            success = new_config.load(str(filepath))
            assert success
            assert new_config.get("devices.camera.fps") == 60
    
    def test_to_dict(self):
        """Test converting configuration to dictionary"""
        config = Config()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "devices" in config_dict
        assert "data" in config_dict
    
    def test_multi_camera_config(self):
        """Test multi-camera configuration"""
        import yaml
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multi-camera config
            config_data = {
                "devices": {
                    "cameras": [
                        {"name": "wrist_camera", "device_id": 0, "width": 640, "height": 480, "fps": 30},
                        {"name": "static_camera", "device_id": 2, "width": 640, "height": 480, "fps": 30},
                    ]
                }
            }
            
            filepath = Path(tmpdir) / "config_multicam.yaml"
            with open(filepath, "w") as f:
                yaml.dump(config_data, f)
            
            # Load and verify
            config = Config(str(filepath))
            cameras = config.get("devices.cameras")
            
            assert len(cameras) == 2
            assert cameras[0]["name"] == "wrist_camera"
            assert cameras[1]["name"] == "static_camera"
            assert cameras[0]["device_id"] == 0
            assert cameras[1]["device_id"] == 2
    
    def test_backward_compatibility_single_camera(self):
        """Test backward compatibility with old single camera config"""
        import yaml
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create old format config (single camera as dict)
            config_data = {
                "devices": {
                    "camera": {
                        "device_id": 0,
                        "width": 640,
                        "height": 480,
                        "fps": 30
                    }
                }
            }
            
            filepath = Path(tmpdir) / "config_old.yaml"
            with open(filepath, "w") as f:
                yaml.dump(config_data, f)
            
            # Load and verify it's converted to new format
            config = Config(str(filepath))
            cameras = config.get("devices.cameras")
            
            assert cameras is not None
            assert isinstance(cameras, list)
            assert len(cameras) == 1
            assert cameras[0]["name"] == "camera"
            assert cameras[0]["device_id"] == 0
            
            # Old format should be removed
            assert config.get("devices.camera") is None

