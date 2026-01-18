# ForceUMI 数据采集系统

---

## 安装

### 从源码安装

```bash
conda create -n forceumi python=3.10
conda activate forceumi
git clone https://github.com/arctic126/ForceUMI.git
cd ForceUMI
pip install -e .
```

---

### 硬件支持

#### 基于 VR Tracker 的位姿感知

```bash
git clone https://github.com/arctic126/PyTracker.git
cd PyTracker
pip install -e .
cd ..
```

**说明：**

* 需要安装 **SteamVR**
* 需要兼容的 VR 硬件（如 Vive Tracker 等）

---

#### Sunrise（宇立）六维力/力矩传感器（使用 PyForce）

```bash
git clone https://github.com/arctic126/PyForce.git
cd PyForce
pip install -e .
cd ..
```

**说明：**

* 需要 Sunrise（宇立）六轴力/力矩传感器
* 通过 TCP/IP 与主机通信

---

## 快速开始

### 1. 启动 GUI 数据采集程序

```bash
python -m forceumi.gui.cv_main_window
```

或使用示例启动脚本：

```bash
python examples/launch_gui.py
```

#### 键盘快捷键说明

* `C`：连接设备
* `D`：断开设备
* `S`：开始采集
* `E`：停止并保存当前 episode
* `Q`：退出程序

---

### 2. 回放已采集的数据

可对已采集的 episode 进行**同步回放与可视化**：

```bash
python examples/replay_episode.py data/episode_xx.hdf5
```

---

### 3. 转换为 LeRobot 数据格式

可将 ForceUMI 采集的数据转换为 **LeRobot** 所使用的数据格式，便于VLA训练。

#### 安装 LeRobot（建议使用单独环境）

```bash
conda create -n lerobot python=3.10
conda activate lerobot
pip install lerobot
```

#### 转换一个采集会话

```bash
python convert_forceumi_to_lerobot.py \
  --data_dir data/session_20250118_143000 \
  --output_repo_id username/forceumi-task1 \
  --task "clean the basin" \
  --target_size 224 224
```

参数说明：

* `--data_dir`：ForceUMI 采集的 session 目录
* `--output_repo_id`：LeRobot 数据集仓库 ID
* `--task`：任务描述
* `--target_size`：图像 resize 尺寸

---


## 数据格式说明

每个 episode 以一个 **HDF5 文件**保存，结构如下：

### 单相机数据格式

```
episode0.hdf5
├── /image              # (N, H, W, 3), uint8
├── /state              # (N, 7), float32
├── /action             # (N, 7), float32
├── /force              # (N, 6), float32
├── /timestamp          # (N,), float64（主循环时间戳）
├── /timestamp_camera   # (N,), float64（相机时间戳）
├── /timestamp_pose     # (N,), float64（位姿时间戳）
├── /timestamp_force    # (N,), float64（力传感器时间戳）
└── /metadata           # 属性：fps、duration、task_description 等
```

---

### 字段定义说明

#### state（状态）

Tracker 相对于**工作站（基坐标系）**的位姿：

```
[x, y, z, rx, ry, rz, gripper]
```

* 位置 `(x, y, z)`：单位为 **米**
* 姿态 `(rx, ry, rz)`：欧拉角，单位为 **弧度**
* `gripper`：夹爪开合状态

  * `0.0` = 完全闭合
  * `1.0` = 完全张开
  * 始终为**绝对值**

---

#### action（动作）

Tracker 相对于**第一帧坐标系**的位姿变化：

```
[x, y, z, rx, ry, rz, gripper]
```

* 第一帧的 action 固定为：

  ```
  [0, 0, 0, 0, 0, 0, gripper]
  ```
* 后续帧表示相对于第一帧的位姿变化
* 位置与姿态均为**相对量**
* `gripper` 始终为绝对值（与 state 相同）

---

#### force（力/力矩）

六维力/力矩传感器数据：

```
[fx, fy, fz, mx, my, mz]
```

* 力 `(fx, fy, fz)`：单位 **牛顿（N）**
* 力矩 `(mx, my, mz)`：单位 **牛·米（Nm）**

---

## 项目结构

```
forceumi/
├── forceumi/              # 主程序包
│   ├── devices/           # 设备接口
│   │   ├── camera.py
│   │   ├── pose_sensor.py
│   │   └── force_sensor.py
│   ├── data/              # 数据管理
│   │   ├── hdf5_manager.py
│   │   └── episode.py
│   ├── gui/               # OpenCV 图形界面
│   │   ├── cv_main_window.py
│   │   └── cv_visualizer.py
│   ├── replay/            # 数据回放模块
│   │   ├── player.py
│   │   └── replay_window.py
│   ├── utils/             # 工具函数
│   │   └── transforms.py  # 坐标变换
│   ├── collector.py       # 数据采集管理器
│   └── config.py          # 配置管理
├── examples/              # 示例脚本
├── tests/                 # 测试代码
├── requirements.txt
├── setup.py
└── README.md
```

---
