# Smart Car Blender 5090 Render

这个仓库用于在 5090 服务器上用 Blender 后台批量生成训练图片。

## 文件说明

1. `main.blend`：Blender 场景文件。
2. `server_render_5090.py`：服务器批量渲染脚本。
3. `src/数据集图片/`：需要替换到图片块上的源贴图数据集。

## 安装 Blender

服务器需要安装 Blender 4.2 LTS 或更高版本，并确保命令行可以执行 `blender`。

## 安装 Pillow

脚本会优先用 Pillow 做高质量缩放和轻微锐化。

```bash
blender -b --python-expr "import sys; print(sys.executable)"
```

用上面输出的 Blender Python 路径安装：

```bash
/path/to/blender/python -m ensurepip
/path/to/blender/python -m pip install -r requirements-blender.txt
```

## 运行

```bash
blender -b main.blend -P server_render_5090.py
```

## 输出

脚本会按训练轮次生成：

```text
img1/
img2/
img3/
```

每个 `imgN` 目录会继承 `DATASET_ROOT` 下面的数据集子目录结构。

最终渲染图片尺寸为 `640x480`，脚本会先按 4 倍超采样渲染，再缩小到最终尺寸。
渲染结果会应用接近真实 `01_original` 拍摄图的室内摄像头色调：中等亮度、低饱和、轻微偏暖。
默认还会为每张 PNG 生成同名 `.yuyv` raw 文件，字节布局为 `Y0 U Y1 V`，方便后续按 YUYV 输入流程继续处理。

当前相机范围按本地 Blender MCP 读取到的 `scene3 / Camera.s3` 好画面校准：位置 X 为 `-0.20~-0.09`，位置 Y 为 `-0.36~-0.20`，位置 Z 固定 `0.31`，欧拉旋转约围绕 `[38, 0, -90]` 度轻微随机。

每轮随机到的灯光和相机参数会记录在：

```text
round_params_5090.json
```

## 修改训练轮数

打开 `server_render_5090.py`，修改：

```python
TRAINING_ROUNDS = 1
```

服务器部署时也可以用环境变量覆盖关键路径和轮数，例如：

```bash
DATASET_ROOT=/root/autodl-tmp/smart-car-data/走马观碑数据集all \
OUTPUT_ROOT=/root/autodl-tmp/smart-car-output \
ARCHIVE_ROOT=/root/autodl-tmp/smart-car-archives \
SELECTED_DIR_INDICES=all \
TRAINING_ROUNDS=1 \
blender -b main.blend -P server_render_5090.py
```

冒烟测试时可以加 `MAX_IMAGES_PER_DIR=1`，每个目录只渲染 1 张。

## 提速参数

默认速度优先：`SUPERSAMPLE_SCALE=2`、`RENDER_SAMPLES=16`。

大批量训练集建议使用下面的组合，仍然输出 `640x480` 和同名 `.yuyv`：

```bash
SUPERSAMPLE_SCALE=2 \
RENDER_SAMPLES=16 \
ENABLE_ARCHIVE=0 \
DATASET_ROOT=/root/autodl-tmp/smart-car-data/走马观碑数据集all \
OUTPUT_ROOT=/root/autodl-tmp/smart-car-output \
SELECTED_DIR_INDICES=all \
TRAINING_ROUNDS=50 \
blender -b main.blend -P server_render_5090.py
```

如果只想快速看训练链路，可以再加 `SAVE_YUYV_OUTPUT=0`，但正式训练 YUYV 输入流程建议保持开启。

如果需要更高画质，可以临时改为 `SUPERSAMPLE_SCALE=4`、`RENDER_SAMPLES=64`，代价是单张渲染时间明显增加。
