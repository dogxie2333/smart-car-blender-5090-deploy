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

每个 `imgN` 目录会继承 `src` 下面的数据集子目录结构。

每轮随机到的灯光和相机参数会记录在：

```text
round_params_5090.json
```

## 修改训练轮数

打开 `server_render_5090.py`，修改：

```python
TRAINING_ROUNDS = 1
```
