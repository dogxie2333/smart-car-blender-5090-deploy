import json
import os
import random
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

import bpy


PROJECT_ROOT = Path(bpy.data.filepath).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", SRC_ROOT / "数据集图片")).expanduser()
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", PROJECT_ROOT)).expanduser()
PARAMS_PATH = PROJECT_ROOT / "round_params_5090.json"
ARCHIVE_ROOT = Path(os.environ.get("ARCHIVE_ROOT", "/home/blender_img_archives")).expanduser()
ROUND_PARAM_VERSION = 2
ENABLE_ARCHIVE = os.environ.get("ENABLE_ARCHIVE", "1").strip().lower() not in {"0", "false", "no"}

SCENE_NAME = "scene3"
CAMERA_NAME = "Camera.s3"
FRONT_LIGHT_NAME = "前方补光灯"
TOP_LIGHT_NAME = "顶部补光灯"
TARGET_OBJECT_CANDIDATES = ("图片块", "广告图-上方")
TARGET_MATERIAL_CANDIDATES = ("广告图-上方-纯图片", "material-cache-loong21")

# 宏定义：训练轮数和要渲染的数据集编号。
TRAINING_ROUNDS = int(os.environ.get("TRAINING_ROUNDS", "100"))
MAX_IMAGES_PER_DIR = int(os.environ.get("MAX_IMAGES_PER_DIR", "0"))
SELECTED_DIR_INDICES = (11, 13, 4, 5, 7, 10)

# 宏定义：最终摄像头分辨率和超采样倍率。
FINAL_RESOLUTION_X = 640
FINAL_RESOLUTION_Y = 480
SUPERSAMPLE_SCALE = 4

# 宏定义：贴图与渲染清晰度设置。
TEXTURE_INTERPOLATION = "Cubic"
CYCLES_FILTER_WIDTH = 0.5
RENDER_SAMPLES = 64
SAVE_YUYV_OUTPUT = True
USE_POST_SHARPEN = True
UNSHARP_RADIUS = 0.7
UNSHARP_PERCENT = 120
UNSHARP_THRESHOLD = 2

# 宏定义：匹配真实摄像头原图的室内色调。
APPLY_CAMERA_TONE = True
CAMERA_TONE_TARGET_LUMA = 134.0
CAMERA_TONE_BRIGHTNESS_MIN = 0.82
CAMERA_TONE_BRIGHTNESS_MAX = 1.18
CAMERA_TONE_CONTRAST = 0.92
CAMERA_TONE_SATURATION = 0.82
CAMERA_TONE_RED_GAIN = 1.02
CAMERA_TONE_GREEN_GAIN = 1.00
CAMERA_TONE_BLUE_GAIN = 0.96

# 宏定义：随机范围。
FRONT_LIGHT_ENERGY_RANGE = (5.0, 10.0)
TOP_LIGHT_ENERGY_RANGE = (8.0, 8.0)
CAMERA_LOCATION_X_RANGE = (-0.20, -0.09)
CAMERA_LOCATION_Y_RANGE = (-0.36, -0.20)
CAMERA_LOCATION_Z = 0.31
CAMERA_ROTATION_X_RANGE = (36.0, 40.0)
CAMERA_ROTATION_Y_RANGE = (-2.0, 2.0)
CAMERA_ROTATION_Z_RANGE = (-92.0, -88.0)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def find_target_object() -> bpy.types.Object:
    """
    @brief 查找需要替换贴图的目标图片块对象。
    @return 目标网格对象。
    """
    for object_name in TARGET_OBJECT_CANDIDATES:
        target_obj = bpy.data.objects.get(object_name)
        if target_obj is not None:
            return target_obj

    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue

        material_names = [
            material_slot.material.name
            for material_slot in obj.material_slots
            if material_slot.material is not None
        ]
        if any(material_name in TARGET_MATERIAL_CANDIDATES for material_name in material_names):
            return obj

    raise ValueError("未找到目标图片块对象，请确认对象名或材质名是否正确。")


def find_image_texture_node(target_obj: bpy.types.Object) -> bpy.types.ShaderNodeTexImage:
    """
    @brief 在目标对象材质中查找图像纹理节点。
    @param target_obj 目标网格对象。
    @return 图像纹理节点。
    """
    for material_slot in target_obj.material_slots:
        material = material_slot.material
        if material is None or not material.use_nodes or material.node_tree is None:
            continue

        for node in material.node_tree.nodes:
            if node.type == "TEX_IMAGE":
                return node

    raise ValueError(f"对象“{target_obj.name}”上没有可用的图像纹理节点。")


def list_leaf_source_dirs(root_path: Path) -> list[Path]:
    """
    @brief 列出数据集下含图片且无下级目录的具体子类目录。
    @param root_path 数据集根目录。
    @return 排序后的叶子目录列表。
    """
    leaf_dirs: list[Path] = []

    for directory in root_path.rglob("*"):
        if not directory.is_dir():
            continue

        child_dirs = [child for child in directory.iterdir() if child.is_dir()]
        image_files = [
            child
            for child in directory.iterdir()
            if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if image_files and not child_dirs:
            leaf_dirs.append(directory)

    leaf_dirs.sort()
    return leaf_dirs


def get_selected_dir_indices() -> tuple[int, ...] | None:
    """
    @brief 读取本次要渲染的数据集目录编号。
    @return None 表示渲染全部叶子目录，否则返回 1 基目录编号元组。
    """
    raw_indices = os.environ.get("SELECTED_DIR_INDICES")
    if raw_indices is None:
        return SELECTED_DIR_INDICES

    raw_indices = raw_indices.strip()
    if raw_indices == "" or raw_indices.lower() == "all":
        return None

    selected_indices: list[int] = []
    for raw_index in raw_indices.split(","):
        raw_index = raw_index.strip()
        if raw_index == "":
            continue
        selected_indices.append(int(raw_index))

    return tuple(selected_indices)


def select_source_dirs() -> list[Path]:
    """
    @brief 根据 SELECTED_DIR_INDICES 选择要渲染的六个具体子类目录。
    @return 选中的源目录列表。
    """
    if not DATASET_ROOT.exists():
        raise ValueError(f"未找到数据集目录：{DATASET_ROOT}")

    leaf_dirs = list_leaf_source_dirs(DATASET_ROOT)
    selected_dir_indices = get_selected_dir_indices()

    if selected_dir_indices is None:
        print("本次选择全部叶子目录：")
        for directory_index, source_dir in enumerate(leaf_dirs, start=1):
            print(f"{directory_index}. {source_dir.relative_to(DATASET_ROOT)}")
        return leaf_dirs

    selected_dirs = []

    for directory_index in selected_dir_indices:
        if directory_index < 1 or directory_index > len(leaf_dirs):
            raise ValueError(f"目录编号超出范围：{directory_index}")
        selected_dirs.append(leaf_dirs[directory_index - 1])

    print("本次选择目录：")
    for directory_index, source_dir in zip(selected_dir_indices, selected_dirs):
        print(f"{directory_index}. {source_dir.relative_to(DATASET_ROOT)}")

    return selected_dirs


def collect_images_in_dir(source_dir: Path) -> list[Path]:
    """
    @brief 收集指定目录下的所有图片。
    @param source_dir 当前源目录。
    @return 排序后的图片路径列表。
    """
    image_paths = [
        image_path
        for image_path in source_dir.iterdir()
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    image_paths.sort()
    if MAX_IMAGES_PER_DIR > 0:
        return image_paths[:MAX_IMAGES_PER_DIR]
    return image_paths


def get_round_output_root(round_index: int) -> Path:
    """
    @brief 获取当前轮次的输出根目录。
    @param round_index 当前训练轮次，从 1 开始。
    @return 当前轮次输出目录，例如 img1、img2。
    """
    return OUTPUT_ROOT / f"img{round_index}"


def build_output_path(source_image_path: Path, round_index: int) -> Path:
    """
    @brief 构造当前轮次输出路径，继承 src 子目录结构，并保持原文件名。
    @param source_image_path 当前源图片路径。
    @param round_index 当前训练轮次，从 1 开始。
    @return 输出 PNG 路径。
    """
    relative_path = source_image_path.relative_to(DATASET_ROOT)
    output_path = get_round_output_root(round_index) / relative_path.parent / f"{source_image_path.stem}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def build_yuyv_output_path(output_path: Path) -> Path:
    """
    @brief 根据 PNG 输出路径构造同名 YUYV422 raw 文件路径。
    @param output_path 当前渲染输出 PNG 路径。
    @return 同目录同文件名的 .yuyv 路径。
    """
    return output_path.with_suffix(".yuyv")


def archive_round_output(round_index: int) -> Path:
    """
    @brief 将当前轮次的 imgN 文件夹压缩保存到 /home，降低误删输出目录的风险。
    @param round_index 当前训练轮次，从 1 开始。
    @return 生成的压缩包路径。
    """
    round_output_root = get_round_output_root(round_index)
    if not round_output_root.exists():
        raise ValueError(f"当前轮次输出目录不存在，无法打包：{round_output_root}")

    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    archive_path = ARCHIVE_ROOT / f"{round_output_root.name}.tar.gz"
    temporary_archive_path = Path(f"{archive_path}.tmp")

    if temporary_archive_path.exists():
        temporary_archive_path.unlink()

    # 先写入临时文件，成功后再原子替换，避免中途崩溃留下损坏压缩包。
    with tarfile.open(temporary_archive_path, "w:gz") as archive_file:
        archive_file.add(round_output_root, arcname=round_output_root.name)

    temporary_archive_path.replace(archive_path)
    print(f"已打包当前轮次输出：{archive_path}")
    return archive_path


def archive_round_output_if_enabled(round_index: int) -> None:
    """
    @brief 根据环境变量决定是否归档当前轮次输出。
    @param round_index 当前训练轮次，从 1 开始。
    @return None
    """
    if not ENABLE_ARCHIVE:
        # 大批量生成 PNG 与 YUYV 时，跳过归档可避免数据盘被重复文件占满。
        print(f"跳过第 {round_index} 轮打包归档：ENABLE_ARCHIVE=0")
        return

    archive_round_output(round_index)


def configure_cycles_device(scene: bpy.types.Scene) -> None:
    """
    @brief 配置 Cycles 优先使用 NVIDIA GPU 渲染，避免服务器只跑 CPU。
    @param scene 当前需要渲染的场景。
    @return None
    """
    if scene.render.engine != "CYCLES":
        return

    cycles_addon = bpy.context.preferences.addons.get("cycles")
    if cycles_addon is None:
        print("未找到 Cycles 插件配置，保持当前渲染设备。")
        return

    cycles_preferences = cycles_addon.preferences

    # 服务器上的 OptiX 内核可能受驱动和 Blender 版本影响崩溃，优先使用更稳的 CUDA。
    selected_device_type = None
    for device_type in ("CUDA", "OPTIX"):
        try:
            cycles_preferences.compute_device_type = device_type
            cycles_preferences.get_devices()
            gpu_devices = [
                device
                for device in cycles_preferences.devices
                if device.type in {"OPTIX", "CUDA"} and "NVIDIA" in device.name.upper()
            ]
            if gpu_devices:
                selected_device_type = device_type
                break
        except Exception as exc:
            print(f"尝试启用 {device_type} 失败：{exc}")

    if selected_device_type is None:
        scene.cycles.device = "CPU"
        print("未识别到可用 NVIDIA GPU，回退到 CPU 渲染。")
        return

    # 只打开当前选中的 GPU 后端，关闭 CPU，防止 CPU拖慢 CUDA/OptiX 调度。
    scene.cycles.device = "GPU"
    for device in cycles_preferences.devices:
        device.use = device.type == selected_device_type and "NVIDIA" in device.name.upper()

    enabled_devices = [device.name for device in cycles_preferences.devices if device.use]
    print(f"Cycles 使用 {selected_device_type} GPU 渲染：{enabled_devices}")


def prepare_render_scene() -> bpy.types.Scene:
    """
    @brief 准备固定渲染设置。
    @return 当前工作场景。
    """
    scene = bpy.data.scenes.get(SCENE_NAME)
    camera_obj = bpy.data.objects.get(CAMERA_NAME)

    if scene is None:
        raise ValueError(f"未找到场景：{SCENE_NAME}")
    if camera_obj is None:
        raise ValueError(f"未找到相机：{CAMERA_NAME}")

    scene.camera = camera_obj
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = FINAL_RESOLUTION_X * SUPERSAMPLE_SCALE
    scene.render.resolution_y = FINAL_RESOLUTION_Y * SUPERSAMPLE_SCALE
    scene.render.resolution_percentage = 100

    if hasattr(scene.render, "use_motion_blur"):
        scene.render.use_motion_blur = False
    if hasattr(scene.view_settings, "view_transform"):
        scene.view_settings.view_transform = "Standard"
    if scene.render.engine == "CYCLES":
        configure_cycles_device(scene)
        # 固定训练渲染采样数，避免沿用 .blend 中较慢的高采样配置。
        scene.cycles.samples = RENDER_SAMPLES
        scene.cycles.use_denoising = False
        if hasattr(scene.cycles, "filter_width"):
            scene.cycles.filter_width = CYCLES_FILTER_WIDTH

    return scene


def load_round_params() -> dict[str, dict]:
    """
    @brief 读取已经生成过的轮次随机参数，用于断点续跑时保持一致。
    @return 轮次参数字典。
    """
    if not PARAMS_PATH.exists():
        return {}

    with PARAMS_PATH.open("r", encoding="utf-8") as params_file:
        round_params = json.load(params_file)

    # 相机单位和欧拉轴修正后，旧参数会让相机飞离目标，因此版本不一致时重新生成。
    metadata = round_params.get("_meta", {})
    if metadata.get("version") != ROUND_PARAM_VERSION:
        print(f"检测到旧随机参数文件，将按版本 {ROUND_PARAM_VERSION} 重新生成：{PARAMS_PATH}")
        return {}

    return {
        round_key: round_param
        for round_key, round_param in round_params.items()
        if round_key != "_meta"
    }


def save_round_params(round_params: dict[str, dict]) -> None:
    """
    @brief 保存每一轮随机参数，方便复现和断点续跑。
    @param round_params 轮次参数字典。
    @return None
    """
    PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)

    output_params = {
        "_meta": {
            "version": ROUND_PARAM_VERSION,
            "camera_note": "Blender MCP scene3 Camera.s3 units and XYZ euler axes",
        },
        **round_params,
    }

    with PARAMS_PATH.open("w", encoding="utf-8") as params_file:
        json.dump(output_params, params_file, ensure_ascii=False, indent=2)


def random_round_param() -> dict:
    """
    @brief 生成一轮新的随机灯光和相机参数。
    @return 一轮随机参数。
    """
    return {
        "front_light_energy": random.uniform(*FRONT_LIGHT_ENERGY_RANGE),
        "top_light_energy": random.uniform(*TOP_LIGHT_ENERGY_RANGE),
        "camera_location": [
            random.uniform(*CAMERA_LOCATION_X_RANGE),
            random.uniform(*CAMERA_LOCATION_Y_RANGE),
            CAMERA_LOCATION_Z,
        ],
        "camera_rotation_degrees": [
            random.uniform(*CAMERA_ROTATION_X_RANGE),
            random.uniform(*CAMERA_ROTATION_Y_RANGE),
            random.uniform(*CAMERA_ROTATION_Z_RANGE),
        ],
    }


def apply_round_param(round_param: dict) -> None:
    """
    @brief 将当前轮次的随机参数应用到灯光和相机。
    @param round_param 当前轮次参数。
    @return None
    """
    front_light = bpy.data.objects.get(FRONT_LIGHT_NAME)
    top_light = bpy.data.objects.get(TOP_LIGHT_NAME)
    camera_obj = bpy.data.objects.get(CAMERA_NAME)

    if front_light is None or front_light.type != "LIGHT":
        raise ValueError(f"未找到灯光：{FRONT_LIGHT_NAME}")
    if top_light is None or top_light.type != "LIGHT":
        raise ValueError(f"未找到灯光：{TOP_LIGHT_NAME}")
    if camera_obj is None or camera_obj.type != "CAMERA":
        raise ValueError(f"未找到相机：{CAMERA_NAME}")

    front_light.data.energy = round_param["front_light_energy"]
    top_light.data.energy = round_param["top_light_energy"]
    camera_obj.location = round_param["camera_location"]
    camera_obj.rotation_euler = [
        angle * 3.141592653589793 / 180.0
        for angle in round_param["camera_rotation_degrees"]
    ]


def assign_source_image(
    image_texture_node: bpy.types.ShaderNodeTexImage,
    reusable_image: bpy.types.Image | None,
    source_image_path: Path,
) -> bpy.types.Image:
    """
    @brief 替换图片块贴图。
    @param image_texture_node 目标图像纹理节点。
    @param reusable_image 循环复用的图片数据块。
    @param source_image_path 当前源图片路径。
    @return 已更新内容的图片数据块。
    """
    image_texture_node.interpolation = TEXTURE_INTERPOLATION

    if reusable_image is None:
        reusable_image = bpy.data.images.load(str(source_image_path), check_existing=True)
        image_texture_node.image = reusable_image
        return reusable_image

    reusable_image.filepath = str(source_image_path)
    reusable_image.reload()
    image_texture_node.image = reusable_image
    return reusable_image


def clamp_byte(value: float) -> int:
    """
    @brief 将浮点通道值限制到 8 位颜色范围。
    @param value 原始浮点通道值。
    @return 0-255 范围内的整数。
    """
    return max(0, min(255, int(round(value))))


def compute_image_luma_mean(image) -> float:
    """
    @brief 计算 RGB 图像的平均亮度。
    @param image Pillow RGB 图像。
    @return 按 BT.601 权重计算的平均亮度。
    """
    histogram = image.histogram()
    red_histogram = histogram[0:256]
    green_histogram = histogram[256:512]
    blue_histogram = histogram[512:768]
    pixel_count = max(image.width * image.height, 1)

    red_mean = sum(index * count for index, count in enumerate(red_histogram)) / pixel_count
    green_mean = sum(index * count for index, count in enumerate(green_histogram)) / pixel_count
    blue_mean = sum(index * count for index, count in enumerate(blue_histogram)) / pixel_count

    return 0.299 * red_mean + 0.587 * green_mean + 0.114 * blue_mean


def apply_camera_tone(image):
    """
    @brief 将渲染图调整为真实摄像头原图的室内低饱和微暖色调。
    @param image Pillow RGB 图像。
    @return 调整后的 Pillow RGB 图像。
    """
    if not APPLY_CAMERA_TONE:
        return image

    from PIL import Image
    from PIL import ImageEnhance

    # 先把整体亮度拉向真实拍摄均值，避免不同灯光随机导致训练分布过散。
    luma_mean = max(compute_image_luma_mean(image), 1.0)
    brightness_factor = CAMERA_TONE_TARGET_LUMA / luma_mean
    brightness_factor = max(CAMERA_TONE_BRIGHTNESS_MIN, min(CAMERA_TONE_BRIGHTNESS_MAX, brightness_factor))
    image = ImageEnhance.Brightness(image).enhance(brightness_factor)

    # 真实原图整体偏灰、低饱和，先收一点对比度和饱和度，再叠轻微暖色偏移。
    image = ImageEnhance.Contrast(image).enhance(CAMERA_TONE_CONTRAST)
    image = ImageEnhance.Color(image).enhance(CAMERA_TONE_SATURATION)

    red_channel, green_channel, blue_channel = image.split()
    red_channel = red_channel.point(lambda value: clamp_byte(value * CAMERA_TONE_RED_GAIN))
    green_channel = green_channel.point(lambda value: clamp_byte(value * CAMERA_TONE_GREEN_GAIN))
    blue_channel = blue_channel.point(lambda value: clamp_byte(value * CAMERA_TONE_BLUE_GAIN))
    return Image.merge("RGB", (red_channel, green_channel, blue_channel))


def apply_camera_tone_to_file(output_path: Path) -> None:
    """
    @brief 对已经保存的 PNG 文件应用真实摄像头色调。
    @param output_path 当前渲染输出 PNG 路径。
    @return None
    """
    if not APPLY_CAMERA_TONE:
        return

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("应用摄像头色调需要 Pillow，请先安装 requirements-blender.txt。") from exc

    with Image.open(output_path) as image:
        adjusted_image = apply_camera_tone(image.convert("RGB"))
        adjusted_image.save(output_path)


def resize_with_pillow(temp_path: Path, output_path: Path) -> bool:
    """
    @brief 使用 Pillow 对超采样结果进行高质量缩小和轻微锐化。
    @param temp_path 临时高分辨率图片路径。
    @param output_path 最终输出路径。
    @return Pillow 可用并成功处理时返回 True。
    """
    try:
        from PIL import Image, ImageFilter
    except ImportError:
        return False

    with Image.open(temp_path) as image:
        image = image.convert("RGB")
        image = image.resize((FINAL_RESOLUTION_X, FINAL_RESOLUTION_Y), Image.Resampling.LANCZOS)

        if USE_POST_SHARPEN:
            image = image.filter(
                ImageFilter.UnsharpMask(
                    radius=UNSHARP_RADIUS,
                    percent=UNSHARP_PERCENT,
                    threshold=UNSHARP_THRESHOLD,
                )
            )

        image = apply_camera_tone(image)
        image.save(output_path)

    return True


def resize_with_blender(temp_path: Path, output_path: Path, scene: bpy.types.Scene) -> None:
    """
    @brief Pillow 不可用时使用 Blender 内置缩放兜底。
    @param temp_path 临时高分辨率图片路径。
    @param output_path 最终输出路径。
    @param scene 当前场景。
    @return None
    """
    image = bpy.data.images.load(str(temp_path), check_existing=False)
    try:
        image.scale(FINAL_RESOLUTION_X, FINAL_RESOLUTION_Y)
        image.save_render(filepath=str(output_path), scene=scene)
    finally:
        bpy.data.images.remove(image)


def rgb_to_yuv_pixel(red: int, green: int, blue: int) -> tuple[int, int, int]:
    """
    @brief 将单个 RGB 像素转换为 BT.601 近似 YUV 分量。
    @param red 红色通道，范围 0-255。
    @param green 绿色通道，范围 0-255。
    @param blue 蓝色通道，范围 0-255。
    @return Y、U、V 三个 8 位分量。
    """
    # 使用常见 BT.601 全范围近似公式，模拟 OpenCV BGR->YUYV 的通道语义。
    y_value = 0.299 * red + 0.587 * green + 0.114 * blue
    u_value = -0.169 * red - 0.331 * green + 0.500 * blue + 128.0
    v_value = 0.500 * red - 0.419 * green - 0.081 * blue + 128.0

    return (
        max(0, min(255, int(round(y_value)))),
        max(0, min(255, int(round(u_value)))),
        max(0, min(255, int(round(v_value)))),
    )


def save_png_as_yuyv(output_path: Path) -> Path:
    """
    @brief 将最终 PNG 转换为同尺寸 YUYV422 raw 文件，供后续识别流程直接使用。
    @param output_path 当前渲染输出 PNG 路径。
    @return 生成的 .yuyv 文件路径。
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("生成 YUYV 输出需要 Pillow，请先安装 requirements-blender.txt。") from exc

    with Image.open(output_path) as image:
        rgb_image = image.convert("RGB")
        width, height = rgb_image.size

        if width % 2 != 0:
            # YUYV422 两个水平像素共享一组 UV，奇数宽度会破坏配对关系。
            width -= 1
            rgb_image = rgb_image.crop((0, 0, width, height))

        pixels = rgb_image.load()
        yuyv_bytes = bytearray(width * height * 2)
        write_index = 0

        for y in range(height):
            for x in range(0, width, 2):
                red0, green0, blue0 = pixels[x, y]
                red1, green1, blue1 = pixels[x + 1, y]

                # 两个像素分别保留亮度，色度取 pair 平均值，匹配 YUYV422 采样方式。
                y0, u0, v0 = rgb_to_yuv_pixel(red0, green0, blue0)
                y1, u1, v1 = rgb_to_yuv_pixel(red1, green1, blue1)
                shared_u = (u0 + u1) // 2
                shared_v = (v0 + v1) // 2

                yuyv_bytes[write_index] = y0
                yuyv_bytes[write_index + 1] = shared_u
                yuyv_bytes[write_index + 2] = y1
                yuyv_bytes[write_index + 3] = shared_v
                write_index += 4

    yuyv_output_path = build_yuyv_output_path(output_path)
    yuyv_output_path.write_bytes(yuyv_bytes)
    return yuyv_output_path


def render_still(scene: bpy.types.Scene, output_path: Path) -> None:
    """
    @brief 先高分辨率渲染，再缩小到最终摄像头分辨率。
    @param scene 当前场景。
    @param output_path 最终输出路径。
    @return None
    """
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / f"{output_path.stem}_supersampled.png"
        scene.render.filepath = str(temp_path)
        bpy.ops.render.render(write_still=True)

        if not resize_with_pillow(temp_path, output_path):
            resize_with_blender(temp_path, output_path, scene)
            apply_camera_tone_to_file(output_path)

    if SAVE_YUYV_OUTPUT:
        save_png_as_yuyv(output_path)


def ensure_yuyv_sidecar(output_path: Path) -> None:
    """
    @brief 在断点续跑跳过 PNG 渲染时补齐缺失的 YUYV sidecar 文件。
    @param output_path 当前已经存在的 PNG 输出路径。
    @return None
    """
    if not SAVE_YUYV_OUTPUT:
        return
    if build_yuyv_output_path(output_path).exists():
        return
    save_png_as_yuyv(output_path)



def main() -> None:
    """
    @brief 5090 服务器批量渲染入口。
    @return None
    """
    if not bpy.data.filepath:
        raise ValueError("请先保存 .blend 文件，再运行脚本。")

    scene = prepare_render_scene()
    source_dirs = select_source_dirs()
    target_obj = find_target_object()
    image_texture_node = find_image_texture_node(target_obj)
    round_params = load_round_params()
    original_image = image_texture_node.image
    original_filepath = None if original_image is None else original_image.filepath
    reusable_image = original_image

    try:
        for round_index in range(1, TRAINING_ROUNDS + 1):
            round_key = str(round_index)

            if round_key not in round_params:
                round_params[round_key] = random_round_param()
                save_round_params(round_params)

            apply_round_param(round_params[round_key])
            print(f"\n开始第 {round_index}/{TRAINING_ROUNDS} 轮：{round_params[round_key]}")

            for source_dir in source_dirs:
                source_images = collect_images_in_dir(source_dir)
                print(f"处理目录：{source_dir.relative_to(DATASET_ROOT)}，共 {len(source_images)} 张")

                for image_index, source_image_path in enumerate(source_images, start=1):
                    output_path = build_output_path(source_image_path, round_index)

                    if output_path.exists():
                        ensure_yuyv_sidecar(output_path)
                        if image_index == 1 or image_index == len(source_images) or image_index % 25 == 0:
                            print(f"[{image_index}/{len(source_images)}] 跳过：{output_path}")
                        continue

                    reusable_image = assign_source_image(
                        image_texture_node,
                        reusable_image,
                        source_image_path,
                    )
                    render_still(scene, output_path)

                    if image_index == 1 or image_index == len(source_images) or image_index % 25 == 0:
                        print(f"[{image_index}/{len(source_images)}] 渲染：{source_image_path} -> {output_path}")

            archive_round_output_if_enabled(round_index)
    finally:
        image_texture_node.image = original_image

        if original_image is not None and original_filepath is not None:
            original_image.filepath = original_filepath
            original_image.reload()

    print("\n5090 批量渲染完成。")


main()
