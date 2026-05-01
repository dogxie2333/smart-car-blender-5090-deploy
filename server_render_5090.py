import json
import random
from pathlib import Path
from tempfile import TemporaryDirectory

import bpy


PROJECT_ROOT = Path(bpy.data.filepath).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
DATASET_ROOT = SRC_ROOT / "数据集图片"
PARAMS_PATH = PROJECT_ROOT / "round_params_5090.json"

SCENE_NAME = "scene3"
CAMERA_NAME = "Camera.s3"
FRONT_LIGHT_NAME = "前方补光灯"
TOP_LIGHT_NAME = "顶部补光灯"
TARGET_OBJECT_CANDIDATES = ("图片块", "广告图-上方")
TARGET_MATERIAL_CANDIDATES = ("广告图-上方-纯图片", "material-cache-loong21")

# 宏定义：训练轮数和要渲染的数据集编号。
TRAINING_ROUNDS = 100
SELECTED_DIR_INDICES = (11, 13, 4, 5, 7, 10)

# 宏定义：最终摄像头分辨率和超采样倍率。
FINAL_RESOLUTION_X = 320
FINAL_RESOLUTION_Y = 240
SUPERSAMPLE_SCALE = 4

# 宏定义：贴图与渲染清晰度设置。
TEXTURE_INTERPOLATION = "Cubic"
CYCLES_FILTER_WIDTH = 0.5
RENDER_SAMPLES = 64
USE_POST_SHARPEN = True
UNSHARP_RADIUS = 0.7
UNSHARP_PERCENT = 120
UNSHARP_THRESHOLD = 2

# 宏定义：随机范围。
FRONT_LIGHT_ENERGY_RANGE = (30000.0, 50000.0)
TOP_LIGHT_ENERGY_RANGE = (8.0, 8.0)
CAMERA_LOCATION_X_RANGE = (-0.675, -0.565)
CAMERA_LOCATION_Y_RANGE = (-0.265, -0.225)
CAMERA_ROTATION_X_RANGE = (67.0, 71.0)
CAMERA_ROTATION_Y_RANGE = (-3.0, 3.0)
CAMERA_ROTATION_Z_RANGE = (-92.0, -89.0)

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


def select_source_dirs() -> list[Path]:
    """
    @brief 根据 SELECTED_DIR_INDICES 选择要渲染的六个具体子类目录。
    @return 选中的源目录列表。
    """
    if not DATASET_ROOT.exists():
        raise ValueError(f"未找到数据集目录：{DATASET_ROOT}")

    leaf_dirs = list_leaf_source_dirs(DATASET_ROOT)
    selected_dirs = []

    for directory_index in SELECTED_DIR_INDICES:
        if directory_index < 1 or directory_index > len(leaf_dirs):
            raise ValueError(f"目录编号超出范围：{directory_index}")
        selected_dirs.append(leaf_dirs[directory_index - 1])

    print("本次选择目录：")
    for directory_index, source_dir in zip(SELECTED_DIR_INDICES, selected_dirs):
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
    return image_paths


def get_round_output_root(round_index: int) -> Path:
    """
    @brief 获取当前轮次的输出根目录。
    @param round_index 当前训练轮次，从 1 开始。
    @return 当前轮次输出目录，例如 img1、img2。
    """
    return PROJECT_ROOT / f"img{round_index}"


def build_output_path(source_image_path: Path, round_index: int) -> Path:
    """
    @brief 构造当前轮次输出路径，继承 src 子目录结构，并保持原文件名。
    @param source_image_path 当前源图片路径。
    @param round_index 当前训练轮次，从 1 开始。
    @return 输出 PNG 路径。
    """
    relative_path = source_image_path.relative_to(SRC_ROOT)
    output_path = get_round_output_root(round_index) / relative_path.parent / f"{source_image_path.stem}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


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

    # 优先使用 OptiX；如果当前 Blender/驱动不支持，再回退到 CUDA。
    selected_device_type = None
    for device_type in ("OPTIX", "CUDA"):
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

    # 只打开 GPU，关闭 CPU，防止 CPU 拖慢 OptiX/CUDA 调度。
    scene.cycles.device = "GPU"
    for device in cycles_preferences.devices:
        device.use = device.type in {"OPTIX", "CUDA"} and "NVIDIA" in device.name.upper()

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
        return json.load(params_file)


def save_round_params(round_params: dict[str, dict]) -> None:
    """
    @brief 保存每一轮随机参数，方便复现和断点续跑。
    @param round_params 轮次参数字典。
    @return None
    """
    PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with PARAMS_PATH.open("w", encoding="utf-8") as params_file:
        json.dump(round_params, params_file, ensure_ascii=False, indent=2)


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
            0.32,
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
    finally:
        image_texture_node.image = original_image

        if original_image is not None and original_filepath is not None:
            original_image.filepath = original_filepath
            original_image.reload()

    print("\n5090 批量渲染完成。")


main()
