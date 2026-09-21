"""
Animation Deformation Scorer & Selector.

用途：在导出 clean/dirty 网格之前，先对动画库打分并筛掉"形变量雷同"的动画。

流程：
  1. 载入一个带蒙皮的角色 FBX（如 T-Pose.fbx），记录 rest 状态顶点坐标。
  2. 逐个动画 FBX：重定向 Armature Modifier → 跳到结束帧（另记中间帧）→
     计算顶点相对 rest 的位移统计量（rms / mean / p95 / max）。
  3. 按分数去重：分数低于阈值的丢弃；与已保留分数差距小于 epsilon 的视为重复。

所有动画的分数都会写入 anim_scores.json，因此调 epsilon / 阈值时可用
--select_only 离线重筛，无需重跑 Blender。

用法（Blender Development）：
  直接 Run Script，按提示输入 base mesh / animations dir / output dir。

用法（命令行）：
  blender --background --python score_animations_by_deformation.py -- \
      --base_mesh_fbx "/path/to/T-Pose.fbx" \
      --animations_dir "/path/to/animations" \
      --output_dir "/path/to/scores"

离线重筛（不需要 Blender 场景，但仍走 Blender 解释器）：
  blender --background --python score_animations_by_deformation.py -- \
      --select_only --output_dir "/path/to/scores" --epsilon 0.006
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import traceback
from pathlib import Path

import numpy as np

# --select_only 只在已有分数上重跑筛选，不碰 Blender，因此允许无 bpy 环境导入本模块。
try:
    import bpy
except ImportError:
    bpy = None


# =========================
# 默认路径与参数
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # .../Data_Processing

DEFAULT_BASE_MESH_FBX = PROJECT_ROOT / "Data" / "input_character_fbx"
DEFAULT_ANIMATIONS_DIR = (
    PROJECT_ROOT / "Data" / "animation_fbx" / "input_animation_fbx" / "06f"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Data" / "animation_selection"

# 顶点抽稀：只取每 VERTEX_STRIDE 个顶点参与统计，纯粹为了提速
VERTEX_STRIDE = 8

# 每个动画在时间轴上均匀采样多少帧用于打分。
# 不能只看首/末帧：Mixamo 大量片段是循环的，末帧会回到起始姿态，
# 只看端点会把真正有大形变的动作误判为"几乎没动"。
SAMPLE_FRAMES = 9

# 比较形变前如何去掉整体运动：rigid（平移+旋转）/ centroid（仅平移）/ none
ALIGN_MODE = "rigid"

# 选择策略默认值（分数均已按角色身高归一化，因此是无量纲的）
DEFAULT_METRIC = "rms_max"
DEFAULT_EPSILON = 0.005
DEFAULT_MIN_SCORE_PERCENTILE = 20.0
DEFAULT_MAX_KEEP = 0  # 0 表示不限制

# tiered 策略：高分段（大形变=重拓扑难例）全部保留，越往低分抽得越稀。
# gap 策略在全库上用同一个 epsilon 去重，而低分段人口密集、高分段稀疏，
# 结果是常见姿态被清空、极端姿态全留，因此默认改用 tiered。
DEFAULT_SELECT_MODE = "tiered"
DEFAULT_KEEP_ALL_ABOVE = 0.35
# "分数下界:配额" 列表，从高到低；每档的上界是前一档的下界（最高档为 KEEP_ALL_ABOVE）
DEFAULT_TIERS = "0.25:45,0.15:40,0.08:35,0.02:25"
# 全留档内部仍要剔除"同一动作的重复文件"：同名族且分数差小于该阈值视为孪生
DEFAULT_TWIN_TOL = 0.004

# 打分用的动画目录是多个角色 zip 解压到一起形成的，含大量 "Xxx (3).fbx" 之类的重名副本，
# 这些文件在原始 zip 里并不存在。用各文件名在 72 个角色 zip 中的真实出现次数来过滤：
# 覆盖数为 0 的是解压产生的副本；覆盖数低的动画只有少数角色有，凑不齐统一的验证集。
# 注意不能靠剥掉 "(N)" 后缀来判断——Mixamo 原始命名里就有 "sword and shield walk (2).fbx"。
DEFAULT_COVERAGE_TSV = DEFAULT_OUTPUT_DIR / "zip_anim_coverage.tsv"
DEFAULT_MIN_ZIP_COVERAGE = 60

ACTIONS_PER_RESET = 100

SCORES_JSON_NAME = "anim_scores.json"
SCORES_CSV_NAME = "anim_scores.csv"
SELECTED_TXT_NAME = "selected_anims.txt"

RUN_LOG_PATH: str | None = None


def log(msg: str) -> None:
    text = f"[AnimScorer] {msg}"
    print(text)
    if RUN_LOG_PATH:
        try:
            with open(RUN_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except OSError:
            pass


def prompt_path(label: str, default_path: Path, must_exist: bool = True,
                create_dir: bool = False) -> Path:
    raw = input(f"Enter {label} [{default_path}]: ").strip()
    selected = Path(raw).expanduser().resolve() if raw else default_path
    if create_dir:
        selected.mkdir(parents=True, exist_ok=True)
    elif must_exist and not selected.exists():
        raise FileNotFoundError(f"{label} not found: {selected}")
    return selected


def show_popup(title: str, message: str, icon: str = "INFO") -> None:
    try:
        if bpy is None or getattr(bpy.app, "background", False):
            return
        wm = bpy.context.window_manager
        if wm is None:
            return

        def _draw(self, _context):
            for line in message.split("\n"):
                self.layout.label(text=line)

        wm.popup_menu(_draw, title=title, icon=icon)
    except Exception:
        pass


def sanitize_name(name: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", name).strip("_") or "unnamed"


# =========================
# Blender 场景工具
# =========================
def purge_orphans() -> None:
    for _ in range(3):
        try:
            bpy.ops.outliner.orphans_purge(
                do_local_ids=True, do_linked_ids=True, do_recursive=True
            )
        except Exception:
            break


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    purge_orphans()


def import_fbx_collect_objects(fbx_path: Path) -> list[bpy.types.Object]:
    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.fbx(filepath=str(fbx_path), automatic_bone_orientation=True)
    after = set(bpy.data.objects.keys())
    return [bpy.data.objects[n] for n in (after - before) if n in bpy.data.objects]


def delete_objects(objects: list[bpy.types.Object]) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    selected = False
    for obj in objects:
        if obj and obj.name in bpy.data.objects:
            obj.select_set(True)
            selected = True
    if selected:
        bpy.ops.object.delete(use_global=False)
    purge_orphans()


def get_primary_armature(objects: list[bpy.types.Object]) -> bpy.types.Object | None:
    armatures = [o for o in objects if o.type == "ARMATURE"]
    if not armatures:
        return None
    return sorted(armatures, key=lambda o: len(o.data.bones), reverse=True)[0]


def collect_mesh_objects(objects: list[bpy.types.Object]) -> list[bpy.types.Object]:
    return [o for o in objects if o.type == "MESH" and o.name in bpy.data.objects]


def probe_fbx_has_mesh(fbx_path: Path) -> bool:
    clear_scene()
    imported = import_fbx_collect_objects(fbx_path)
    try:
        return any(obj.type == "MESH" for obj in imported)
    finally:
        delete_objects(imported)


def resolve_base_mesh_fbx(path: Path) -> Path:
    """支持传入单个 FBX 或角色目录；传目录时取第一个含网格的 FBX。"""
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"base_mesh_fbx not found: {path}")

    for candidate in sorted(path.glob("*.fbx")):
        if probe_fbx_has_mesh(candidate):
            log(f"Auto-picked base mesh from directory: {candidate.name}")
            return candidate
    raise RuntimeError(f"No FBX with MESH data found in directory: {path}")


def ensure_armature_modifier(mesh_obj: bpy.types.Object) -> None:
    if not any(mod.type == "ARMATURE" for mod in mesh_obj.modifiers):
        mesh_obj.modifiers.new(name="Armature", type="ARMATURE")


def retarget_mesh_to_armature(
    mesh_objects: list[bpy.types.Object], armature_obj: bpy.types.Object
) -> None:
    for mesh_obj in mesh_objects:
        ensure_armature_modifier(mesh_obj)
        for mod in mesh_obj.modifiers:
            if mod.type == "ARMATURE":
                mod.object = armature_obj


def collect_vertex_group_names(mesh_objects: list[bpy.types.Object]) -> set[str]:
    names: set[str] = set()
    for mesh_obj in mesh_objects:
        names |= {vg.name for vg in mesh_obj.vertex_groups}
    return names


def bone_key(name: str) -> str:
    """去掉命名空间前缀，例如 mixamorig5:Hips -> Hips。"""
    return name.split(":")[-1]


def align_armature_bone_names(
    armature_obj: bpy.types.Object, target_names: set[str]
) -> int:
    """
    把动画骨骼重命名为角色顶点组使用的名字。

    动画库里混着 mixamorig: 与 mixamorig5: 等不同前缀，前缀对不上时蒙皮
    不会被驱动，网格会停在绑定姿态，打分结果是一个恒定的"假低分"。
    重命名骨骼后 Blender 会同步更新 action 的通道路径。
    """
    if not target_names:
        return 0

    target_by_key = {bone_key(name): name for name in target_names}
    renamed = 0
    for bone in armature_obj.data.bones:
        wanted = target_by_key.get(bone_key(bone.name))
        if wanted and wanted != bone.name:
            bone.name = wanted
            renamed += 1
    return renamed


# =========================
# 形变度量
# =========================
def evaluated_world_coords(
    mesh_objects: list[bpy.types.Object], stride: int
) -> np.ndarray:
    """当前帧所有网格的世界坐标（按 stride 抽稀后拼接）。"""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    chunks: list[np.ndarray] = []

    for src_obj in mesh_objects:
        if src_obj.name not in bpy.data.objects:
            continue
        eval_obj = src_obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        if mesh is None:
            continue
        try:
            count = len(mesh.vertices)
            if count == 0:
                continue
            flat = np.empty(count * 3, dtype=np.float64)
            mesh.vertices.foreach_get("co", flat)
            coords = flat.reshape(count, 3)
            matrix = np.array(eval_obj.matrix_world, dtype=np.float64)
            coords = coords @ matrix[:3, :3].T + matrix[:3, 3]
            if stride > 1:
                coords = coords[::stride]
            chunks.append(coords)
        finally:
            eval_obj.to_mesh_clear()

    if not chunks:
        raise RuntimeError("No evaluated mesh vertices available.")
    return np.concatenate(chunks, axis=0)


def character_scale(rest_coords: np.ndarray) -> float:
    """用 rest 姿态包围盒最长边作为归一化尺度，使不同角色分数可比。"""
    extent = rest_coords.max(axis=0) - rest_coords.min(axis=0)
    scale = float(extent.max())
    return scale if scale > 1e-6 else 1.0


def align_point_clouds(
    rest_coords: np.ndarray, current_coords: np.ndarray, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    对齐两组点云后再比较，避免"整体位移"被当成形变。

    Mixamo 里 Flying / Sprint 这类动作会跑出很远，不对齐的话它们的分数
    完全由 root motion 决定，真正的姿态形变反而被淹没。
      - none    : 直接比较世界坐标
      - centroid: 去掉平移
      - rigid   : 去掉平移与旋转（Kabsch），只保留姿态差异
    """
    if mode == "none":
        return rest_coords, current_coords

    rest_centered = rest_coords - rest_coords.mean(axis=0)
    current_centered = current_coords - current_coords.mean(axis=0)
    if mode == "centroid":
        return rest_centered, current_centered

    covariance = current_centered.T @ rest_centered
    u, _, vt = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    return rest_centered, current_centered @ rotation.T


def displacement_metrics(
    rest_coords: np.ndarray,
    current_coords: np.ndarray,
    scale: float,
    align_mode: str,
) -> dict[str, float]:
    if rest_coords.shape != current_coords.shape:
        raise RuntimeError(
            f"Vertex count mismatch: rest={rest_coords.shape}, current={current_coords.shape}"
        )
    rest_aligned, current_aligned = align_point_clouds(
        rest_coords, current_coords, align_mode
    )
    dist = np.linalg.norm(current_aligned - rest_aligned, axis=1) / scale
    return {
        "rms": float(np.sqrt(np.mean(dist ** 2))),
        "mean": float(np.mean(dist)),
        "p95": float(np.percentile(dist, 95)),
        "max": float(np.max(dist)),
    }


# =========================
# 打分主流程
# =========================
def init_base_scene(
    base_mesh_fbx: Path, stride: int
) -> tuple[list[bpy.types.Object], np.ndarray, float, set[str]]:
    """载入角色，返回 (网格对象, rest 顶点坐标, 归一化尺度, 顶点组名集合)。"""
    clear_scene()
    imported = import_fbx_collect_objects(base_mesh_fbx)
    mesh_objects = collect_mesh_objects(imported)
    if not mesh_objects:
        raise RuntimeError(f"Base mesh FBX contains no MESH objects: {base_mesh_fbx}")

    ref_armature = get_primary_armature(imported)
    if ref_armature is not None:
        retarget_mesh_to_armature(mesh_objects, ref_armature)
        # rest 状态：清掉角色自带的动画，确保是绑定姿态
        if ref_armature.animation_data is not None:
            ref_armature.animation_data.action = None
    bpy.context.view_layer.update()

    rest_coords = evaluated_world_coords(mesh_objects, stride)
    scale = character_scale(rest_coords)
    vgroup_names = collect_vertex_group_names(mesh_objects)
    log(
        f"Base mesh loaded: {len(mesh_objects)} mesh objects, "
        f"{rest_coords.shape[0]} sampled vertices, scale={scale:.4f}, "
        f"{len(vgroup_names)} vertex groups"
    )
    return mesh_objects, rest_coords, scale, vgroup_names


def get_action_frame_range(armature_obj: bpy.types.Object) -> tuple[int, int]:
    anim = armature_obj.animation_data
    if anim is None or anim.action is None:
        raise RuntimeError(f"Armature has no animation action: {armature_obj.name}")
    return int(anim.action.frame_range[0]), int(anim.action.frame_range[1])


def sample_frame_numbers(frame_start: int, frame_end: int, count: int) -> list[int]:
    """在 [start, end] 上均匀取 count 帧（含端点），去重后升序返回。"""
    if frame_end <= frame_start or count <= 1:
        return [frame_end]
    positions = np.linspace(frame_start, frame_end, num=count)
    return sorted({int(round(p)) for p in positions})


def score_single_animation(
    anim_fbx: Path,
    mesh_objects: list[bpy.types.Object],
    rest_coords: np.ndarray,
    scale: float,
    stride: int,
    vgroup_names: set[str],
    sample_frames: int,
    align_mode: str,
) -> dict:
    """在整段动画上均匀采样若干帧，统计相对 rest 的形变。"""
    imported: list[bpy.types.Object] = []
    try:
        imported = import_fbx_collect_objects(anim_fbx)

        # 动画文件里若夹带网格，先删掉；删除后原引用会失效，用名字重新取回
        extra_meshes = [o for o in imported if o.type == "MESH"]
        if extra_meshes:
            keep_names = [o.name for o in imported if o.type != "MESH"]
            delete_objects(extra_meshes)
            imported = [
                bpy.data.objects[n] for n in keep_names if n in bpy.data.objects
            ]

        anim_armature = get_primary_armature(imported)
        if anim_armature is None:
            raise RuntimeError("No armature found in animation FBX")

        renamed = align_armature_bone_names(anim_armature, vgroup_names)
        bone_names = {b.name for b in anim_armature.data.bones}
        overlap = len(bone_names & vgroup_names)
        if overlap == 0:
            raise RuntimeError(
                "Animation armature shares no bone names with character vertex groups; "
                "mesh would not deform"
            )

        retarget_mesh_to_armature(mesh_objects, anim_armature)

        scene = bpy.context.scene
        frame_start, frame_end = get_action_frame_range(anim_armature)
        frames = sample_frame_numbers(frame_start, frame_end, sample_frames)

        per_frame: list[dict[str, float]] = []
        for frame in frames:
            scene.frame_set(frame)
            bpy.context.view_layer.update()
            per_frame.append(
                displacement_metrics(
                    rest_coords,
                    evaluated_world_coords(mesh_objects, stride),
                    scale,
                    align_mode,
                )
            )

        rms_series = [m["rms"] for m in per_frame]
        peak_idx = int(np.argmax(rms_series))

        return {
            "name": anim_fbx.stem,
            "file": anim_fbx.name,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "sampled_frames": frames,
            "bone_overlap": overlap,
            "bones_renamed": renamed,
            # 主指标：整段峰值形变，循环动作也不会被端点拉低
            "rms_max": float(np.max(rms_series)),
            "rms_mean": float(np.mean(rms_series)),
            "rms_end": rms_series[-1],
            "p95_max": float(max(m["p95"] for m in per_frame)),
            "max_max": float(max(m["max"] for m in per_frame)),
            "peak_frame": frames[peak_idx],
            "rms_series": [round(v, 6) for v in rms_series],
        }
    finally:
        if imported:
            delete_objects(imported)


def score_animation_library(
    base_mesh_fbx: Path,
    animations_dir: Path,
    stride: int,
    max_animations: int | None,
    error_log_path: Path,
    sample_frames: int,
    align_mode: str,
) -> list[dict]:
    anim_files = sorted(
        p
        for p in animations_dir.glob("*.fbx")
        if p.resolve() != base_mesh_fbx.resolve()
    )
    if max_animations is not None and max_animations > 0:
        anim_files = anim_files[:max_animations]
    if not anim_files:
        raise FileNotFoundError(f"No animation FBX found in: {animations_dir}")

    mesh_objects, rest_coords, scale, vgroup_names = init_base_scene(
        base_mesh_fbx, stride
    )
    log(f"Animations to score: {len(anim_files)} (sample_frames={sample_frames})")

    records: list[dict] = []
    for idx, anim_fbx in enumerate(anim_files, start=1):
        try:
            record = score_single_animation(
                anim_fbx,
                mesh_objects,
                rest_coords,
                scale,
                stride,
                vgroup_names,
                sample_frames,
                align_mode,
            )
            records.append(record)
            if idx % 25 == 0 or idx == len(anim_files):
                log(f"  [{idx}/{len(anim_files)}] scored, latest={record['name']}")
        except Exception as exc:
            tb = traceback.format_exc()
            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write(f"{anim_fbx}\n{exc}\n{tb}\n{'-' * 80}\n")
            log(f"  Failed: {anim_fbx.name} | {exc}")
        finally:
            if idx % ACTIONS_PER_RESET == 0:
                log("  Periodic scene reset to free memory...")
                mesh_objects, rest_coords, scale, vgroup_names = init_base_scene(
                    base_mesh_fbx, stride
                )

    return records


# =========================
# 按分数去重
# =========================
def select_by_score_gap(
    records: list[dict],
    metric: str,
    epsilon: float,
    min_score: float,
    max_keep: int,
) -> list[dict]:
    """分数从高到低贪心：与已保留分数差距小于 epsilon 的视为重复，丢弃。"""
    ranked = sorted(records, key=lambda r: r.get(metric, 0.0), reverse=True)
    kept_scores: list[float] = []

    for record in ranked:
        score = float(record.get(metric, 0.0))
        if score < min_score:
            record["keep"] = False
            record["skip_reason"] = "below_min_score"
            continue
        if max_keep > 0 and len(kept_scores) >= max_keep:
            record["keep"] = False
            record["skip_reason"] = "max_keep_reached"
            continue
        if any(abs(score - kept) < epsilon for kept in kept_scores):
            record["keep"] = False
            record["skip_reason"] = "similar_score_exists"
            continue
        record["keep"] = True
        record["skip_reason"] = ""
        kept_scores.append(score)

    return ranked


# =========================
# 可用性过滤
# =========================
def load_zip_coverage(path: Path) -> dict[str, int]:
    """读取 "文件名<TAB>出现在多少个角色 zip 中" 的清单。"""
    coverage: dict[str, int] = {}
    if not path.is_file():
        return coverage
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            name, _, count = line.rpartition("\t")
            try:
                coverage[name] = int(count)
            except ValueError:
                continue
    return coverage


def filter_by_zip_coverage(
    records: list[dict], coverage: dict[str, int], min_coverage: int
) -> list[dict]:
    """只保留在足够多角色 zip 中都存在的动画，其余标记丢弃。"""
    if not coverage:
        log("No coverage table loaded; skipping availability filter")
        return records

    available: list[dict] = []
    missing = 0
    rare = 0
    for record in records:
        got = coverage.get(record.get("file", ""), 0)
        record["zip_coverage"] = got
        if got == 0:
            record["keep"] = False
            record["skip_reason"] = "not_in_any_zip"
            missing += 1
        elif got < min_coverage:
            record["keep"] = False
            record["skip_reason"] = "rare_across_characters"
            rare += 1
        else:
            available.append(record)

    log(
        f"Availability filter (min_zip_coverage={min_coverage}): "
        f"{len(available)} usable, {missing} unzip artifacts, {rare} too rare"
    )
    return available


# =========================
# 分层配额选择
# =========================
def parse_tiers(spec: str) -> list[tuple[float, int]]:
    """把 "0.25:45,0.15:40" 解析成 [(0.25, 45), (0.15, 40)]，按分数下界降序。"""
    tiers: list[tuple[float, int]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        lo_text, _, quota_text = chunk.partition(":")
        if not quota_text:
            raise ValueError(f"Bad tier spec (expect 'low:quota'): {chunk}")
        tiers.append((float(lo_text), int(quota_text)))
    tiers.sort(key=lambda t: t[0], reverse=True)
    return tiers


def name_family(name: str) -> str:
    """去掉副本/编号后缀，用于识别同一动作的重复文件。"""
    text = name.lower().replace("_", " ")
    text = re.sub(r"\(\d+\)|副本\d*|\bvar\w*\s*\d*|\b\d+\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def drop_twins(group: list[dict], metric: str, twin_tol: float) -> list[dict]:
    """同名族且分数几乎相同的只留一个，其余全部保留。"""
    kept: list[dict] = []
    seen: dict[str, list[float]] = {}
    for record in sorted(group, key=lambda r: r.get(metric, 0.0), reverse=True):
        score = float(record.get(metric, 0.0))
        family = name_family(record["name"])
        scores = seen.setdefault(family, [])
        if any(abs(score - s) < twin_tol for s in scores):
            record["keep"] = False
            record["skip_reason"] = "duplicate_twin"
            continue
        scores.append(score)
        record["keep"] = True
        record["skip_reason"] = ""
        kept.append(record)
    return kept


def spread_pick(group: list[dict], metric: str, quota: int) -> list[dict]:
    """档内按分数升序等距取 quota 个，避免全挤在同一个分数附近。"""
    ordered = sorted(group, key=lambda r: r.get(metric, 0.0))
    if len(ordered) <= quota:
        chosen = ordered
    else:
        step = len(ordered) / quota
        chosen = [ordered[int(j * step)] for j in range(quota)]
    chosen_ids = {id(r) for r in chosen}
    for record in ordered:
        if id(record) in chosen_ids:
            record["keep"] = True
            record["skip_reason"] = ""
        else:
            record["keep"] = False
            record["skip_reason"] = "tier_quota_full"
    return chosen


def select_tiered(
    records: list[dict],
    metric: str,
    keep_all_above: float,
    tiers: list[tuple[float, int]],
    twin_tol: float,
) -> list[dict]:
    """高分段全留（仅去孪生），低分段按档位配额稀疏抽样。"""
    for record in records:
        record["keep"] = False
        record["skip_reason"] = "below_min_score"

    top = [r for r in records if float(r.get(metric, 0.0)) >= keep_all_above]
    kept_top = drop_twins(top, metric, twin_tol)
    log(
        f"  tier >={keep_all_above:.3f}  pool={len(top):4} kept={len(kept_top):4} "
        f"(dropped {len(top) - len(kept_top)} twins)"
    )

    upper = keep_all_above
    for lo, quota in tiers:
        group = [
            r for r in records if lo <= float(r.get(metric, 0.0)) < upper
        ]
        chosen = spread_pick(group, metric, quota)
        log(
            f"  tier {lo:.3f}–{upper:.3f}  pool={len(group):4} kept={len(chosen):4} "
            f"({(len(chosen) / len(group) if group else 0):.1%})"
        )
        upper = lo

    below = sum(1 for r in records if float(r.get(metric, 0.0)) < upper)
    log(f"  below {upper:.3f}: {below} dropped as near-static")

    return sorted(records, key=lambda r: r.get(metric, 0.0), reverse=True)


# 仅用于体检选择结果：按文件名关键字粗分类，统计每类被选中的比例。
# 这些标签不参与选择本身，只是用来发现"整类动作被清空"这种问题。
COVERAGE_THEMES = [
    ("walk", "walk"),
    ("run", "run"),
    ("idle", "idle"),
    ("jump", "jump"),
    ("turn", "turn"),
    ("crouch", "crouch"),
    ("sword", "sword"),
    ("punch|kick|attack|slash|fight|combat", "combat"),
    ("rifle|pistol|gun|shoot|reload", "gun"),
    ("danc", "dance"),
    ("sit|stand up|standing up", "sitstand"),
    ("death|dying|die|hit react|shoved|stagger", "impact"),
    ("climb|hang|swing", "climb"),
    ("fall|land", "fall"),
    ("crawl", "crawl"),
    ("throw|catch|pick|drop", "object"),
]


def themes_of(name: str) -> list[str]:
    lowered = name.lower()
    hits = [tag for pattern, tag in COVERAGE_THEMES if re.search(pattern, lowered)]
    return hits or ["other"]


def log_category_coverage(records: list[dict]) -> None:
    """报告每个动作类别被选中的比例，零覆盖的类别会被显式警告。"""
    total: dict[str, int] = {}
    kept: dict[str, int] = {}
    for record in records:
        for tag in themes_of(record["name"]):
            total[tag] = total.get(tag, 0) + 1
            if record.get("keep"):
                kept[tag] = kept.get(tag, 0) + 1

    log("Category coverage (kept/total):")
    for tag in sorted(total, key=lambda t: -total[t]):
        got = kept.get(tag, 0)
        log(f"  {tag:10} {got:4}/{total[tag]:4}  {got / total[tag]:6.1%}")
    empty = [t for t in total if kept.get(t, 0) == 0]
    if empty:
        log(f"  WARNING zero-coverage categories: {', '.join(sorted(empty))}")


def log_export_estimate(records: list[dict], frame_gap: int = 20) -> None:
    """按保留片段的帧数估算下游导出规模。"""
    kept = [r for r in records if r.get("keep")]
    lengths = [
        int(r.get("frame_end", 0)) - int(r.get("frame_start", 0)) + 1 for r in kept
    ]
    if not lengths:
        return
    per_char = sum(max(1, length // frame_gap) for length in lengths)
    log(
        f"Export estimate: {len(kept)} clips, {sum(lengths)} frames, "
        f"frame_gap={frame_gap} -> {per_char} meshes/character"
    )


def resolve_min_score(
    records: list[dict], metric: str, min_score: float | None, percentile: float
) -> float:
    if min_score is not None:
        return float(min_score)
    scores = [float(r.get(metric, 0.0)) for r in records]
    if not scores:
        return 0.0
    return float(np.percentile(scores, percentile))


def write_outputs(
    output_dir: Path,
    records: list[dict],
    metric: str,
    epsilon: float,
    min_score: float,
    max_keep: int,
    base_mesh_fbx: Path | None,
    animations_dir: Path | None,
    select_mode: str = DEFAULT_SELECT_MODE,
    keep_all_above: float | None = None,
    tiers: str | None = None,
    min_zip_coverage: int | None = None,
) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    kept = [r for r in records if r.get("keep")]

    payload = {
        "config": {
            "select_mode": select_mode,
            "metric": metric,
            "epsilon": epsilon,
            "min_score": min_score,
            "max_keep": max_keep,
            "keep_all_above": keep_all_above,
            "tiers": tiers,
            "min_zip_coverage": min_zip_coverage,
            "vertex_stride": VERTEX_STRIDE,
            "sample_frames": SAMPLE_FRAMES,
            "align": ALIGN_MODE,
            "base_mesh_fbx": str(base_mesh_fbx) if base_mesh_fbx else None,
            "animations_dir": str(animations_dir) if animations_dir else None,
        },
        "summary": {
            "total": len(records),
            "kept": len(kept),
            "dropped": len(records) - len(kept),
        },
        "items": records,
    }
    with (output_dir / SCORES_JSON_NAME).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    fieldnames = [
        "name",
        "file",
        "keep",
        "skip_reason",
        metric,
        "rms_max",
        "rms_mean",
        "rms_end",
        "p95_max",
        "peak_frame",
        "frame_start",
        "frame_end",
        "bone_overlap",
        "bones_renamed",
    ]
    seen: set[str] = set()
    fieldnames = [f for f in fieldnames if not (f in seen or seen.add(f))]
    with (output_dir / SCORES_CSV_NAME).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    with (output_dir / SELECTED_TXT_NAME).open("w", encoding="utf-8") as f:
        for record in sorted(kept, key=lambda r: r["name"].lower()):
            f.write(f"{record['file']}\n")

    return len(kept), len(records)


def log_score_distribution(records: list[dict], metric: str) -> None:
    scores = np.array([float(r.get(metric, 0.0)) for r in records])
    if scores.size == 0:
        return
    quantiles = [0, 10, 25, 50, 75, 90, 100]
    parts = [f"p{q}={np.percentile(scores, q):.4f}" for q in quantiles]
    log(f"Score distribution ({metric}): " + ", ".join(parts))


# =========================
# 入口
# =========================
def apply_selection(
    records: list[dict],
    output_dir: Path,
    metric: str,
    select_mode: str,
    epsilon: float,
    min_score: float | None,
    min_score_percentile: float,
    max_keep: int,
    keep_all_above: float,
    tiers_spec: str,
    twin_tol: float,
    base_mesh_fbx,
    animations_dir,
    coverage: dict[str, int] | None = None,
    min_zip_coverage: int = DEFAULT_MIN_ZIP_COVERAGE,
) -> tuple[int, int]:
    """跑选择策略并落盘，同时输出类别覆盖与导出规模体检报告。"""
    log_score_distribution(records, metric)

    # 先剔除原始 zip 里不存在、或只有少数角色才有的动画，再在剩下的池子里做选择
    candidates = filter_by_zip_coverage(records, coverage or {}, min_zip_coverage)

    if select_mode == "tiered":
        log(f"select_mode=tiered, keep_all_above={keep_all_above}, tiers={tiers_spec}")
        select_tiered(
            candidates, metric, keep_all_above, parse_tiers(tiers_spec), twin_tol
        )
        resolved_min = keep_all_above
    else:
        resolved_min = resolve_min_score(
            candidates, metric, min_score, min_score_percentile
        )
        log(f"select_mode=gap, min_score={resolved_min:.4f}, epsilon={epsilon}")
        select_by_score_gap(candidates, metric, epsilon, resolved_min, max_keep)

    ranked = sorted(records, key=lambda r: r.get(metric, 0.0), reverse=True)
    log_category_coverage(ranked)
    log_export_estimate(ranked)

    return write_outputs(
        output_dir,
        ranked,
        metric,
        epsilon,
        resolved_min,
        max_keep,
        base_mesh_fbx,
        animations_dir,
        select_mode=select_mode,
        keep_all_above=keep_all_above if select_mode == "tiered" else None,
        tiers=tiers_spec if select_mode == "tiered" else None,
        min_zip_coverage=min_zip_coverage,
    )


def run_scoring(
    base_mesh_fbx: Path,
    animations_dir: Path,
    output_dir: Path,
    metric: str,
    epsilon: float,
    min_score: float | None,
    min_score_percentile: float,
    max_keep: int,
    stride: int,
    max_animations: int | None,
    sample_frames: int = SAMPLE_FRAMES,
    align_mode: str = ALIGN_MODE,
    select_mode: str = DEFAULT_SELECT_MODE,
    keep_all_above: float = DEFAULT_KEEP_ALL_ABOVE,
    tiers_spec: str = DEFAULT_TIERS,
    twin_tol: float = DEFAULT_TWIN_TOL,
    coverage_tsv: Path = DEFAULT_COVERAGE_TSV,
    min_zip_coverage: int = DEFAULT_MIN_ZIP_COVERAGE,
) -> None:
    global ALIGN_MODE
    ALIGN_MODE = align_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    global RUN_LOG_PATH
    RUN_LOG_PATH = str(output_dir / "run_log.txt")
    error_log_path = output_dir / "error_log.txt"

    log("Scoring started")
    log(f"BASE_MESH_FBX={base_mesh_fbx}")
    log(f"ANIMATIONS_DIR={animations_dir}")
    log(f"OUTPUT_DIR={output_dir}")
    log(f"metric={metric}, epsilon={epsilon}, stride={stride}, align={align_mode}")

    base_mesh_fbx = resolve_base_mesh_fbx(base_mesh_fbx)
    records = score_animation_library(
        base_mesh_fbx,
        animations_dir,
        stride,
        max_animations,
        error_log_path,
        sample_frames,
        align_mode,
    )
    if not records:
        raise RuntimeError("No animation was scored successfully.")

    kept, total = apply_selection(
        records,
        output_dir,
        metric,
        select_mode,
        epsilon,
        min_score,
        min_score_percentile,
        max_keep,
        keep_all_above,
        tiers_spec,
        twin_tol,
        base_mesh_fbx,
        animations_dir,
        load_zip_coverage(Path(coverage_tsv)),
        min_zip_coverage,
    )

    summary = f"Scoring done: kept {kept}/{total} animations -> {output_dir}"
    log(summary)
    show_popup("Animation Scorer", summary, icon="INFO")


def run_selection_only(
    output_dir: Path,
    metric: str,
    epsilon: float,
    min_score: float | None,
    min_score_percentile: float,
    max_keep: int,
    select_mode: str = DEFAULT_SELECT_MODE,
    keep_all_above: float = DEFAULT_KEEP_ALL_ABOVE,
    tiers_spec: str = DEFAULT_TIERS,
    twin_tol: float = DEFAULT_TWIN_TOL,
    coverage_tsv: Path = DEFAULT_COVERAGE_TSV,
    min_zip_coverage: int = DEFAULT_MIN_ZIP_COVERAGE,
) -> None:
    """读取已有 anim_scores.json，仅重跑筛选（不启动 Blender 导入）。"""
    global RUN_LOG_PATH
    RUN_LOG_PATH = str(output_dir / "run_log.txt")

    scores_path = output_dir / SCORES_JSON_NAME
    if not scores_path.is_file():
        raise FileNotFoundError(f"Scores file not found: {scores_path}")

    with scores_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    records = payload.get("items", [])
    if not records:
        raise RuntimeError(f"No scored items in: {scores_path}")

    config = payload.get("config", {})
    log(f"Re-selecting from {scores_path} ({len(records)} items)")

    kept, total = apply_selection(
        records,
        output_dir,
        metric,
        select_mode,
        epsilon,
        min_score,
        min_score_percentile,
        max_keep,
        keep_all_above,
        tiers_spec,
        twin_tol,
        config.get("base_mesh_fbx"),
        config.get("animations_dir"),
        load_zip_coverage(Path(coverage_tsv)),
        min_zip_coverage,
    )

    summary = f"Re-selection done: kept {kept}/{total} animations -> {output_dir}"
    log(summary)
    show_popup("Animation Scorer", summary, icon="INFO")


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []

    parser = argparse.ArgumentParser(
        description="Score animations by end-pose deformation and drop near-duplicates"
    )
    parser.add_argument("--base_mesh_fbx", default=str(DEFAULT_BASE_MESH_FBX))
    parser.add_argument("--animations_dir", default=str(DEFAULT_ANIMATIONS_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        choices=("rms_max", "rms_mean", "rms_end", "p95_max", "max_max"),
        help="Score used for selection",
    )
    parser.add_argument(
        "--sample_frames",
        type=int,
        default=SAMPLE_FRAMES,
        help="Frames sampled per animation for scoring",
    )
    parser.add_argument(
        "--align",
        default=ALIGN_MODE,
        choices=("rigid", "centroid", "none"),
        help="Remove global motion before measuring deformation",
    )
    parser.add_argument(
        "--select_mode",
        default=DEFAULT_SELECT_MODE,
        choices=("tiered", "gap"),
        help="tiered: keep all high-deformation clips, sample lower bands by quota. "
        "gap: legacy global epsilon dedup",
    )
    parser.add_argument(
        "--keep_all_above",
        type=float,
        default=DEFAULT_KEEP_ALL_ABOVE,
        help="tiered mode: score above which every clip is kept (minus twins)",
    )
    parser.add_argument(
        "--tiers",
        default=DEFAULT_TIERS,
        help="tiered mode: comma-separated 'low:quota' bands, highest first",
    )
    parser.add_argument(
        "--twin_tol",
        type=float,
        default=DEFAULT_TWIN_TOL,
        help="tiered mode: score gap below which same-named clips count as duplicates",
    )
    parser.add_argument(
        "--coverage_tsv",
        default=str(DEFAULT_COVERAGE_TSV),
        help="TSV of 'animation filename<TAB>number of character zips containing it'",
    )
    parser.add_argument(
        "--min_zip_coverage",
        type=int,
        default=DEFAULT_MIN_ZIP_COVERAGE,
        help="Drop animations present in fewer than this many character zips",
    )
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument(
        "--min_score",
        type=float,
        default=None,
        help="Absolute score floor; overrides --min_score_percentile",
    )
    parser.add_argument(
        "--min_score_percentile", type=float, default=DEFAULT_MIN_SCORE_PERCENTILE
    )
    parser.add_argument("--max_keep", type=int, default=DEFAULT_MAX_KEEP)
    parser.add_argument("--vertex_stride", type=int, default=VERTEX_STRIDE)
    parser.add_argument("--max_animations", type=int, default=None)
    parser.add_argument(
        "--select_only",
        action="store_true",
        help="Skip Blender scoring; re-run selection from existing anim_scores.json",
    )
    return parser.parse_args(argv)


def main_cli() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if args.select_only:
        run_selection_only(
            output_dir=output_dir,
            metric=args.metric,
            epsilon=args.epsilon,
            min_score=args.min_score,
            min_score_percentile=args.min_score_percentile,
            max_keep=args.max_keep,
            select_mode=args.select_mode,
            keep_all_above=args.keep_all_above,
            tiers_spec=args.tiers,
            twin_tol=args.twin_tol,
            coverage_tsv=Path(args.coverage_tsv),
            min_zip_coverage=args.min_zip_coverage,
        )
        return

    run_scoring(
        base_mesh_fbx=Path(args.base_mesh_fbx).expanduser().resolve(),
        animations_dir=Path(args.animations_dir).expanduser().resolve(),
        output_dir=output_dir,
        metric=args.metric,
        epsilon=args.epsilon,
        min_score=args.min_score,
        min_score_percentile=args.min_score_percentile,
        max_keep=args.max_keep,
        stride=max(1, int(args.vertex_stride)),
        max_animations=args.max_animations,
        sample_frames=max(1, int(args.sample_frames)),
        align_mode=args.align,
        select_mode=args.select_mode,
        keep_all_above=args.keep_all_above,
        tiers_spec=args.tiers,
        twin_tol=args.twin_tol,
        coverage_tsv=Path(args.coverage_tsv),
        min_zip_coverage=args.min_zip_coverage,
    )


def main() -> None:
    base_mesh_fbx = prompt_path(
        "base_mesh_fbx (file or character dir)",
        DEFAULT_BASE_MESH_FBX.expanduser().resolve(),
    )
    animations_dir = prompt_path(
        "animations_dir", DEFAULT_ANIMATIONS_DIR.expanduser().resolve()
    )
    output_dir = prompt_path(
        "output_dir", DEFAULT_OUTPUT_DIR.expanduser().resolve(), create_dir=True
    )
    run_scoring(
        base_mesh_fbx=base_mesh_fbx,
        animations_dir=animations_dir,
        output_dir=output_dir,
        metric=DEFAULT_METRIC,
        epsilon=DEFAULT_EPSILON,
        min_score=None,
        min_score_percentile=DEFAULT_MIN_SCORE_PERCENTILE,
        max_keep=DEFAULT_MAX_KEEP,
        stride=VERTEX_STRIDE,
        max_animations=None,
        sample_frames=SAMPLE_FRAMES,
        align_mode=ALIGN_MODE,
    )


def run_entrypoint() -> None:
    if "--" in sys.argv:
        main_cli()
    else:
        main()


# 兼容 Blender Development 插件 runpy（会注入 CTX）
if __name__ == "__main__" or "CTX" in globals():
    try:
        run_entrypoint()
    except Exception as exc:
        log(f"Fatal error: {exc}")
        traceback.print_exc()
        show_popup("Animation Scorer", f"Error:\n{exc}", icon="ERROR")
        raise
