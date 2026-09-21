"""按选中的动画列表，从角色 zip 中解压出精简数据集。

每个角色 zip 里既有该角色的全部动画，也有一个角色模型 FBX。模型文件不能靠体积
判断（Big Vegas.fbx 只有 3.6MB，比不少动画还小），但它的文件名只在自己那个 zip
里出现，而动画名是跨角色共享的 —— 用这个特征即可稳定识别。

输出结构：
    <output_dir>/<角色名>/<角色名>.fbx      角色模型（含蒙皮网格）
    <output_dir>/<角色名>/animations/*.fbx  选中的动画

用法：
    python3 build_selected_dataset.py \
        --zip_root  .../Mixamo_Data/ModelsWithAnimation \
        --selected  .../Data/animation_selection/selected_anims.txt \
        --output    .../Mixamo_Data/ModelWithAnimationSelected
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_ZIP_ROOT = PROJECT_ROOT / "Mixamo_Data" / "ModelsWithAnimation"
DEFAULT_SELECTED = PROJECT_ROOT / "Data" / "animation_selection" / "selected_anims.txt"
DEFAULT_OUTPUT = PROJECT_ROOT / "Mixamo_Data" / "ModelWithAnimationSelected"

# 只有这些子目录里的 zip 是角色包；0ef/ 里的 *_Pack.zip 是动画包，不是角色
BATCH_DIRS = ("1-36", "37-72", "73-90", "91-108")

# FBX 二进制里出现该标记说明含多边形网格；纯骨架文件没有
MESH_MARKER = b"PolygonVertexIndex"


def log(msg: str) -> None:
    print(f"[BuildDataset] {msg}", flush=True)


def collect_character_zips(zip_root: Path) -> dict[str, Path]:
    """按文件名去重收集角色 zip（同一个 zip 在多个批次目录里重复出现过）。"""
    found: dict[str, Path] = {}
    for batch in BATCH_DIRS:
        batch_dir = zip_root / batch
        if not batch_dir.is_dir():
            continue
        for path in sorted(batch_dir.glob("*.zip")):
            found.setdefault(path.name, path)
    return found


def list_fbx(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        return [
            n for n in zf.namelist()
            if n.lower().endswith(".fbx") and not n.startswith("__MACOSX")
        ]


def find_character_models(zips: dict[str, Path]) -> dict[str, list[str]]:
    """角色模型 = 只在自己那个 zip 里出现的文件名。"""
    listings = {name: list_fbx(path) for name, path in zips.items()}
    freq = Counter(n for names in listings.values() for n in set(names))
    unique_only = {n for n, c in freq.items() if c == 1}
    return {
        name: sorted(n for n in set(names) if n in unique_only)
        for name, names in listings.items()
    }


def has_mesh(zip_path: Path, member: str) -> bool:
    """流式读取，检测 FBX 里是否真的有网格数据。"""
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as f:
        tail = b""
        while chunk := f.read(1 << 20):
            if MESH_MARKER in tail + chunk:
                return True
            tail = chunk[-len(MESH_MARKER):]
    return False


def pick_model(zip_path: Path, candidates: list[str]) -> str | None:
    """多个独占文件时挑真正带网格的那个。"""
    for member in candidates:
        if has_mesh(zip_path, member):
            return member
    return None


def build(zip_root: Path, selected_file: Path, output_dir: Path, overwrite: bool) -> None:
    selected = [
        line.strip() for line in selected_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not selected:
        raise RuntimeError(f"No animations listed in {selected_file}")
    log(f"Selected animations: {len(selected)}")

    zips = collect_character_zips(zip_root)
    log(f"Character zips: {len(zips)}")
    if not zips:
        raise RuntimeError(f"No character zips under {zip_root}")

    log("Identifying character models (unique-filename scan)...")
    unique_names = find_character_models(zips)

    output_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(selected)
    report: list[dict] = []
    skipped: list[dict] = []

    for idx, (zip_name, zip_path) in enumerate(sorted(zips.items()), 1):
        zip_id = zip_path.stem
        candidates = unique_names.get(zip_name, [])
        if not candidates:
            skipped.append({"zip": zip_id, "reason": "no_unique_filename"})
            log(f"[{idx}/{len(zips)}] {zip_id}: no character model found, skipped")
            continue

        model = pick_model(zip_path, candidates)
        if model is None:
            skipped.append(
                {"zip": zip_id, "reason": "model_has_no_mesh", "candidates": candidates}
            )
            log(f"[{idx}/{len(zips)}] {zip_id}: {candidates} has no mesh, skipped")
            continue

        char_name = Path(model).stem
        char_dir = output_dir / char_name
        if char_dir.exists():
            if not overwrite:
                log(f"[{idx}/{len(zips)}] {char_name}: exists, skipped")
                continue
            shutil.rmtree(char_dir)
        anim_dir = char_dir / "animations"
        anim_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path) as zf:
            members = set(zf.namelist())
            with zf.open(model) as src, (char_dir / Path(model).name).open("wb") as dst:
                shutil.copyfileobj(src, dst)

            present = [a for a in selected if a in members]
            for anim in present:
                with zf.open(anim) as src, (anim_dir / Path(anim).name).open("wb") as dst:
                    shutil.copyfileobj(src, dst)

        missing = sorted(wanted - set(present))
        report.append(
            {
                "zip_id": zip_id,
                "character": char_name,
                "model_file": model,
                "animations": len(present),
                "missing": missing,
            }
        )
        log(
            f"[{idx}/{len(zips)}] {char_name}: model + {len(present)}/{len(selected)} "
            f"animations" + (f" ({len(missing)} missing)" if missing else "")
        )

    manifest = {
        "selected_animations": len(selected),
        "characters_built": len(report),
        "characters_skipped": skipped,
        "items": report,
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    total = sum(r["animations"] for r in report)
    log(f"Done: {len(report)} characters, {total} animation files, {len(skipped)} skipped")
    log(f"Manifest -> {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip_root", default=str(DEFAULT_ZIP_ROOT))
    parser.add_argument("--selected", default=str(DEFAULT_SELECTED))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild character folders that already exist",
    )
    args = parser.parse_args()

    build(
        Path(args.zip_root).expanduser().resolve(),
        Path(args.selected).expanduser().resolve(),
        Path(args.output).expanduser().resolve(),
        args.overwrite,
    )


if __name__ == "__main__":
    sys.exit(main())
