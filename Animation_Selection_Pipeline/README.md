# Mesh_Clean_To_Dirty（动画筛选）

本目录只放 **动画筛选 / 组包** 脚本。  
clean → dirty 请用 [`../Animation_Mesh_Pipeline/`](../Animation_Mesh_Pipeline/)。

```
Mesh_Clean_To_Dirty/
├── score_animations_by_deformation.py  # ① ~2000 → ~200：形变打分 + 去重筛选
├── build_selected_dataset.py           # ② 按 selected 列表从角色 zip 解压精简包
└── README.md
```

---

## 流程

```
动画库 (~2000 FBX)
  → ① score_animations_by_deformation.py
      → Data/animation_selection/anim_scores.json
      → Data/animation_selection/selected_anims.txt  (~200)
  → ② build_selected_dataset.py
      → Mixamo_Data/ModelWithAnimationSelected/
          <char>/<char>.fbx + animations/*.fbx
  → Animation_Mesh_Pipeline（stage1 采样 → stage2 变脏）
```

---

## ① 打分筛选

在 Blender 中运行（需 `bpy`）：

```bash
blender --background --python score_animations_by_deformation.py -- \
  --base_mesh_fbx ".../Data/input_character_fbx/Amy.fbx" \
  --animations_dir ".../Data/animation_fbx/input_animation_fbx/06f" \
  --output_dir ".../Data/animation_selection"
```

已有 `anim_scores.json` 时，可离线重筛（改 epsilon / 阈值）：

```bash
blender --background --python score_animations_by_deformation.py -- \
  --select_only --output_dir ".../Data/animation_selection" --epsilon 0.006
```

---

## ② 组精简数据集

纯 Python，不需要 Blender：

```bash
python3 build_selected_dataset.py \
  --zip_root  ".../Mixamo_Data/ModelsWithAnimation" \
  --selected  ".../Data/animation_selection/selected_anims.txt" \
  --output    ".../Mixamo_Data/ModelWithAnimationSelected"
```

输出布局与 `Animation_Mesh_Pipeline` stage1 的输入约定一致。
