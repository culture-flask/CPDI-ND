# CPDI-ND 使用指南

---

## 1. 模块总览

| 文件 | 类型 | 作用 |
|:--|:--|:--|
| `speck.py` | 库 | Speck32/64 加密实现、密钥扩展、差分样本生成（`make_train_data` / `make_test_data` 等） |
| `cpdi_nd_model.py` | 库 | 模型定义：`CipherPairEncoder`、`CPDI_ND`（含可选自注意力） |
| `cpdi_nd_train.py` | 库 | 训练层：`ContrastiveLoss`、`DiversityRegularization`、`cyclic_lr`、`CPDI_ND_Dataset`、`train_one_epoch`、`evaluate`、`train_cpdi_nd` |
| `main.py` | 入口 | 7/8 轮（任意轮数）训练主入口：数据派生、多种子、训练指标汇总（不做自动测试） |
| `model_9_round_train.py` | 入口 | 9 轮三阶段训练（必须从 8 轮模型迁移） |
| `model_wide_noattn_train.py` | 入口 | 参数量对齐的加宽无注意力基线（对照实验） |
| `batch_test_models.py` | 入口 | 批量测试已保存的模型，输出 CSV/JSON + 控制台透视表 |
| `key_recovery.py` | 入口 | 基于 n 轮区分器的 (n+2) 轮密钥恢复攻击成功率评估 |

---

## 2. 环境准备

依赖库（Python 3.8+）：

- `numpy`
- `pandas`
- `torch`（建议 CUDA 版；CPU 版也能跑，但训练/测试会非常慢）

安装：

```bash
pip install numpy pandas
# PyTorch 请按本机 CUDA 版本选择对应安装命令，例如 CUDA 11.8：
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

验证 GPU 是否可用：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

> 代码中各脚本的 `--gpu` 默认值不同（`main.py` 为 `"2"`，`model_9_round_train.py` 为 `"0"`，`model_wide_noattn_train.py` 为 `"4"`，`key_recovery.py` 为 `"2"`）。
> 请按实际空闲卡显式传 `--gpu`，避免跑到错误设备上。

---

## 3. main.py：训练 → 验证 → 汇总

### 3.1 最简用法

```bash
# 默认：exp1、差分 0x0040/0x0000、7 轮、k=32、单种子 → result/0040
python main.py --gpu 0
```

### 3.2 标准实验命令（8 轮 exp2、3 个固定种子）

```bash
python main.py \
    --mode exp2 \
    --diff 0x2800,0x0010 \
    --num-rounds 8 \
    --k 32 \
    --train-samples 1000000 \
    --val-samples 100000 \
    --epochs 50 \
    --batch-size 2000 \
    --num-seeds 3 \
    --seeds 42,43,44 \
    --gpu 0
```

一次运行自动完成：

1. **逐种子串行训练**（42 → 43 → 44），每个 epoch 用验证集选优保存模型；
2. **汇总**：写 `summary.txt`（每个种子的训练指标 + 平均值/标准差/方差）。

### 3.3 全部参数

| 参数 | 默认 | 说明 |
|:--|:--|:--|
| `--diff` | `0x0040,0x0000` | 正样本目标差分，格式 `hex_high,hex_low` |
| `--num-rounds` | 7 | Speck 加密轮数（9 轮请用 `model_9_round_train.py`） |
| `--k` | 32 | 每个样本的密文对数量（训练用） |
| `--train-samples` | 1000000 | 训练样本数（正负各半） |
| `--val-samples` | 100000 | 验证样本数（每 epoch 验证、选最优） |
| `--batch-size` | 2000 | 批次大小 |
| `--epochs` | 50 | 训练轮数（循环学习率 1e-3→1e-4，周期 10；AdamW） |
| `--online` | 关 | 在线生成训练数据（大样本防内存溢出） |
| `--batch-gen-size` | 2000 | 在线模式下每次生成的样本数 |
| `--mode` | exp1 | `exp1`=纯 BCE；`exp2`=BCE+对比损失；`exp3`=BCE+对比+多样性；`exp4`=BCE+多样性 |
| `--no-attention` | 关 | 移除自注意力模块 |
| `--no-contrastive` | 关 | 在 exp2/exp3 中禁用对比损失 |
| `--no-diversity` | 关 | 关闭多样性正则 |
| `--gpu` | `"2"` | GPU 设备 ID；传空字符串使用 CPU |
| `--save-dir` | 空 → 自动 | 默认按差分命名：`result/<差分高16位>` |
| `--num-seeds` | 1 | 随机种子重复训练次数 |
| `--seeds` | 空 → 随机 | 显式种子（逗号分隔）。**数量必须与 `--num-seeds` 一致**，否则拒绝运行 |

### 3.4 输出结构

```
result/2800/                        ← 差分目录（--save-dir）
└── 8_k32_exp2/                     ← <轮数>_k<k>_<mode>[_no-attention]
    ├── model_42.pth                ← 每个种子的最优模型（按验证集选优）
    ├── log_42.csv                  ← 训练日志（epoch, train_loss, train_acc, val_loss, val_acc）
    ├── model_43.pth / log_43.csv
    ├── model_44.pth / log_44.csv
    └── summary.txt                 ← 训练指标汇总（每次运行整体重写）
```

目录命名规则：

- 有注意力（默认）：`<轮数>_k<k>_<mode>`，如 `8_k32_exp2`
- 无注意力（`--no-attention`）：追加 `_no-attention`，如 `8_k32_exp2_no-attention`
- 9 轮三阶段：`9_k<k>_<mode>[_no-attention]`
- 加宽无注意力基线：`<轮数>_k<k>_wide-<mode>`

**注意**：`summary.txt` 只反映**本次调用**的种子。要累积 5 个种子的统计，须在一次调用里给 5 个种子；分多次跑会导致后一次的 summary 覆盖前一次（模型与日志仍按种子保留）。

---

## 4. 复现性说明

| 项目 | 机制 |
|:--|:--|
| 训练种子 | `--seeds` 显式指定 → 完全可复现；不指定 → 随机抽取，实际值记录在文件名与 summary 中 |
| 训练/验证数据 | 由训练种子确定性派生：全量模式 `rng=[种子, 标签]` 一次性生成；在线模式按 `(种子, epoch, worker)` 派生，每个 epoch 数据全新且同种子跨运行逐位一致 |
| 参数初始化 / 洗牌 | 种子控制 + `cudnn.deterministic=True` |
| 测试集 | 数据种子固定为 `TEST_DATA_SEED = 20241016`（定义在 `main.py`，由 `generate_test_set` 使用）：同一（轮数, 差分, K）下测试集逐比特一致，供 `batch_test_models.py` 调用 |
| 测试协议 | 测试差分/轮数与训练相同；K 扫描 {4,...,256}；正负各半；阈值 0.5 |

---

## 5. 常用实验场景

```bash
# ① 论文主结果：8 轮 exp2，3 种子（训练后另用 batch_test_models.py 做 K 泛化测试）
python main.py --mode exp2 --diff 0x2800,0x0010 --num-rounds 8 \
    --num-seeds 5 --gpu 0 --online        # 可通过--seeds s1,s2,s3,s4,s5指定种子训练

# ② 损失函数消融（exp1–exp4；exp2 与 ① 共用）
python main.py --mode exp1 --diff 0x2800,0x0010 --num-rounds 8 \
    --num-seeds 5 --gpu 0 --online        # exp3/exp4 同理改 --mode

# ③ 无注意力消融
python main.py --mode exp2 --diff 0x2800,0x0010 --num-rounds 8 \
    --no-attention --num-seeds 5 --gpu 0 --online
```

---

## 6. model_9_round_train.py：9 轮三阶段训练

9 轮无法端到端直接训练，必须从 8 轮模型迁移。先确认预训练模型存在：

```bash
ls result/2800/8_k32_exp2/*.pth
```

> `--pretrained-8round-path` 必须指向**实际存在的文件**，且 `--k` 要与该模型的训练 k 一致
> （`8_k32_exp2` → `--k 32`）。

完整命令：

```bash
python model_9_round_train.py \
    --mode exp2 \
    --diff 0x2800,0x0010 \
    --stage1-diff 0x8000,0x840a \
    --k 32 \
    --pretrained-8round-path result/2800/8_k32_exp2/model_2060699015.pth \
    --val-samples 100000 \
    --test-samples 100000 \
    --stage1-rounds 5 \
    --stage1-samples 1000000 \
    --stage1-epochs 10 \
    --stage1-batch 5000 \
    --stage1-lr 1e-4 \
    --stage2-samples 5000000 \
    --stage2-epochs 50 \
    --stage2-batch 5000 \
    --stage2-lr 1e-4 \
    --stage3-lr-list 1e-5 1e-6 \
    --stage3-samples-per-step 5000000 \
    --stage3-batch 5000 \
    --batch-size 2000 \
    --num-seeds 3 \
    --seeds 42,43,44 \
    --save-dir result/2800 \
    --gpu 0
```

关键参数：

| 参数 | 默认 / 取值 | 说明 |
|:--|:--|:--|
| `--diff` | `0x2800,0x0010` | 9 轮目标任务差分 |
| `--stage1-diff` | `0x8000,0x840a` | 阶段1 源任务差分（目标差分经 4 轮传播后的最优输出差分） |
| `--stage1-rounds` | 5 | 阶段1 的 Speck 轮数（9 − 4 = 5） |
| `--pretrained-8round-path` | 必填 | 8 轮预训练模型路径 |
| `--stage1-model-path` / `--skip-stage1` | — | 跳过阶段1、从已有阶段1模型续跑 |
| `--stage1-samples/-epochs/-batch/-lr` | 1e6 / 10 / 5000 / 1e-4 | 阶段1：8 轮模型定向微调 |
| `--stage2-samples/-epochs/-batch/-lr` | 5e6 / 50 / 5000 / 1e-4 | 阶段2：9 轮任务大规模初始化 |
| `--stage3-lr-list` | `1e-5 1e-6` | 阶段3 阶梯式学习率（空格分隔，`nargs='+'`） |
| `--stage3-samples-per-step` | 5e6 | 阶段3 每步样本数 |
| `--val-samples` / `--test-samples` | 100000 | 验证 / 每个测试 K 的样本数 |
| `--batch-size` | 2000 | 验证/测试批次（训练批次由各阶段 `-batch` 控制） |
| `--num-seeds` + `--seeds` | 1 / 随机 | 数量必须一致 |

输出（`result/2800/9_k32_exp2/`）：

```
model_seed.pth              ← 全局最优模型（K 泛化测试用它）
model_seed_stage1.pth       ← 阶段1结束模型（可用于 --skip-stage1 续跑）
model_seed_stage2.pth       ← 阶段2结束模型
log_seed.csv                ← 三阶段合并日志（stage,epoch,lr,train_loss,acc,val_loss,acc）
summary.txt               ← 训练表 + K 泛化三矩阵 + 均值/方差
```

命令：

```bash
# 完整训练
python model_9_round_train.py --mode exp2 --diff 0x2800,0x0010 --stage1-diff 0x8000,0x840a --k 64 --pretrained-8round-path result/2800/8_k64_exp2_2/model_566467815.pth --val-samples 100000 --test-samples 100000 --stage1-rounds 5 --stage1-samples 1000000 --stage1-epochs 10 --stage1-batch 5000 --stage1-lr 1e-4 --stage2-samples 5000000 --stage2-epochs 50 --stage2-batch 5000 --stage2-lr 1e-4 --stage3-lr-list 1e-5 1e-6 --stage3-samples-per-step 5000000 --stage3-batch 5000 --batch-size 2000 --num-seeds 5  --gpu 0
```

---

## 7. model_wide_noattn_train.py：参数量对齐的加宽无注意力基线

用途：排除“性能提升来自参数量/网络深度”的替代解释。该基线**参数量与完整模型对齐**、且编码器更深（对基线更有利的保守设置）。

| 配置 | num_filters | enc_blocks | 分类头 | 参数量 | 偏差 |
|:--|:-:|:-:|:-:|:-:|:-:|
| 完整模型（目标） | 32 | 3（+3 层注意力） | 128 | **121,985** | — |
| 原无注意力基线 | 32 | 3 | 128 | 71,681 | -41.2% |
| **本脚本默认** | **48** | **5** | **224** | **121,153** | **-0.68%** |
| 最严格对齐（可选） | 48 | 4 | 256 | 121,761 | -0.18% |

```bash
# 查看参数量对齐候选配置（不训练）
python model_wide_noattn_train.py --search-params

# 完整对照实验
python model_wide_noattn_train.py \
    --mode exp2 --diff 0x2800,0x0010 --num-rounds 8 --k 32 \
    --train-samples 1000000 --val-samples 100000 --test-samples 100000 \
    --epochs 50 --batch-size 2000 \
    --num-seeds 5 --gpu 0
```

关键参数：

| 参数 | 默认 | 说明 |
|:--|:--|:--|
| `--num-filters` | 48 | 编码器通道数（完整模型为 32） |
| `--enc-blocks` | 5 | 编码器 InceptionResBlock 层数（原为 3） |
| `--hidden1` / `--hidden2` | 224 / 64 | 分类头两层宽度（原为 128 / 64） |
| `--search-params` | 关 | 只打印参数量对齐候选表并退出 |
| `--online` / `--batch-gen-size` | 关 / 2000 | 同 `main.py` |
| `--gpu` | `"4"` | 默认 4 号卡 |

产物目录：`result/2800/8_k32_wide-exp2/`（与 `8_k32_exp2/` 并列），含 `model_<种子>.pth`、`log_<种子>.csv`、`summary.txt`（头部额外记录结构与参数量对齐信息，K 泛化矩阵格式与 `main.py` 完全一致）。

---

## 8. batch_test_models.py：批量测试已保存的模型

只做测试与结果输出。输出独立保存在 `result/batch_test/<时间戳>/`。

```bash
# 测 result/2800 下所有目录，每目录取 1 个模型（默认）
python batch_test_models.py --root result/2800

# 只测指定目录（子串匹配），每目录取 3 个种子
python batch_test_models.py --root result/2800 --only "8_k32_exp2,8_k32_exp2_no-attention" \
    --limit-per-dir 3

# 直接指定模型文件
python batch_test_models.py \
    --models result/2800/8_k32_exp2/model_1612946774.pth,result/2800/8_k32_exp2_no-attention/model_2088613111.pth

# 自定义测试 K 与每个 K 的样本数
python batch_test_models.py --root result/2800 --k-list 8,32,128 --test-samples 50000
```

| 参数 | 默认 | 说明 |
|:--|:--|:--|
| `--root` | `result/2800` | 模型根目录（按 `<差分>/<轮数>_k<k>_<模式>[_no-attention]/model_<种子>.pth` 递归查找） |
| `--only` | 空 | 只测目录名包含这些子串的目录（逗号分隔） |
| `--models` | 空 | 直接指定模型文件（指定后忽略 `--root`/`--only`） |
| `--limit-per-dir` | 1 | 每个目录最多测几个模型（0 = 全部） |
| `--require-log` | 关 | 只测有配套 `log_<种子>.csv` 的模型（排除训练中断产物） |
| `--k-list` | `4,8,16,32,64,128` | 测试 K 列表 |
| `--test-samples` | 100000 | 每个测试 K 的样本数（正负各半） |
| `--batch-size` | 2048 | 前向批大小 |
| `--diff` | 空 | 显式覆盖差分；默认从目录名解析（已登记 `0x2800/0x0010`、`0x0040/0x0000`） |
| `--attn-layers` | 3 | 有注意力模型的注意力层数 |
| `--wide-filters` / `--wide-blocks` / `--wide-hidden1` | 48 / 5 / 224 | 加载 `wide-*` 模型的结构参数 |
| `--gpu` / `--output-root` | `"0"` / `result/batch_test` | 设备 ID / 输出根目录 |

输出：

```
result/batch_test/<时间戳>/results.csv   ← 每模型 × 每 K 的 acc/TPR/TNR
result/batch_test/<时间戳>/results.json  ← 同上 + 配置（k_list、test_samples、data_seed）
```

同时控制台按模型打印 acc/TPR/TNR 透视表。测试集由 `main.generate_test_set` 按固定数据种子现场生成，同一（轮数, 差分, K）跨运行一致。

---

## 9. key_recovery.py：密钥恢复攻击

基于 n 轮区分器的 1+n+1 结构：猜测最后一轮子密钥，剥掉一轮后交给区分器打分，取分数最高的猜测密钥；正确密钥排名 ≤ `--top-k` 记为一次成功。

```bash
# 8 轮区分器 → 10 轮密钥恢复，10 组密文对，重复 5 次
python key_recovery.py --model-path result/2800/8_k32_exp2/model_2060699015.pth \
    --num-rounds 8 --c 10 --num-trials 5 --gpu 0
```

| 参数 | 默认 | 说明 |
|:--|:--|:--|
| `--model-path` | 必填 | 区分器模型路径 |
| `--num-rounds` | 8 | 区分器轮数（实际攻击轮数 = `num_rounds + 2`） |
| `--c` | 10 | 同一次攻击中使用的 k 组数（共享同一密钥） |
| `--num-trials` | 5 | 独立实验次数（每次不同随机密钥） |
| `--k` | 32 | 每组密文对数量（需与模型训练 k 一致） |
| `--diff` | `0x2800,0x0010` | 目标差分 |
| `--top-k` | 10 | 正确密钥排名 ≤ top-k 视为攻击成功 |
| `--no-attention` | 关 | 加载无注意力模型时使用 |
| `--d-model` / `--n-heads` / `--dropout` / `--enc-blocks` / `--num-filters` | 64 / 4 / 0.3 / 3 / 32 | 模型结构（需与 checkpoint 一致） |
| `--batch-size` | 16384 | 前向批大小 |
| `--log-dir` | `result/key_recovery_logs` | 攻击日志目录（输出 `k<k>_attack_log_<时间戳>.log`） |

---

## 10. 常见问题

| 现象 | 原因 / 解决 |
|:--|:--|
| `--seeds 给定了 N 个种子，与 --num-seeds=M 数量不一致，拒绝运行` | 设计行为：两者必须匹配 |
| 跑在错误 GPU 上 / 找不到 GPU | 忘传 `--gpu`（各脚本默认值见 §2，可能不是 0） |
| 9 轮直接训练准确率 50% | 正常：9 轮需用 `model_9_round_train.py` 三阶段训练 |
| 想让 summary 含更多种子 | summary 只含本次调用的种子；把所有种子放进一次调用重跑 |
| 旧模型加载报 size mismatch | 检查模型是否用 `--no-attention` 训练（加载时结构必须一致；`wide-*` 还需一致的 `num_filters/enc_blocks/hidden1`） |
| `batch_test_models.py` 提示“未登记的差分目录” | 新增差分需先在 `DIFF_HIGH_MAP` 登记，或用 `--diff` 显式指定 |
