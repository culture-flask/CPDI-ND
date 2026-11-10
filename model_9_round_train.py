"""
CPDI-ND 9轮神经区分器三阶段训练
======================================================================
本脚本的**训练/验证/测试/保存/复现逻辑与 main.py 完全一致**，唯一区别是
9 轮弱信号无法端到端直接训练，因此训练过程分为三个阶段：

    阶段1 ─ 8轮模型定向微调
        加载预训练好的8轮模型，在较少轮数(默认5轮)的Speck上用源差分微调，
        让模型适应新的差分模式，为9轮任务做准备。

    阶段2 ─ 9轮任务初始化训练
        在9轮Speck上用目标差分进行大规模训练（在线生成数据，无内存压力），
        让模型初步建立9轮区分能力。

    阶段3 ─ 阶梯式学习率精细化微调
        逐步降低学习率(默认1e-5 → 1e-6)，每步用全新样本训练1轮，
        精细调整决策边界，最大化9轮区分精度。

与 main.py 一致的部分：
    - 负样本始终为随机差分（与 main 相同，不提供固定差分选项）
    - 训练数据由训练种子确定性派生：在线模式按 (种子, 阶段, epoch, worker)
      派生，每个 epoch 数据全新（抗过拟合），同种子跨运行逐位一致（可复现）
    - cudnn 确定性算法（同种子可完整复现）
    - 多种子重复训练：--num-seeds / --seeds（数量必须一致）
    - 保存结构: <save_dir>/9_k<k>_<mode>/model_<种子>.pth + log_<种子>.csv
    - 训练结束自动在固定测试集上做 K∈{4,...,256} 泛化扫描，结果汇总到 summary.txt
    - 测试集由固定数据种子 TEST_DATA_SEED 生成，跨运行、跨模型完全一致

使用示例:
    # 单种子：从 8 轮模型迁移训练 9 轮区分器
    python model_9_round_train.py --mode exp2 --gpu 0 \
        --pretrained-8round-path result/2800/8_k32_exp2/model_42.pth

    # 3 个固定种子（可复现），自动 K 泛化测试
    python model_9_round_train.py --mode exp2 --gpu 0 --num-seeds 3 --seeds 42,43,44 \
        --pretrained-8round-path result/2800/8_k32_exp2/model_42.pth

    # 跳过阶段1（已有时直接进入阶段2）
    python model_9_round_train.py --mode exp2 --skip-stage1 \
        --stage1-model-path result/2800/9_k32_exp2/model_42_stage1.pth --gpu 0
"""

import os
import sys
import argparse
import time
import random
import secrets
import io
import contextlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import IterableDataset

import speck as sp
from cpdi_nd_model import CPDI_ND
import cpdi_nd_train as trainer


# ============================================================================
# 在线数据生成模块（与 main.py 一致）
# ============================================================================

class CPDI_NDIterableDataset(IterableDataset):
    """在线生成 CPDI-ND 训练数据的迭代式数据集。

    当样本量超过内存容量时（如10^7以上），无法一次性加载所有数据。
    本数据集每次迭代生成一个 mini-batch 的数据，避免内存溢出。

    负样本始终为随机差分（与 main.py 一致）。

    可复现性：指定 data_seed 时，本 worker 本 epoch 的数据由
    (data_seed, epoch, worker_id) 确定性派生 —— epoch 参与派生保证每个
    epoch 数据全新（不对固定数据集过拟合），同种子跨运行逐位一致；
    data_seed 为 None 时退回系统熵源 urandom（每次独立，不可复现）。
    """

    def __init__(self, total_samples, num_rounds, diff, k,
                 batch_gen_size=2000, data_seed=None):
        super().__init__()
        self.total_samples = total_samples
        self.num_rounds = num_rounds
        self.diff = diff
        self.k = k
        self.batch_gen_size = batch_gen_size
        self.total_steps = max(total_samples // batch_gen_size, 1)
        self.data_seed = data_seed
        self.epoch = 0

    def __iter__(self):
        # 获取当前 worker 信息，避免多 worker 重复遍历
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            per_worker = max(self.total_steps // worker_info.num_workers, 1)
            start = worker_info.id * per_worker
            end = start + per_worker
            worker_id = worker_info.id
        else:
            start, end = 0, self.total_steps
            worker_id = 0

        # (data_seed, epoch, worker_id) 派生本 worker 本 epoch 的确定性随机源
        rng = (None if self.data_seed is None
               else np.random.default_rng([self.data_seed, self.epoch, worker_id]))
        self.epoch += 1

        for step in range(start, end):
            # 静默生成数据，抑制 load_cpdi_data 的冗余打印
            with contextlib.redirect_stdout(io.StringIO()):
                pairs, labels = trainer.load_cpdi_data(
                    self.batch_gen_size, self.num_rounds,
                    diff=self.diff, k=self.k,
                    random_neg=True, rng=rng
                )
            pair_tensor = torch.tensor(pairs, dtype=torch.float32)
            label_tensor = torch.tensor(labels, dtype=torch.float32)

            for i in range(self.batch_gen_size):
                yield pair_tensor[i], label_tensor[i]


def get_cpdi_dataloader(total_samples, num_rounds, diff, k, batch_size=2000,
                        num_workers=4, batch_gen_size=2000, data_seed=None, rng=None):
    """获取 CPDI-ND 格式的 DataLoader（负样本固定为随机差分）。

    小样本量(≤5×10^5)用全量加载，大样本量用在线迭代生成（三阶段默认样本量
    均远超此阈值，因此训练数据实际走在线生成）。

    参数:
        data_seed: 在线模式的数据种子——数据按 (data_seed, epoch, worker) 确定性
                   派生，每个 epoch 全新且同种子跨运行逐位一致（可复现）。
        rng: 全量加载模式的 numpy Generator（确定性生成）；None 时用系统熵源。
    """
    if total_samples <= 500000:
        # 小样本：全量加载（DataLoader 已按批洗牌，跳过数组级冗余拷贝）
        print(f"  全量加载 {total_samples:,} 样本...")
        pairs, labels = trainer.load_cpdi_data(
            total_samples, num_rounds, diff=diff, k=k,
            random_neg=True, shuffle=False, rng=rng
        )
        dataset = trainer.CPDI_ND_Dataset(pairs, labels)
        dataloader = Data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0)
        )
    else:
        # 大样本：在线迭代生成
        print(f"  在线生成 {total_samples:,} 样本 (负样本: 随机差分)...")
        dataset = CPDI_NDIterableDataset(
            total_samples, num_rounds, diff, k,
            batch_gen_size=batch_gen_size, data_seed=data_seed
        )
        dataloader = Data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0)
        )
    return dataloader


# ============================================================================
# 训练与验证函数
# ============================================================================

def train_one_epoch_cpdi(model, dataloader, optimizer, criterion, device,
                         use_contrastive=False, use_diversity=False,
                         lam1=0.0, lam2=0.0, log_every=2000):
    """CPDI-ND 单轮训练，返回训练 loss 和准确率。"""
    model.train()
    train_loss = 0.0
    train_corrects = 0
    train_num = 0
    start_time = time.time()

    contrast_fn = trainer.ContrastiveLoss(temperature=0.1) if use_contrastive else None
    diversity_fn = trainer.DiversityRegularization() if use_diversity else None

    for step, batch in enumerate(dataloader):
        b_x, b_y = batch
        b_x, b_y = b_x.to(device), b_y.to(device)

        output, features = model(b_x, return_features=True)

        # 主损失：BCE
        bce = criterion(output, b_y[:, 1:2])
        loss = bce

        # 辅助损失
        if contrast_fn is not None:
            loss = loss + lam1 * contrast_fn(features, b_y)
        if diversity_fn is not None:
            loss = loss + lam2 * diversity_fn(features)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * b_x.size(0)
        preds = (output > 0.5).float().squeeze()
        true_labels = b_y[:, 1]
        train_corrects += (preds == true_labels).sum().item()
        train_num += b_x.size(0)

        # 进度打印（每 log_every 步或首步）
        if step == 0 or (step + 1) % log_every == 0:
            elapsed = time.time() - start_time
            print(f"  Step [{step+1}] | Loss: {loss.item():.4f} | "
                  f"Acc: {train_corrects/train_num:.4f} | {elapsed:.1f}s")

    epoch_loss = train_loss / train_num
    epoch_acc = train_corrects / train_num
    print(f"  ===== Epoch 完成 | Loss: {epoch_loss:.6f} | Acc: {epoch_acc:.4f} | "
          f"耗时: {time.time() - start_time:.2f}s =====")
    return epoch_loss, epoch_acc


def validate_cpdi(model, dataloader, criterion, device,
                  use_contrastive=False, use_diversity=False,
                  lam1=0.0, lam2=0.0):
    """CPDI-ND 验证函数，返回验证 loss 和准确率。"""
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    val_num = 0

    contrast_fn = trainer.ContrastiveLoss(temperature=0.1) if use_contrastive else None
    diversity_fn = trainer.DiversityRegularization() if use_diversity else None

    with torch.no_grad():
        for batch in dataloader:
            b_x, b_y = batch
            b_x, b_y = b_x.to(device), b_y.to(device)

            output, features = model(b_x, return_features=True)

            bce = criterion(output, b_y[:, 1:2])
            loss = bce
            if contrast_fn is not None:
                loss = loss + lam1 * contrast_fn(features, b_y)
            if diversity_fn is not None:
                loss = loss + lam2 * diversity_fn(features)

            val_loss += loss.item() * b_x.size(0)
            preds = (output > 0.5).float().squeeze()
            true_labels = b_y[:, 1]
            val_corrects += (preds == true_labels).sum().item()
            val_num += b_x.size(0)

    val_loss = val_loss / val_num
    val_acc = val_corrects / val_num
    print(f"  ===== 验证完成 | Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.4f} =====")
    return val_loss, val_acc


# ============================================================================
# 种子、测试与复现工具（与 main.py 完全一致）
# ============================================================================

def set_global_seed(seed):
    """设置全局随机种子（模型初始化、DataLoader 洗牌顺序、数据派生）。

    cudnn 确定性模式：保证同种子下训练过程逐位一致。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 测试集数据种子：固定值——所有模型共用同一测试集，且跨运行完全可复现
TEST_DATA_SEED = 20241016

# 验证集数据的派生标签（与训练种子组成 SeedSequence 熵池）
_RNG_TAG_VAL = 1_000_000_000

# 各阶段训练数据的种子偏移：data_seed = seed * 10000 + 阶段标记
# 目的：不同阶段（尤其阶段2与阶段3的首个数据块）不会因 epoch 同为 0 而生成相同数据
_STAGE_SEED_BASE = 100
_STAGE_SEED_STEP = 300


def stage_data_seed(seed, stage_id, step=0):
    """派生某阶段（某步）的训练数据种子。stage_id: 1/2/3。"""
    if stage_id == 3:
        return seed * 10000 + _STAGE_SEED_STEP + step
    return seed * 10000 + _STAGE_SEED_BASE * stage_id


def parse_diff(diff_str):
    """解析差分字符串 "hex_high,hex_low" 为 (high, low) 元组。"""
    parts = diff_str.split(',')
    return (int(parts[0], 16), int(parts[1], 16))


def generate_test_set(num_samples, num_rounds, diff, k):
    """生成固定测试集（TEST_DATA_SEED 固定 → 同配置下跨运行完全一致）。

    正样本在前、负样本在后（正负各半），顺序不打乱，便于分块统计 TPR/TNR。
    """
    rng = np.random.default_rng(TEST_DATA_SEED)
    n_pos = num_samples // 2
    n_neg = num_samples - n_pos
    pos_pairs, _, pos_y = sp.make_multi_pair_data(
        n_pos, num_rounds, diff, k=k, label=1, rng=rng)
    neg_pairs, _, _ = sp.make_multi_pair_data(
        n_neg, num_rounds, diff, k=k, label=0, rng=rng)
    neg_y = sp.one_hot_encode_binary(np.full(n_neg, 0, dtype=np.uint8))
    pairs = np.concatenate([pos_pairs, neg_pairs], axis=0)
    labels = np.concatenate([pos_y, neg_y], axis=0)
    return pairs, labels


@torch.no_grad()
def evaluate_model_on(model, pairs, labels, batch_size, device):
    """在给定测试集上评估模型，返回 (acc, TPR, TNR)。"""
    model.eval()
    preds = []
    for i in range(0, len(pairs), batch_size):
        x = torch.as_tensor(pairs[i:i + batch_size], dtype=torch.float32, device=device)
        out = model(x).squeeze(-1)
        preds.append(out.cpu().numpy())
    p = np.concatenate(preds)
    y = labels[:, 1] > 0.5          # 独热标签第 1 列为正类
    pred_pos = p > 0.5
    acc = float((pred_pos == y).mean())
    tpr = float(pred_pos[y].mean())
    tnr = float((~pred_pos[~y]).mean())
    return acc, tpr, tnr


def resolve_loss_config(mode):
    """按实验模式返回 (use_contrastive, use_diversity, lam1, lam2, desc)。"""
    table = {
        'exp1': (False, False, 0.0, 0.0, 'Exp-1: Base model (BCE only)'),
        'exp2': (True, False, 0.1, 0.0, 'Exp-2: Base model + Contrastive'),
        'exp3': (True, True, 0.1, 0.05, 'Exp-3: Base model + Contrastive + Diversity'),
        'exp4': (False, True, 0.0, 0.05, 'Exp-4: Base model + Diversity (no contrastive)'),
    }
    if mode not in table:
        raise ValueError(f"未知模式: {mode}，可选: exp1, exp2, exp3, exp4")
    return table[mode]


def build_model(args):
    """创建 CPDI_ND 模型（配置与 main.py 一致）。"""
    return CPDI_ND(
        k=args.k, d_model=64, n_heads=4, dropout=0.3,
        enc_blocks=3, num_filters=32, attn_layers=args.attn_layers,
        use_attention=not args.no_attention
    )


# ============================================================================
# 三阶段训练（单个种子）
# ============================================================================

def train_three_stage(args, seed, delta_target, delta_stage1, device, run_dir):
    """对单个种子执行完整的三阶段训练，返回该次运行的指标。

    产物:
        model_<种子>.pth            全局最优模型（三阶段中验证准确率最高）
        model_<种子>_stage1.pth     阶段1结束模型（可用于 --skip-stage1 续跑）
        model_<种子>_stage2.pth     阶段2结束模型
        log_<种子>.csv              三阶段完整训练日志
    """
    use_contrastive, use_diversity, lam1, lam2, mode_desc = resolve_loss_config(args.mode)
    print(f"\nRunning {mode_desc}  |  seed = {seed}")

    model = build_model(args)

    # 加载初始权重：优先阶段1模型（--skip-stage1 时），否则 8 轮预训练模型
    init_path = None
    if args.skip_stage1 and args.stage1_model_path and os.path.exists(args.stage1_model_path):
        init_path = args.stage1_model_path
    elif args.pretrained_8round_path and os.path.exists(args.pretrained_8round_path):
        init_path = args.pretrained_8round_path

    if init_path:
        print(f"加载初始权重: {init_path}")
        model.load_state_dict(
            torch.load(init_path, map_location=device, weights_only=False))
        if args.skip_stage1 and args.num_seeds > 1:
            print("  注意: 已跳过阶段1，所有种子共用该初始权重（种子差异体现在后续训练）")
    else:
        print("警告: 未找到预训练模型，使用随机初始化")
        print("  强烈建议指定 --pretrained-8round-path，否则训练效果会大幅下降")

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {num_params:,}")

    criterion = nn.BCELoss()

    # 固定验证集（9轮，目标差分）——由训练种子确定性派生，跨运行一致
    print("生成固定验证集...")
    with contextlib.redirect_stdout(io.StringIO()):
        val_dataloader = get_cpdi_dataloader(
            total_samples=args.val_samples,
            num_rounds=9,
            diff=delta_target,
            k=args.k,
            batch_size=args.batch_size,
            num_workers=4,
            rng=np.random.default_rng([seed, _RNG_TAG_VAL])
        )

    records = []          # 三阶段训练日志
    best_val_acc = 0.0
    best_wts = None
    final_val_loss = float('nan')
    t_start = time.time()

    # ---------------------- 阶段1: 8轮模型定向微调 ----------------------
    if args.skip_stage1:
        print("\n" + "=" * 80)
        print("【阶段1】已跳过（--skip-stage1）")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print(f"【阶段1】模型定向微调: {args.stage1_rounds}轮Speck, "
              f"差分{hex(delta_stage1[0])}/{hex(delta_stage1[1])}")
        print(f"  样本数: {args.stage1_samples:,} | Epochs: {args.stage1_epochs} | "
              f"LR: {args.stage1_lr}")
        print("=" * 80)

        # 循环外创建一次：每个 epoch 重新 iter 时 dataset.epoch 递增 → 数据全新
        stage1_loader = get_cpdi_dataloader(
            total_samples=args.stage1_samples,
            num_rounds=args.stage1_rounds,
            diff=delta_stage1,
            k=args.k,
            batch_size=args.stage1_batch,
            num_workers=8,
            batch_gen_size=min(args.stage1_batch, 5000),
            data_seed=stage_data_seed(seed, 1),
            rng=np.random.default_rng(stage_data_seed(seed, 1))
        )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.stage1_lr,
            betas=(0.9, 0.999), weight_decay=1e-5
        )

        for epoch in range(1, args.stage1_epochs + 1):
            print(f"\n--- 阶段1 Epoch [{epoch}/{args.stage1_epochs}] ---")
            train_loss, train_acc = train_one_epoch_cpdi(
                model, stage1_loader, optimizer, criterion, device,
                use_contrastive=use_contrastive, use_diversity=use_diversity,
                lam1=lam1, lam2=lam2
            )
            # 阶段1 每 2 个 epoch（及末轮）在 9 轮验证集上验证一次
            if epoch % 2 == 0 or epoch == args.stage1_epochs:
                val_loss, val_acc = validate_cpdi(
                    model, val_dataloader, criterion, device,
                    use_contrastive=use_contrastive, use_diversity=use_diversity,
                    lam1=lam1, lam2=lam2
                )
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_wts = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    torch.save(model.state_dict(),
                               os.path.join(run_dir, f'model_{seed}_stage1.pth'))
                    print(f"  阶段1最优模型已保存, 验证准确率: {val_acc:.4f}")
            else:
                val_loss, val_acc = float('nan'), float('nan')
            records.append(dict(stage=1, epoch=epoch, lr=args.stage1_lr,
                                train_loss=train_loss, train_acc=train_acc,
                                val_loss=val_loss, val_acc=val_acc))

        torch.save(model.state_dict(),
                   os.path.join(run_dir, f'model_{seed}_stage1.pth'))
        print(f"阶段1训练完成，模型已保存至: {run_dir}/model_{seed}_stage1.pth")

    # ---------------------- 阶段2: 9轮任务初始化训练 ----------------------
    print("\n" + "=" * 80)
    print(f"【阶段2】9轮任务初始化训练: 9轮Speck, "
          f"差分{hex(delta_target[0])}/{hex(delta_target[1])}")
    print(f"  样本数: {args.stage2_samples:,} | Epochs: {args.stage2_epochs} | "
          f"LR: {args.stage2_lr}")
    print("=" * 80)

    # 同上：循环外创建一次，每 epoch 重新 iter 得到全新数据
    stage2_loader = get_cpdi_dataloader(
        total_samples=args.stage2_samples,
        num_rounds=9,
        diff=delta_target,
        k=args.k,
        batch_size=args.stage2_batch,
        num_workers=8,
        batch_gen_size=min(args.stage2_batch, 5000),
        data_seed=stage_data_seed(seed, 2),
        rng=np.random.default_rng(stage_data_seed(seed, 2))
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.stage2_lr,
        betas=(0.9, 0.999), weight_decay=1e-5
    )

    for epoch in range(1, args.stage2_epochs + 1):
        print(f"\n--- 阶段2 Epoch [{epoch}/{args.stage2_epochs}] ---")
        train_loss, train_acc = train_one_epoch_cpdi(
            model, stage2_loader, optimizer, criterion, device,
            use_contrastive=use_contrastive, use_diversity=use_diversity,
            lam1=lam1, lam2=lam2
        )
        val_loss, val_acc = validate_cpdi(
            model, val_dataloader, criterion, device,
            use_contrastive=use_contrastive, use_diversity=use_diversity,
            lam1=lam1, lam2=lam2
        )
        records.append(dict(stage=2, epoch=epoch, lr=args.stage2_lr,
                            train_loss=train_loss, train_acc=train_acc,
                            val_loss=val_loss, val_acc=val_acc))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_wts = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"  阶段2新的最优模型, 验证准确率: {val_acc:.4f}")

    torch.save(model.state_dict(),
               os.path.join(run_dir, f'model_{seed}_stage2.pth'))
    print(f"阶段2训练完成，模型已保存至: {run_dir}/model_{seed}_stage2.pth")

    # ---------------------- 阶段3: 阶梯式学习率微调 ----------------------
    print("\n" + "=" * 80)
    print(f"【阶段3】阶梯式学习率微调, 学习率序列: {args.stage3_lr_list}")
    print(f"  每步样本数: {args.stage3_samples_per_step:,}")
    print("=" * 80)

    for idx, lr in enumerate(args.stage3_lr_list):
        print(f"\n--- 微调步骤 {idx+1}/{len(args.stage3_lr_list)}, 学习率: {lr} ---")

        # 每步使用全新数据：data_seed 随步号变化（与 epoch 派生叠加）
        stage3_loader = get_cpdi_dataloader(
            total_samples=args.stage3_samples_per_step,
            num_rounds=9,
            diff=delta_target,
            k=args.k,
            batch_size=args.stage3_batch,
            num_workers=8,
            batch_gen_size=min(args.stage3_batch, 5000),
            data_seed=stage_data_seed(seed, 3, step=idx),
            rng=np.random.default_rng(stage_data_seed(seed, 3, step=idx))
        )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr,
            betas=(0.9, 0.999), weight_decay=1e-5
        )

        train_loss, train_acc = train_one_epoch_cpdi(
            model, stage3_loader, optimizer, criterion, device,
            use_contrastive=use_contrastive, use_diversity=use_diversity,
            lam1=lam1, lam2=lam2
        )
        val_loss, val_acc = validate_cpdi(
            model, val_dataloader, criterion, device,
            use_contrastive=use_contrastive, use_diversity=use_diversity,
            lam1=lam1, lam2=lam2
        )
        records.append(dict(stage=3, epoch=idx + 1, lr=lr,
                            train_loss=train_loss, train_acc=train_acc,
                            val_loss=val_loss, val_acc=val_acc))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_wts = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"  新的最优模型, 验证准确率: {val_acc:.4f}")

    train_time = time.time() - t_start

    # ---------------------- 保存最优模型与日志 ----------------------
    if best_wts is not None:
        model.load_state_dict(best_wts)
    torch.save(model.state_dict(), os.path.join(run_dir, f'model_{seed}.pth'))
    final_val_loss = records[-1]['val_loss'] if records else float('nan')

    pd.DataFrame(records).to_csv(
        os.path.join(run_dir, f'log_{seed}.csv'), index=False)

    print("\n" + "=" * 80)
    print(f"9轮三阶段训练完成! 最优验证准确率: {best_val_acc:.4f}")
    print(f"最优模型: {run_dir}/model_{seed}.pth")
    print(f"训练日志: {run_dir}/log_{seed}.csv")
    print("=" * 80)

    return dict(
        seed=seed,
        best_val_acc=float(best_val_acc),
        final_val_acc=float(records[-1]['val_acc']) if records else float('nan'),
        val_loss=float(final_val_loss),
        train_time_s=train_time,
    )


# ============================================================================
# 多种子汇总（格式与 main.py 完全一致）
# ============================================================================

def write_summary(results, run_dir, args):
    """写多种子汇总 summary.txt：训练结果 + K泛化测试矩阵 + 平均值/方差。"""
    accs = np.array([r['best_val_acc'] for r in results])
    losses = np.array([r['val_loss'] for r in results])
    n = len(results)
    has_test = bool(results[0].get('test'))
    ks = sorted(results[0]['test'].keys()) if has_test else []
    col_w = 10
    width = max(70, 6 + 3 + col_w * len(ks)) if has_test else 70

    lines = []
    lines.append('CPDI-ND 9轮三阶段训练汇总')
    lines.append('=' * width)
    lines.append(f"配置: 轮数=9, k={args.k}, 模式={args.mode}, "
                 f"注意力={'关(no-attention)' if args.no_attention else '开'}, "
                 f"差分={args.diff}, "
                 f"阶段1={args.stage1_rounds}轮/{args.stage1_samples}样本/"
                 f"{args.stage1_epochs}ep, 阶段2={args.stage2_samples}样本/"
                 f"{args.stage2_epochs}ep, 阶段3={args.stage3_samples_per_step}样本×"
                 f"{len(args.stage3_lr_list)}步, 验证集={args.val_samples}")
    if getattr(args, 'seed_list', None):
        lines.append(f"种子（--seeds 显式指定）: {args.seed_list}")
    else:
        lines.append(f"种子（每次运行随机抽取）: 共 {n} 个（见下表）")
    if has_test:
        lines.append(f"测试: 每个测试K的样本数={args.test_samples}（正负各半）, "
                     f"数据种子={TEST_DATA_SEED}（固定，所有模型共用同一测试集，"
                     "可复现），差分/轮数与训练相同")
        lines.append(f"测试K（K泛化扫描）: {ks}")
    lines.append("说明: 训练/验证数据由训练种子确定性派生（在线模式按 "
                 "(种子,阶段,epoch,worker)，每个 epoch 数据全新且同种子跨运行逐位一致），"
                 "种子同时控制参数初始化、数据加载顺序与 cudnn 确定性算法选择。"
                 "负样本始终为随机差分（与 main.py 一致）。")
    lines.append('=' * width)

    # ---- 训练指标表 ----
    lines.append(f"{'seed':>6} | {'best_val_acc':>12} | {'final_val_acc':>13} | "
                 f"{'val_loss':>9} | {'time(s)':>8}")
    lines.append('-' * width)
    for r in results:
        lines.append(f"{r['seed']:>6} | {r['best_val_acc']:>12.4f} | "
                     f"{r['final_val_acc']:>13.4f} | {r['val_loss']:>9.4f} | "
                     f"{r['train_time_s']:>8.0f}")
    lines.append('-' * width)
    lines.append(f"best_val_acc 平均值: {accs.mean():.4f}")
    lines.append(f"val_loss     平均值: {losses.mean():.4f}")
    if n > 1:
        lines.append(f"best_val_acc 方差(样本): {accs.var(ddof=1):.6f}")
        lines.append(f"best_val_acc 标准差:     {accs.std(ddof=1):.4f}")
        lines.append(f"val_loss     方差(样本): {losses.var(ddof=1):.6f}")
    else:
        lines.append("方差: N/A（仅 1 次运行）")

    # ---- K泛化测试矩阵：test_acc / TPR / TNR ----
    if has_test:
        def matrix(title, key, fmt):
            lines.append('')
            lines.append(f"--- {title}（行=种子，列=测试K） ---")
            header = f"{'seed':>6} |" + ''.join(f"{'K=' + str(K):>{col_w}}" for K in ks)
            lines.append(header)
            lines.append('-' * len(header))
            for r in results:
                lines.append(f"{r['seed']:>6} |" + ''.join(
                    f"{r['test'][K][key]:>{col_w}{fmt}}" for K in ks))
            vals = np.array([[r['test'][K][key] for K in ks] for r in results])
            lines.append('-' * len(header))
            lines.append(f"{'平均值':>6} |" + ''.join(
                f"{v:>{col_w}.4f}" for v in vals.mean(axis=0)))
            if n > 1:
                lines.append(f"{'标准差':>6} |" + ''.join(
                    f"{v:>{col_w}.4f}" for v in vals.std(axis=0, ddof=1)))
                lines.append(f"{'方差':>6} |" + ''.join(
                    f"{v:>{col_w}.6f}" for v in vals.var(axis=0, ddof=1)))

        matrix('test_acc', 'acc', '.4f')
        matrix('TPR', 'tpr', '.4f')
        matrix('TNR', 'tnr', '.4f')

    summary_p = os.path.join(run_dir, 'summary.txt')
    with open(summary_p, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\nSummary saved to {summary_p}")


# ============================================================================
# 命令行参数解析
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='CPDI-ND 9轮神经区分器三阶段训练（逻辑与 main.py 一致）'
    )

    # --- 数据与差分（与 main.py 一致；负样本固定为随机差分，无选项） ---
    parser.add_argument('--diff', type=str, default='0x2800,0x0010',
                        help='目标差分，格式: hex_high,hex_low（默认 0x2800,0x0010）')
    parser.add_argument('--stage1-diff', type=str, default='0x8000,0x840a',
                        help='阶段1源任务差分（目标差分经4轮传播后的最优输出差分，'
                             '默认 0x8000,0x840a）')
    parser.add_argument('--k', type=int, default=32,
                        help='每个样本的密文对数量')
    parser.add_argument('--val-samples', type=int, default=100000,
                        help='验证样本数')
    parser.add_argument('--test-samples', type=int, default=100000,
                        help='每个测试K的测试集样本数（正负各半，默认100000）。训练结束后'
                             '在 K∈{4,8,16,32,64,128,256}（自动包含训练k）的固定测试集'
                             '上统一评估，K泛化结果写入 summary.txt；设为 0 跳过测试')

    # --- 模型 ---
    parser.add_argument('--mode', type=str, default='exp1',
                        choices=['exp1', 'exp2', 'exp3', 'exp4'],
                        help='损失函数模式: exp1=BCE, exp2=BCE+对比, '
                             'exp3=BCE+对比+多样性, exp4=BCE+多样性')
    parser.add_argument('--no-attention', action='store_true',
                        help='禁用自注意力模块（消融实验用，跳过 Self-Attention 层）')
    parser.add_argument('--attn-layers', type=int, default=3,
                        help='Self-Attention 层数（须与预训练模型一致）')

    # --- 预训练权重 ---
    parser.add_argument('--pretrained-8round-path', type=str, default='',
                        help='预训练好的8轮模型路径，如 result/2800/8_k32_exp2/model_42.pth')
    parser.add_argument('--stage1-model-path', type=str, default='',
                        help='阶段1训练好的模型路径（--skip-stage1 时加载该模型直接进入阶段2）')
    parser.add_argument('--skip-stage1', action='store_true',
                        help='跳过阶段1，直接进入阶段2训练')

    # --- 阶段1参数: 模型定向微调 ---
    parser.add_argument('--stage1-rounds', type=int, default=5,
                        help='阶段1训练的Speck轮数（用较少轮数做迁移学习）')
    parser.add_argument('--stage1-samples', type=int, default=10**6,
                        help='阶段1训练样本数')
    parser.add_argument('--stage1-epochs', type=int, default=10,
                        help='阶段1训练轮数')
    parser.add_argument('--stage1-batch', type=int, default=5000,
                        help='阶段1批次大小')
    parser.add_argument('--stage1-lr', type=float, default=1e-4,
                        help='阶段1学习率')

    # --- 阶段2参数: 9轮任务初始化训练 ---
    parser.add_argument('--stage2-samples', type=int, default=5*10**6,
                        help='阶段2每epoch训练样本数')
    parser.add_argument('--stage2-epochs', type=int, default=50,
                        help='阶段2训练轮数（每epoch重新生成数据）')
    parser.add_argument('--stage2-batch', type=int, default=5000,
                        help='阶段2批次大小')
    parser.add_argument('--stage2-lr', type=float, default=1e-4,
                        help='阶段2学习率')

    # --- 阶段3参数: 阶梯式学习率微调 ---
    parser.add_argument('--stage3-lr-list', type=float, nargs='+',
                        default=[1e-5, 1e-6],
                        help='阶段3学习率序列（逐步降低）')
    parser.add_argument('--stage3-samples-per-step', type=int, default=5*10**6,
                        help='阶段3每步训练样本数')
    parser.add_argument('--stage3-batch', type=int, default=5000,
                        help='阶段3批次大小')

    # --- 运行与保存（与 main.py 一致） ---
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU 设备 ID（默认 "0"）')
    parser.add_argument('--save-dir', type=str, default='',
                        help='模型保存目录（默认按差分自动命名: result/<差分高16位>）')
    parser.add_argument('--batch-size', type=int, default=2000,
                        help='验证/测试批次大小')

    # --- 多种子重复（与 main.py 一致） ---
    parser.add_argument('--num-seeds', type=int, default=1,
                        help='用随机种子重复训练的次数（默认1）。搭配 --seeds 可显式指定种子')
    parser.add_argument('--seeds', type=str, default='',
                        help='显式指定随机种子，逗号分隔（如 "42,43,44"）。数量必须与 '
                             '--num-seeds 一致，否则拒绝运行；不指定时每次运行随机抽取种子')

    args = parser.parse_args()

    # --- --seeds 与 --num-seeds 的一致性检查（与 main.py 一致） ---
    args.seed_list = None
    if args.seeds:
        try:
            seed_list = [int(s) for s in args.seeds.split(',')]
        except ValueError:
            parser.error(f'--seeds 格式错误: "{args.seeds}"，'
                         '应为逗号分隔的非负整数，如 "42,43,44"')
        if any(not (0 <= s < 2 ** 31 - 1) for s in seed_list):
            parser.error('--seeds 的每个种子需为 [0, 2^31-1) 内的整数')
        if len(seed_list) != args.num_seeds:
            parser.error(f'--seeds 给定了 {len(seed_list)} 个种子，与 --num-seeds='
                         f'{args.num_seeds} 数量不一致，拒绝运行：两者必须匹配；'
                         '不指定 --seeds 时将按 --num-seeds 次数随机抽取种子')
        args.seed_list = seed_list
    return args


# ============================================================================
# 入口
# ============================================================================

def main():
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    delta_target = parse_diff(args.diff)
    delta_stage1 = parse_diff(args.stage1_diff)

    print("=" * 80)
    print("CPDI-ND 9轮区分器三阶段训练（数据/测试/复现逻辑与 main.py 一致）")
    print("=" * 80)
    print(f"Target diff: {args.diff} | Stage1 diff: {args.stage1_diff} | "
          f"k={args.k} | rounds=9")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")

    # 保存目录：与 main.py 一致（未显式指定时按差分自动命名）
    if not args.save_dir:
        args.save_dir = os.path.join('result', f'{delta_target[0]:04x}')
    # 无注意力消融加后缀，避免与同 mode 的有注意力实验互相覆盖
    run_dir_name = f'9_k{args.k}_{args.mode}'
    if args.no_attention:
        run_dir_name += '_no-attention'
    run_dir = os.path.join(args.save_dir, run_dir_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Save dir: {run_dir}")

    # 种子：显式指定 → 可复现；否则随机抽取并记录
    seeds = (args.seed_list if args.seed_list is not None
             else [secrets.randbelow(2 ** 31 - 1) for _ in range(args.num_seeds)])
    print(f"Seeds ({'specified' if args.seed_list is not None else 'random'}): {seeds}")

    # =========================================================================
    # 逐种子三阶段训练
    # =========================================================================
    results = []
    for i, seed in enumerate(seeds):
        print(f"\n{'=' * 70}")
        print(f"Run {i + 1}/{args.num_seeds}  |  seed = {seed}")
        print(f"{'=' * 70}")
        set_global_seed(seed)
        r = train_three_stage(args, seed, delta_target, delta_stage1, device, run_dir)
        results.append(r)

    # =========================================================================
    # 统一测试：K 泛化扫描（与 main.py 完全一致）
    # =========================================================================
    if args.test_samples > 0:
        test_ks = sorted(set([4, 8, 16, 32, 64, 128, 256] + [args.k]))
        print(f"\n{'=' * 70}")
        print(f"Testing all {len(results)} models across K = {test_ks}")
        print(f"  ({args.test_samples} samples per K, data seed = {TEST_DATA_SEED})")
        print(f"{'=' * 70}")
        test_models = {}
        for r in results:
            model_path = os.path.join(run_dir, f"model_{r['seed']}.pth")
            m = build_model(args)
            m.load_state_dict(
                torch.load(model_path, map_location=device, weights_only=False))
            test_models[r['seed']] = m.to(device)
        for r in results:
            r['test'] = {}
        # 逐测试K：生成一次固定测试集 → 所有模型评估 → 释放（控制内存峰值）
        for tk in test_ks:
            pairs, labels = generate_test_set(
                args.test_samples, 9, delta_target, tk)
            for r in results:
                acc, tpr, tnr = evaluate_model_on(
                    test_models[r['seed']], pairs, labels, args.batch_size, device)
                r['test'][tk] = dict(acc=acc, tpr=tpr, tnr=tnr)
                print(f"  seed={r['seed']}  K={tk:>3}: acc={acc:.4f}  "
                      f"(TPR={tpr:.4f}, TNR={tnr:.4f})")
            del pairs, labels

    write_summary(results, run_dir, args)
    print("\nDone!")


if __name__ == '__main__':
    main()
