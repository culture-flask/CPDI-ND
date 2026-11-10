"""
CPDI-ND 随机差分负样本训练入口
======================================================================
本文件是 CPDI-ND 使用随机差分负样本（Gohr 原始策略）的训练脚本：
负样本不使用固定差分 Δ_neg，而是完全随机的 P'。

目标差分 Δ_target 通过 --diff 指定，模型保存目录默认按差分自动命名
（如 0x0040 → result/0040，0x2800 → result/2800）。

快速上手:
    # 消融实验 1（基础模型），目标差分 0x0040/0x0000，保存至 result/0040
    python main_random_neg.py --mode exp1 --k 32 --train-samples 50000

    # 目标差分 0x2800/0x0010，保存至 result/2800
    python main_random_neg.py --mode exp2 --diff 0x2800,0x0010

    # 用 3 个随机种子重复训练，结果汇总于 summary.txt
    python main_random_neg.py --mode exp2 --diff 0x2800,0x0010 --num-seeds 3

    # 指定种子以复现实验（种子数量必须与 --num-seeds 一致，否则拒绝运行）
    python main_random_neg.py --mode exp2 --diff 0x2800,0x0010 --num-seeds 3 --seeds 42,43,44

运行模式说明:
    exp1/2/3/4 = 消融实验（逐步添加组件）

命令行参数一览:
    --diff           正样本目标差分，格式 "hex_high,hex_low"，默认 "0x0040,0x0000"
    --num-rounds     加密轮数，默认 7（Speck32/64）
    --k             每个样本的密文对数量，默认 32
    --train-samples  训练样本数，默认 1000000
    --val-samples    验证样本数，默认 100000
    --batch-size     批次大小，默认 2000
    --epochs         目标差分微调 epoch 数，默认 50
    --gpu            指定 GPU 设备 ID，默认 "2"
    --mode           实验模式（见上）
    --save-dir       模型保存目录，默认按差分命名: result/<差分高16位>
    --num-seeds      用随机种子重复训练的次数，默认 1（多种子统计）
    --seeds          显式指定随机种子（逗号分隔，如 "42,43,44"），用于可复现实验；
                     数量必须与 --num-seeds 完全一致，否则拒绝运行；
                     不指定时每次运行随机抽取种子，记录在文件名与 summary.txt 中
    --no-contrastive  禁用对比损失
    --no-diversity    禁用多样性正则

输出文件（保存至 <save_dir>/<轮数>_k<k>_<mode>/，如 result/2800/8_k32_exp2/）:
    model_<种子>.pth   每个种子的模型权重
    log_<种子>.csv     每个种子的训练日志
    summary.txt        多种子汇总（每个种子的训练指标、平均值、方差）

    训练结束后不再自动做 K 泛化测试；需要测试时用 batch_test_models.py。

复现性:
    训练/验证数据由训练种子确定性派生：在线模式按 (种子, epoch, worker) 派生，
    每个 epoch 生成全新数据（避免对固定数据集过拟合），同种子重跑逐位一致；
    cudnn 使用确定性算法。K 泛化测试集由固定数据种子 TEST_DATA_SEED=20241016
    生成（见 generate_test_set，供 batch_test_models.py 调用），跨运行、跨模型完全一致。
    指定 --seeds 后，同一命令可完整复现全部训练结果。
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
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import IterableDataset

import speck as sp
from cpdi_nd_model import CPDI_ND
import cpdi_nd_train as trainer


# ============================================================================
# 在线数据生成模块
# ============================================================================

class CPDI_NDIterableDataset(IterableDataset):
    """在线生成 CPDI-ND 训练数据的迭代式数据集。

    当样本量超过内存容量时，每次迭代生成一个 mini-batch 的数据，避免内存溢出。

    可复现性：指定 data_seed 时，本 worker 本 epoch 的数据由
    (data_seed, epoch, worker_id) 确定性派生 —— epoch 参与派生保证每个
    epoch 数据全新（不对固定数据集过拟合），同种子跨运行逐位一致（可复现）；
    data_seed 为 None 时退回系统熵源 urandom（每次独立，不可复现）。
    """

    def __init__(self, total_samples, num_rounds, diff, k,
                 random_neg=True, batch_gen_size=2000, data_seed=None,
                 shared_key=False):
        super().__init__()
        self.total_samples = total_samples
        self.num_rounds = num_rounds
        self.diff = diff
        self.k = k
        self.random_neg = random_neg
        self.batch_gen_size = batch_gen_size
        self.total_steps = max(total_samples // batch_gen_size, 1)
        self.data_seed = data_seed
        self.shared_key = shared_key
        self.epoch = 0

    def __iter__(self):
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
        if self.data_seed is None:
            rng = None
        else:
            rng = np.random.default_rng([self.data_seed, self.epoch, worker_id])
        self.epoch += 1

        for step in range(start, end):
            with contextlib.redirect_stdout(io.StringIO()):
                pairs, labels = trainer.load_cpdi_data(
                    self.batch_gen_size, self.num_rounds,
                    diff=self.diff, k=self.k,
                    random_neg=self.random_neg, rng=rng,
                    shared_key=self.shared_key
                )
            pair_tensor = torch.tensor(pairs, dtype=torch.float32)
            label_tensor = torch.tensor(labels, dtype=torch.float32)

            for i in range(self.batch_gen_size):
                yield pair_tensor[i], label_tensor[i]


def get_cpdi_dataloader(total_samples, num_rounds, diff, k,
                        random_neg=True, batch_size=2000, num_workers=4,
                        batch_gen_size=2000, online=False, data_seed=None, rng=None,
                        shared_key=False):
    """获取 CPDI-ND 格式的 DataLoader。

    参数:
        online: 是否使用在线生成。False 时全量加载（默认），True 时在线迭代生成。
        data_seed: 在线模式的数据种子——数据按 (data_seed, epoch, worker) 确定性
                   派生，每个 epoch 全新且同种子跨运行逐位一致（可复现）。
        rng: 全量加载模式的 numpy Generator（确定性生成）；None 时用系统熵源。
        shared_key: 密钥策略（默认 False = 每对独立密钥；True = 样本内共享密钥）。
    """
    if not online:
        # 全量加载（shuffle=False：DataLoader 已按批洗牌，跳过全量数组的冗余拷贝）
        print(f"  全量加载 {total_samples:,} 样本...")
        pairs, labels = trainer.load_cpdi_data(
            total_samples, num_rounds, diff=diff, k=k,
            random_neg=random_neg, shuffle=False, rng=rng, shared_key=shared_key
        )
        dataset = trainer.CPDI_ND_Dataset(pairs, labels)
        dataloader = Data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0)
        )
    else:
        # 在线迭代生成（分块内部打乱，保证每批正负混合）
        print(f"  在线生成 {total_samples:,} 样本 (负样本: 随机差分)...")
        dataset = CPDI_NDIterableDataset(
            total_samples, num_rounds, diff, k,
            random_neg=random_neg,
            batch_gen_size=batch_gen_size,
            data_seed=data_seed, shared_key=shared_key
        )
        dataloader = Data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0)
        )
    return dataloader


# ============================================================================
# 命令行参数解析
# ============================================================================

def parse_diff(diff_str):
    """解析差分字符串 "hex_high,hex_low" 为 (high, low) 元组。"""
    parts = diff_str.split(',')
    return (int(parts[0], 16), int(parts[1], 16))


def parse_args():
    """解析命令行参数，返回 argparse.Namespace 对象。"""
    parser = argparse.ArgumentParser(description='CPDI-ND Neural Distinguisher Training (Random Negative Diff)')

    # --- 数据参数 ---
    parser.add_argument('--diff', type=str, default='0x0040,0x0000',
                        help='正样本目标差分，格式: hex_high,hex_low（如 0x0040,0x0000 或 0x2800,0x0010）')
    parser.add_argument('--num-rounds', type=int, default=7,
                        help='Speck 加密轮数（影响区分器难度，轮数越多越难区分）')
    parser.add_argument('--k', type=int, default=32,
                        help='每个样本包含的密文对数量。k 越大，二阶统计信息越丰富。')

    parser.add_argument('--train-samples', type=int, default=1000000,
                        help='训练样本数。正负各半。')
    parser.add_argument('--val-samples', type=int, default=100000,
                        help='验证样本数')

    # --- 训练参数 ---
    parser.add_argument('--batch-size', type=int, default=2000,
                        help='批次大小。')
    parser.add_argument('--epochs', type=int, default=50,
                        help='目标差分训练 epoch 数')

    # --- 运行参数 ---
    parser.add_argument('--gpu', type=str, default='2',
                        help='GPU 设备 ID（CUDA_VISIBLE_DEVICES），默认 "2"。设为空字符串使用 CPU')
    parser.add_argument('--mode', type=str, default='exp1',
                        choices=['exp1', 'exp2', 'exp3', 'exp4'],
                        help='实验模式:\n'
                             '  exp1     - 基础模型\n'
                             '  exp2     - 基础模型 + 对比损失\n'
                             '  exp3     - 基础模型 + 对比损失 + 多样性正则\n'
                             '  exp4     - 基础模型 + 多样性正则\n')
    parser.add_argument('--save-dir', type=str, default='',
                        help='模型和日志保存目录（默认按差分自动命名: result/<差分高16位>）')

    # --- 多种子重复 ---
    parser.add_argument('--num-seeds', type=int, default=1,
                        help='用随机种子重复训练的次数（默认1）。搭配 --seeds 可显式指定种子')
    parser.add_argument('--seeds', type=str, default='',
                        help='显式指定随机种子，逗号分隔（如 "42,43,44"）。数量必须与 '
                             '--num-seeds 一致，否则拒绝运行；不指定时每次运行随机抽取种子')

    # --- 消融开关 ---

    parser.add_argument('--no-contrastive', action='store_true',
                        help='禁用对比损失（loss 中去掉 L_contrast 项）')
    parser.add_argument('--no-diversity', action='store_true',
                        help='禁用多样性正则（loss 中去掉 L_div 项）')
    parser.add_argument('--no-attention', action='store_true',
                        help='禁用自注意力模块（消融实验用，跳过 Self-Attention 层）')
    parser.add_argument('--online', action='store_true',
                        help='启用在线数据生成（大样本量时避免内存溢出，默认关闭）')
    parser.add_argument('--batch-gen-size', type=int, default=2000,
                        help='在线生成时每次生成的 mini-batch 样本数（默认 2000）')

    args = parser.parse_args()

    # --- --seeds 与 --num-seeds 的一致性检查（保证可复现） ---
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
# 主函数：模式分发
# ============================================================================

def set_global_seed(seed):
    """设置全局随机种子（模型初始化、DataLoader 洗牌顺序、数据派生）。

    cudnn 确定性模式：卷积反向传播选择确定性算法，保证同种子下训练
    过程逐位一致（GPU 并行浮点仍可能有极微小的非确定性来源，但本项目的
    卷积/注意力/BN 路径在确定性模式下可完整复现）。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 测试集数据种子：固定值——所有模型共用同一测试集，且跨运行完全可复现
TEST_DATA_SEED = 20241016

# 训练/验证数据的派生标签：与训练种子组成 SeedSequence 熵池（[seed, 标签]）。
# 在线模式的数据按 (种子, epoch, worker) 派生，epoch/worker 取值远小于
# 以下标签值，不会发生碰撞。
_RNG_TAG_VAL = 1_000_000_000          # 验证集（一次性生成）
_RNG_TAG_TRAIN_FULL = 1_000_000_001   # 全量加载的训练集（一次性生成）


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


def main():
    """CPDI-ND 实验主函数（随机差分负样本版本）。

    工作流程:
      1. 解析命令行参数（目标差分由 --diff 指定；--num-seeds 控制重复训练次数）
      2. 配置 CUDA 设备
      3. 按 --num-seeds 个随机种子逐一训练，每个种子独立保存模型与训练日志
      4. 汇总所有种子的结果，写 summary.txt（每个种子、平均值、方差）
    """
    args = parse_args()

    # --- CUDA 设备配置 ---
    # 通过 CUDA_VISIBLE_DEVICES 环境变量控制在哪个 GPU 上运行
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"CPDI-ND Experiment (Random Neg): diff={args.diff}, rounds={args.num_rounds}, k={args.k}")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    # =========================================================================
    # 多种子重复训练
    # =========================================================================
    # --seeds 显式指定 → 按指定列表运行（可复现）；否则每次运行随机抽取种子并记录
    seeds = (args.seed_list if args.seed_list is not None
             else [secrets.randbelow(2 ** 31 - 1) for _ in range(args.num_seeds)])
    print(f"Seeds ({'specified' if args.seed_list is not None else 'random'}): {seeds}")

    # =========================================================================
    # 差分策略配置
    # =========================================================================
    # 目标差分 Δ_target 由 --diff 指定，负样本始终使用随机差分
    delta_target = parse_diff(args.diff)

    # 保存目录：未显式指定时按差分自动命名（如 0x0040 → result/0040）
    if not args.save_dir:
        args.save_dir = os.path.join('result', f'{delta_target[0]:04x}')
    # 本配置的所有产物保存至: <save_dir>/<轮数>_k<k>_<mode>[_no-attention]/
    # 无注意力消融加后缀，避免与同 mode 的有注意力实验互相覆盖
    run_dir_name = f'{args.num_rounds}_k{args.k}_{args.mode}'
    if args.no_attention:
        run_dir_name += '_no-attention'
    run_dir = os.path.join(args.save_dir, run_dir_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Save dir: {run_dir}")

    # --- 消融开关的否定逻辑转换 ---
    use_contrastive = not args.no_contrastive
    use_diversity = not args.no_diversity
    use_attention = not args.no_attention

    # =========================================================================
    # 逐种子训练
    # =========================================================================
    results = []
    for i, seed in enumerate(seeds):
        print(f"\n{'=' * 70}")
        print(f"Run {i + 1}/{args.num_seeds}  |  seed = {seed}")
        print(f"{'=' * 70}")
        set_global_seed(seed)
        r = run_ablation(args, device, delta_target,
                         use_contrastive, use_diversity, use_attention, seed, run_dir)
        results.append(r)

    # =========================================================================
    # 汇总本次所有种子的训练指标（K 泛化测试请用 batch_test_models.py）
    # =========================================================================
    write_summary(results, run_dir, args)
    print("\nDone!")


def write_summary(results, run_dir, args):
    """写多种子汇总 summary.txt：每个种子的训练指标 + 平均值/方差。"""
    accs = np.array([r['best_val_acc'] for r in results])
    losses = np.array([r['val_loss'] for r in results])
    n = len(results)
    width = 70

    lines = []
    lines.append('CPDI-ND 多种子训练汇总')
    lines.append('=' * width)
    lines.append(f"配置: 轮数={args.num_rounds}, k={args.k}, 模式={args.mode}, "
                 f"注意力={'关(no-attention)' if args.no_attention else '开'}, "
                 f"差分={args.diff}, 训练样本={args.train_samples}, "
                 f"epochs={args.epochs}, batch={args.batch_size}")
    if getattr(args, 'seed_list', None):
        lines.append(f"种子（--seeds 显式指定）: {args.seed_list}")
    else:
        lines.append(f"种子（每次运行随机抽取）: 共 {n} 个（见下表）")
    lines.append("说明: 训练/验证数据由训练种子确定性派生（在线模式按 (种子,epoch,worker)，"
                 "每个 epoch 数据全新且同种子跨运行逐位一致），种子同时控制参数"
                 "初始化、数据加载顺序与 cudnn 确定性算法选择。")
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

    summary_p = os.path.join(run_dir, 'summary.txt')
    with open(summary_p, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\nSummary saved to {summary_p}")


# ============================================================================
# 消融实验框架
# ============================================================================

def run_ablation(args, device, delta_target,
                 use_contrastive, use_diversity, use_attention, seed, run_dir):
    """运行一次训练（指定随机种子），返回该次运行的关键指标。

    产物按种子保存: <run_dir>/model_<种子>.pth 与 <run_dir>/log_<种子>.csv

    消融实验设计（对应方案 §2.5）:

    ┌─────────┬──────────────┬────────────┬─────────────────────────┐
    │ 实验    │ Contrastive  │ Diversity  │ 说明                     │
    ├─────────┼──────────────┼────────────┼─────────────────────────┤
    │ exp1    │ ✗            │ ✗          │ 基础模型                 │
    │ exp2    │ ✓            │ ✗          │ + 对比损失               │
    │ exp3    │ ✓            │ ✓          │ + 多样性正则             │
    └─────────┴──────────────┴────────────┴─────────────────────────┘

    每组实验使用相同的数据和基架，仅变更损失函数配置。
    对比各组在验证集上的准确率，即可量化每个组件的贡献。
    """
    exp_name = args.mode

    # 各组件的消融配置
    exp_configs = {
        'exp1': {
            'use_contrastive': False,
            'use_diversity': False,
            'desc': 'Exp-1: Base model (no contrastive, no diversity)'
        },
        'exp2': {
            'use_contrastive': use_contrastive,
            'use_diversity': False,
            'desc': 'Exp-2: Base model + Contrastive Loss'
        },
        'exp3': {
            'use_contrastive': use_contrastive,
            'use_diversity': True,
            'desc': 'Exp-3: Base model + Contrastive + Diversity'
        },
        'exp4': {
            'use_contrastive': False,
            'use_diversity': True,
            'desc': 'Exp-4: Base model + Diversity (no contrastive)'
        },
    }

    config = exp_configs.get(exp_name, exp_configs['exp1'])
    print(f"\nRunning {config['desc']}")

    # 根据消融配置重新创建模型
    model = CPDI_ND(
        k=args.k, d_model=64, n_heads=4, dropout=0.3,
        enc_blocks=3, num_filters=32,
        use_attention=use_attention
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {num_params:,}")

    # 生成目标差分数据
    # 训练/验证数据均由训练种子确定性派生（在线模式按 (seed, epoch, worker)，
    # 每个 epoch 全新且同种子跨运行逐位一致）
    train_loader = get_cpdi_dataloader(
        args.train_samples, args.num_rounds, diff=delta_target,
        k=args.k, random_neg=True, batch_size=args.batch_size,
        num_workers=4, batch_gen_size=args.batch_gen_size,
        online=args.online,
        data_seed=seed,
        rng=None if args.online else np.random.default_rng([seed, _RNG_TAG_TRAIN_FULL])
    )
    val_loader = get_cpdi_dataloader(
        args.val_samples, args.num_rounds, diff=delta_target,
        k=args.k, random_neg=True, batch_size=args.batch_size,
        num_workers=4, batch_gen_size=args.batch_gen_size,
        online=False,  # 验证集始终全量加载
        data_seed=seed, rng=np.random.default_rng([seed, _RNG_TAG_VAL])
    )

    save_p = os.path.join(run_dir, f'model_{seed}.pth')
    log_p = os.path.join(run_dir, f'log_{seed}.csv')

    # 训练（仅使用目标差分，不涉及多差分预训练）
    t_start = time.time()
    model, train_process = trainer.train_cpdi_nd(
        model, train_loader, val_loader,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        high_lr=1e-3, low_lr=1e-4,
        lam1=0.10 if config['use_contrastive'] else 0.0,
        lam2=0.05 if config['use_diversity'] else 0.0,
        save_path=save_p, log_path=log_p
    )
    train_time = time.time() - t_start

    # 评估
    criterion = nn.BCELoss()
    model.to(device)
    val_loss, val_acc, _, _, _ = trainer.evaluate(model, val_loader, criterion, device)
    print(f"\nseed={seed} ({exp_name}) - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return dict(
        seed=seed,
        best_val_acc=float(train_process['val_acc'].max()),
        final_val_acc=float(val_acc),
        val_loss=float(val_loss),
        train_time_s=train_time,
    )


# ============================================================================
# 入口
# ============================================================================

if __name__ == '__main__':
    main()
