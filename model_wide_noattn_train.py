"""
CPDI-ND 参数量对齐的加宽无注意力基线
======================================================================
用途：
    无注意力模型（71,681 参数）比完整模型（121,985 参数）少约 41%，
    因此无法排除性能提升来源于"参数量/网络深度"的可能性。
    本脚本构造一个**参数量与完整模型对齐**的无注意力基线：

        - 编码器加宽：num_filters 32 → 48
        - 编码器加深：InceptionResBlock 3 → 5
        - 分类头加宽：hidden 128 → 224
        - 总参数量 121,153（完整模型 121,985，偏差 -0.68%）

    由于该基线比完整模型更深，若其在本任务上仍显著落后，则说明增益来自
    跨对交互结构而非参数容量。

训练/验证/测试/保存/复现逻辑与 main.py 完全一致（复用 main.py 的工具函数，
不修改 main.py 本身）：
    - 负样本始终为随机差分
    - 训练数据由训练种子确定性派生（在线按 (种子, epoch, worker)），
      每个 epoch 全新且同种子跨运行逐位一致
    - 多种子重复（--num-seeds / --seeds，数量必须一致）
    - cudnn 确定性算法
    - 保存结构: <save_dir>/<轮数>_k<k>_wide-<mode>/model_<种子>.pth + log_<种子>.csv
    - 训练结束自动在固定测试集上做 K∈{4,...,256} 泛化扫描，写入 summary.txt

使用示例:
    # 论文对照实验（8轮、k=32、exp2、3个固定种子）
    python model_wide_noattn_train.py --mode exp2 --num-rounds 8 --k 32 --gpu 4 \
        --num-seeds 3 --seeds 42,43,44

    # 查看参数量对齐候选配置（不训练）
    python model_wide_noattn_train.py --search-params

    # 改用严格 -0.18% 对齐的配置（48通道/4块/256隐藏）
    python model_wide_noattn_train.py --mode exp2 --num-filters 48 --enc-blocks 4 \
        --hidden1 256 --gpu 4
"""

import os
import sys
import argparse
import time
import random
import secrets

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data

import main as base                     # 复用数据派生 / 种子 / 测试集 / 评估工具
import cpdi_nd_train as trainer
from cpdi_nd_model import CipherPairEncoder


# ============================================================================
# 完整模型参数量（对齐目标）
# ============================================================================

TARGET_PARAMS = 121985          # CPDI_ND(k=32, 默认配置) 的可训练参数量
ORIG_NOATTN_PARAMS = 71681      # 原无注意力基线参数量（对照参考）

# 默认的对齐配置（偏差 -0.68%；基线比完整模型更深，结论更保守）
DEFAULT_NUM_FILTERS = 48
DEFAULT_ENC_BLOCKS = 5
DEFAULT_HIDDEN1 = 224
DEFAULT_HIDDEN2 = 64


# ============================================================================
# 加宽无注意力模型
# ============================================================================

class WideNoAttnCPDI_ND(nn.Module):
    """参数量对齐的加宽无注意力基线。

    与 CPDI_ND 的区别：
        1. 不含自注意力模块（跳过跨对交互）
        2. 编码器更宽（num_filters 48）且更深（5 个 InceptionResBlock）
        3. 分类头首层更宽（hidden1=224）
        —— 三项合计使总参数量与完整模型对齐，从而控制"参数容量"这一混淆因素。

    forward 数据流:
        pair_data (B, K, 64)
          → view(B, K, 4, 16) → reshape(B*K, 4, 16)
          → CipherPairEncoder（共享编码器，逐对独立编码）
          → (B, K, d_model) → AvgPool ⊕ MaxPool → (B, 2*d_model)
          → 分类头 MLP → (B, 1)

    注意：无注意力模型不做 cross-pair 交互——这正是对照实验要检验的变量。
    """

    def __init__(self, k=32, d_model=64, dropout=0.3, enc_blocks=DEFAULT_ENC_BLOCKS,
                 num_filters=DEFAULT_NUM_FILTERS, hidden1=DEFAULT_HIDDEN1,
                 hidden2=DEFAULT_HIDDEN2):
        super().__init__()
        self.k = k
        self.in_channels = 4
        self.encoder_c = CipherPairEncoder(
            num_filters=num_filters, num_blocks=enc_blocks, d_model=d_model
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, hidden1, bias=False),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2, bias=False),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
            nn.Sigmoid()
        )

    def forward(self, pair_data, return_features=False):
        if pair_data.dim() == 3:
            B, K, D = pair_data.shape
            pair_data = pair_data.view(B, K, self.in_channels, D // self.in_channels)
        B, K, C, L = pair_data.shape

        # 逐对独立编码（无注意力交互）
        feats = self.encoder_c(pair_data.reshape(B * K, C, L)).view(B, K, -1)
        fused = torch.cat([feats.mean(dim=1), feats.max(dim=1)[0]], dim=1)
        output = self.classifier(fused)

        if return_features:
            return output, feats
        return output


def build_model(args):
    """按参数构建加宽无注意力模型。"""
    return WideNoAttnCPDI_ND(
        k=args.k, d_model=64, dropout=0.3,
        enc_blocks=args.enc_blocks, num_filters=args.num_filters,
        hidden1=args.hidden1, hidden2=args.hidden2
    )


def count_params(num_filters, enc_blocks, hidden1, hidden2=DEFAULT_HIDDEN2,
                 d_model=64):
    """统计某配置的可训练参数量（不占 GPU）。"""
    model = WideNoAttnCPDI_ND(
        d_model=d_model, enc_blocks=enc_blocks,
        num_filters=num_filters, hidden1=hidden1, hidden2=hidden2
    )
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def search_aligned_configs(target=TARGET_PARAMS, tol=2.5):
    """搜索参数量落在 target±tol% 内的配置，按偏差绝对值升序返回。"""
    found = []
    for f in (32, 40, 48, 56, 64):
        for blocks in (3, 4, 5, 6):
            for h1 in (128, 160, 192, 224, 256, 288):
                n = count_params(f, blocks, h1)
                dev = (n - target) / target * 100.0
                if abs(dev) <= tol:
                    found.append(dict(num_filters=f, enc_blocks=blocks,
                                      hidden1=h1, params=n, dev=dev))
    found.sort(key=lambda r: abs(r['dev']))
    return found


def print_param_table(candidates):
    print(f"参数量对齐候选（目标 = 完整模型 {TARGET_PARAMS:,}；"
          f"原无注意力基线 {ORIG_NOATTN_PARAMS:,}）")
    print('=' * 88)
    print(f"{'num_filters':>11} {'enc_blocks':>10} {'hidden1':>8} "
          f"{'params':>10} {'deviation':>10}  说明")
    print('-' * 88)
    for idx, c in enumerate(candidates):
        note = ''
        if idx == 0:
            note = '← 最严格对齐'
        if (c['num_filters'] == DEFAULT_NUM_FILTERS
                and c['enc_blocks'] == DEFAULT_ENC_BLOCKS
                and c['hidden1'] == DEFAULT_HIDDEN1):
            note += ('  ' if note else '') + '← 本脚本默认'
        print(f"{c['num_filters']:>11} {c['enc_blocks']:>10} {c['hidden1']:>8} "
              f"{c['params']:>10,} {c['dev']:>+9.2f}%  {note}")
    print('=' * 88)


# ============================================================================
# 多种子汇总
# ============================================================================

def write_summary(results, run_dir, args, model_params):
    """写 summary.txt：模型对齐信息 + 训练结果 + K泛化矩阵 + 平均值/方差。"""
    accs = np.array([r['best_val_acc'] for r in results])
    losses = np.array([r['val_loss'] for r in results])
    n = len(results)
    ks = sorted(results[0]['test'].keys()) if results[0].get('test') else []
    col_w = 10
    width = max(70, 6 + 3 + col_w * len(ks)) if ks else 70

    dev = (model_params - TARGET_PARAMS) / TARGET_PARAMS * 100.0

    lines = []
    lines.append('CPDI-ND 参数量对齐的加宽无注意力基线 —— 多种子训练汇总')
    lines.append('=' * width)
    lines.append(f"配置: 轮数={args.num_rounds}, k={args.k}, 模式={args.mode}, "
                 f"差分={args.diff}, 训练样本={args.train_samples}, "
                 f"epochs={args.epochs}, batch={args.batch_size}")
    lines.append(f"模型: 加宽无注意力基线（num_filters={args.num_filters}, "
                 f"enc_blocks={args.enc_blocks}, 分类头={args.hidden1}→{args.hidden2}）")
    lines.append(f"参数量: {model_params:,} | 完整模型: {TARGET_PARAMS:,} | "
                 f"偏差: {dev:+.2f}% | 原无注意力基线: {ORIG_NOATTN_PARAMS:,}")
    if getattr(args, 'seed_list', None):
        lines.append(f"种子（--seeds 显式指定）: {args.seed_list}")
    else:
        lines.append(f"种子（每次运行随机抽取）: 共 {n} 个（见下表）")
    if ks:
        lines.append(f"测试: 每个测试K的样本数={args.test_samples}（正负各半）, "
                     f"数据种子={base.TEST_DATA_SEED}（固定，所有模型共用同一测试集，"
                     "可复现），差分/轮数与训练相同")
        lines.append(f"测试K（K泛化扫描）: {ks}")
    lines.append("说明: 训练/验证数据由训练种子确定性派生（在线模式按 (种子,epoch,worker)，"
                 "每个 epoch 数据全新且同种子跨运行逐位一致），种子同时控制参数"
                 "初始化、数据加载顺序与 cudnn 确定性算法选择。负样本始终为随机差分。")
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

    # ---- K泛化测试矩阵 ----
    if ks:
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
# 单种子训练
# ============================================================================

def run_one_seed(args, seed, delta_target, device, run_dir):
    """训练一个种子的加宽无注意力基线，返回该次运行的指标。"""
    use_contrastive, use_diversity, lam1, lam2, desc = _resolve_loss_config(args.mode)
    print(f"\nRunning {desc}  |  seed = {seed}")

    model = build_model(args)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {num_params:,}  "
          f"(完整模型 {TARGET_PARAMS:,}, 偏差 {(num_params - TARGET_PARAMS) / TARGET_PARAMS * 100:+.2f}%)")

    train_loader = base.get_cpdi_dataloader(
        args.train_samples, args.num_rounds, diff=delta_target,
        k=args.k, random_neg=True, batch_size=args.batch_size,
        num_workers=4, batch_gen_size=args.batch_gen_size,
        online=args.online,
        data_seed=seed,
        rng=None if args.online else np.random.default_rng([seed, base._RNG_TAG_TRAIN_FULL])
    )
    val_loader = base.get_cpdi_dataloader(
        args.val_samples, args.num_rounds, diff=delta_target,
        k=args.k, random_neg=True, batch_size=args.batch_size,
        num_workers=4, batch_gen_size=args.batch_gen_size,
        online=False,                      # 验证集始终全量加载
        data_seed=seed, rng=np.random.default_rng([seed, base._RNG_TAG_VAL])
    )

    save_p = os.path.join(run_dir, f'model_{seed}.pth')
    log_p = os.path.join(run_dir, f'log_{seed}.csv')

    t_start = time.time()
    model, train_process = trainer.train_cpdi_nd(
        model, train_loader, val_loader,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        high_lr=1e-3, low_lr=1e-4,
        lam1=lam1, lam2=lam2,
        save_path=save_p, log_path=log_p
    )
    train_time = time.time() - t_start

    criterion = nn.BCELoss()
    model.to(device)
    val_loss, val_acc, _, _, _ = trainer.evaluate(model, val_loader, criterion, device)
    print(f"\nseed={seed} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return dict(
        seed=seed,
        best_val_acc=float(train_process['val_acc'].max()),
        final_val_acc=float(val_acc),
        val_loss=float(val_loss),
        train_time_s=train_time,
    )


def _resolve_loss_config(mode):
    """损失配置（与 main.py 的 exp1–exp4 定义一致）。"""
    table = {
        'exp1': (False, False, 0.0, 0.0, 'Exp-1: Base model (BCE only)'),
        'exp2': (True, False, 0.1, 0.0, 'Exp-2: Base model + Contrastive'),
        'exp3': (True, True, 0.1, 0.05, 'Exp-3: Base model + Contrastive + Diversity'),
        'exp4': (False, True, 0.0, 0.05, 'Exp-4: Base model + Diversity (no contrastive)'),
    }
    if mode not in table:
        raise ValueError(f"未知模式: {mode}，可选: exp1, exp2, exp3, exp4")
    return table[mode]


# ============================================================================
# 命令行参数解析
# ============================================================================

def parse_diff(diff_str):
    """解析差分字符串 "hex_high,hex_low"。"""
    parts = diff_str.split(',')
    return (int(parts[0], 16), int(parts[1], 16))


def parse_args():
    parser = argparse.ArgumentParser(
        description='CPDI-ND 参数量对齐的加宽无注意力基线'
    )

    # --- 数据与差分（与 main.py 保持一致） ---
    parser.add_argument('--diff', type=str, default='0x0040,0x0000',
                        help='目标差分，格式: hex_high,hex_low（默认 0x0040,0x0000，'
                             '论文 8 轮实验请用 0x2800,0x0010）')
    parser.add_argument('--num-rounds', type=int, default=7,
                        help='Speck 加密轮数（8 轮论文实验传 8）')
    parser.add_argument('--k', type=int, default=32,
                        help='每个样本的密文对数量')
    parser.add_argument('--train-samples', type=int, default=1000000,
                        help='训练样本数（正负各半）')
    parser.add_argument('--val-samples', type=int, default=100000,
                        help='验证样本数')
    parser.add_argument('--test-samples', type=int, default=100000,
                        help='每个测试K的测试集样本数；设 0 跳过 K 扫描')

    # --- 训练（与 main.py 保持一致） ---
    parser.add_argument('--batch-size', type=int, default=2000,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--online', action='store_true',
                        help='在线生成训练数据（大样本防内存溢出）')
    parser.add_argument('--batch-gen-size', type=int, default=2000,
                        help='在线模式下每次生成的样本数')

    # --- 损失模式 ---
    parser.add_argument('--mode', type=str, default='exp2',
                        choices=['exp1', 'exp2', 'exp3', 'exp4'],
                        help='损失函数模式（与 main.py 定义一致，默认 exp2）')

    # --- 加宽基线结构（参数量对齐） ---
    parser.add_argument('--num-filters', type=int, default=DEFAULT_NUM_FILTERS,
                        help=f'编码器卷积通道数（默认 {DEFAULT_NUM_FILTERS}；原完整模型为 32）')
    parser.add_argument('--enc-blocks', type=int, default=DEFAULT_ENC_BLOCKS,
                        help=f'编码器 InceptionResBlock 层数（默认 {DEFAULT_ENC_BLOCKS}；原为 3）')
    parser.add_argument('--hidden1', type=int, default=DEFAULT_HIDDEN1,
                        help=f'分类头第一层宽度（默认 {DEFAULT_HIDDEN1}；原为 128）')
    parser.add_argument('--hidden2', type=int, default=DEFAULT_HIDDEN2,
                        help=f'分类头第二层宽度（默认 {DEFAULT_HIDDEN2}）')
    parser.add_argument('--search-params', action='store_true',
                        help='只打印参数量对齐候选配置并退出（不训练）')

    # --- 运行与保存 ---
    parser.add_argument('--gpu', type=str, default='4',
                        help='GPU 设备 ID（默认 "4"）')
    parser.add_argument('--save-dir', type=str, default='',
                        help='保存目录（默认按差分自动命名: result/<差分高16位>）')

    # --- 多种子重复（与 main.py 一致） ---
    parser.add_argument('--num-seeds', type=int, default=1,
                        help='用随机种子重复训练的次数（默认1）')
    parser.add_argument('--seeds', type=str, default='',
                        help='显式指定随机种子，逗号分隔（如 "42,43,44"）；'
                             '数量必须与 --num-seeds 一致，否则拒绝运行')

    args = parser.parse_args()

    # --- --seeds 与 --num-seeds 一致性检查 ---
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

    # 参数量对齐检查/搜索模式（不训练）
    if args.search_params:
        print_param_table(search_aligned_configs())
        return

    model_params = count_params(args.num_filters, args.enc_blocks,
                                args.hidden1, args.hidden2)
    dev = (model_params - TARGET_PARAMS) / TARGET_PARAMS * 100.0
    print("=" * 80)
    print("CPDI-ND 参数量对齐的加宽无注意力基线")
    print("=" * 80)
    print(f"模型配置: num_filters={args.num_filters}, enc_blocks={args.enc_blocks}, "
          f"分类头={args.hidden1}→{args.hidden2}")
    print(f"参数量: {model_params:,} | 完整模型: {TARGET_PARAMS:,} | "
          f"偏差: {dev:+.2f}% | 原无注意力基线: {ORIG_NOATTN_PARAMS:,}")
    if abs(dev) > 2.0:
        print(f"警告: 参数量偏差 {dev:+.2f}% 超出 ±2% 对齐范围，"
              "建议用 --search-params 选择其他配置")

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    delta_target = parse_diff(args.diff)

    if not args.save_dir:
        args.save_dir = os.path.join('result', f'{delta_target[0]:04x}')
    run_dir = os.path.join(args.save_dir,
                           f'{args.num_rounds}_k{args.k}_wide-{args.mode}')
    os.makedirs(run_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Save dir: {run_dir}")

    seeds = (args.seed_list if args.seed_list is not None
             else [secrets.randbelow(2 ** 31 - 1) for _ in range(args.num_seeds)])
    print(f"Seeds ({'specified' if args.seed_list is not None else 'random'}): {seeds}")

    # =========================================================================
    # 逐种子训练
    # =========================================================================
    results = []
    for i, seed in enumerate(seeds):
        print(f"\n{'=' * 70}")
        print(f"Run {i + 1}/{args.num_seeds}  |  seed = {seed}")
        print(f"{'=' * 70}")
        base.set_global_seed(seed)
        results.append(run_one_seed(args, seed, delta_target, device, run_dir))

    # =========================================================================
    # 统一 K 泛化测试（与 main.py 完全一致）
    # =========================================================================
    if args.test_samples > 0:
        test_ks = sorted(set([4, 8, 16, 32, 64, 128] + [args.k]))
        print(f"\n{'=' * 70}")
        print(f"Testing all {len(results)} models across K = {test_ks}")
        print(f"  ({args.test_samples} samples per K, data seed = {base.TEST_DATA_SEED})")
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
        for tk in test_ks:
            pairs, labels = base.generate_test_set(
                args.test_samples, args.num_rounds, delta_target, tk)
            for r in results:
                acc, tpr, tnr = base.evaluate_model_on(
                    test_models[r['seed']], pairs, labels, args.batch_size, device)
                r['test'][tk] = dict(acc=acc, tpr=tpr, tnr=tnr)
                print(f"  seed={r['seed']}  K={tk:>3}: acc={acc:.4f}  "
                      f"(TPR={tpr:.4f}, TNR={tnr:.4f})")
            del pairs, labels

    write_summary(results, run_dir, args, model_params)
    print("\nDone!")


if __name__ == '__main__':
    main()
