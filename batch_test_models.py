"""
批量测试已保存的模型（一键版）
======================================================================
自动解析 result/<差分>/<轮数>_k<训练k>_<模式>[_no-attention]/model_<种子>.pth
的路径信息（差分、轮数、训练 k、损失模式、是否有注意力、是否为加宽基线），
自动匹配对应轮数与差分的测试数据，批量输出 K = 4/8/16/32/64/128 的
准确率、TPR、TNR。

特点:
    - 无需手动指定差分/轮数：从目录名解析
    - 每个测试 K 的固定测试集只生成一次（数据种子固定，可复现），所有模型共用
    - 支持有注意力、无注意力、加宽无注意力（wide-*）三类模型
    - 输出控制台表格 + CSV/JSON 文件

用法:
    # 测 result/2800 下所有目录，每目录取 1 个模型（默认）
    python batch_test_models.py --root result/2800

    # 只测指定目录模式（子串匹配），每目录取 3 个种子
    python batch_test_models.py --root result/2800 --only "8_k32_exp2,8_k32_exp2_no-attention" \
        --limit-per-dir 3

    # 直接指定若干模型文件（逗号分隔）
    python batch_test_models.py \
        --models result/2800/8_k32_exp2/model_1612946774.pth,result/2800/8_k32_exp2_no-attention/model_2088613111.pth

    # 自定义测试 K 与每个 K 的样本数
    python batch_test_models.py --root result/2800 --k-list 8,32,128 --test-samples 50000

输出:
    result/batch_test/<时间戳>/results.csv   （每个模型 × 每个 K 的 acc/TPR/TNR）
    result/batch_test/<时间戳>/results.json
"""

import os
import re
import glob
import json
import time
import argparse
from datetime import datetime

import pandas as pd
import torch

import main as base
from cpdi_nd_model import CPDI_ND
from model_wide_noattn_train import WideNoAttnCPDI_ND


# 目录名 → 差分（result/2800 → (0x2800, 0x0010)）
# 若出现未登记的差分目录，请在 --diff 中显式指定
DIFF_HIGH_MAP = {
    0x2800: 0x0010,
    0x0040: 0x0000,
}

# 目录名格式: <轮数>_k<训练k>_<模式>[_no-attention]
DIR_RE = re.compile(r'^(\d+)_k(\d+)_(wide-exp\d|exp\d)(_no-attention)?$')


def parse_args():
    p = argparse.ArgumentParser(description='批量测试已保存的模型（自动解析路径与测试数据）')
    p.add_argument('--root', type=str, default='result/2800',
                   help='模型根目录（默认为差分目录，如 result/2800）')
    p.add_argument('--only', type=str, default='',
                   help='只测试目录名包含这些子串的目录，逗号分隔（默认全部）')
    p.add_argument('--models', type=str, default='',
                   help='直接指定模型文件，逗号分隔（指定后忽略 --root/--only）')
    p.add_argument('--limit-per-dir', type=int, default=1,
                   help='每个目录最多测几个模型（默认 1；设 0 表示全部）')
    p.add_argument('--require-log', action='store_true',
                   help='只测有配套 log_<种子>.csv 的模型（排除训练中断产物）')
    p.add_argument('--k-list', type=str, default='4,8,16,32,64,128',
                   help='测试 K 列表，逗号分隔（默认 4,8,16,32,64,128）')
    p.add_argument('--test-samples', type=int, default=100000,
                   help='每个测试 K 的样本总数（正负各半，默认 100000）')
    p.add_argument('--batch-size', type=int, default=2048,
                   help='前向批大小（默认 2048）')
    p.add_argument('--diff', type=str, default='',
                   help='覆盖差分（格式 0x2800,0x0010）；默认从目录名解析')
    p.add_argument('--attn-layers', type=int, default=3,
                   help='注意力层数（默认 3，用于加载有注意力模型）')
    p.add_argument('--wide-filters', type=int, default=48,
                   help='加宽基线的编码器通道数（默认 48，用于加载 wide-* 模型）')
    p.add_argument('--wide-blocks', type=int, default=5,
                   help='加宽基线的编码器块数（默认 5）')
    p.add_argument('--wide-hidden1', type=int, default=224,
                   help='加宽基线分类头首层宽度（默认 224）')
    p.add_argument('--gpu', type=str, default='0', help='GPU 设备 ID（默认 0）')
    p.add_argument('--output-root', type=str, default='result/batch_test',
                   help='结果输出根目录（默认 result/batch_test）')
    return p.parse_args()


def parse_model_path(path, override_diff=None):
    """从模型路径解析出测试所需的配置。

    期望路径: <root>/<差分4位hex>/<轮数>_k<训练k>_<模式>[_no-attention]/model_<种子>.pth
    返回 None 表示无法解析。
    """
    path = os.path.normpath(path)
    model_file = os.path.basename(path)
    model_dir = os.path.basename(os.path.dirname(path))
    diff_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))

    m = DIR_RE.match(model_dir)
    if not m:
        return None
    rounds, train_k, mode, no_attn_suffix = m.groups()
    wide = mode.startswith('wide-')

    if override_diff:
        parts = override_diff.split(',')
        diff = (int(parts[0], 16), int(parts[1], 16))
    else:
        try:
            high = int(diff_dir, 16)
        except ValueError:
            return None
        if high not in DIFF_HIGH_MAP:
            print(f"  [跳过] 未登记的差分目录 '{diff_dir}'，请用 --diff 显式指定")
            return None
        diff = (high, DIFF_HIGH_MAP[high])

    return dict(
        path=path,
        file=model_file,
        group=model_dir,
        rounds=int(rounds),
        train_k=int(train_k),
        mode=mode,
        wide=wide,
        use_attention=(not wide and no_attn_suffix is None),
        diff=diff,
    )


def build_model(info, args):
    """按解析出的配置构建模型结构（参数形状必须与保存的 state_dict 一致）。"""
    if info['wide']:
        return WideNoAttnCPDI_ND(
            k=info['train_k'], d_model=64, dropout=0.3,
            enc_blocks=args.wide_blocks, num_filters=args.wide_filters,
            hidden1=args.wide_hidden1, hidden2=64
        )
    return CPDI_ND(
        k=info['train_k'], d_model=64, n_heads=4, dropout=0.3,
        enc_blocks=3, num_filters=32,
        attn_layers=args.attn_layers,
        use_attention=info['use_attention']
    )


def _has_training_log(info):
    """判断模型是否有配套的训练日志（用于排除训练中断留下的孤儿模型）。"""
    d = os.path.dirname(info['path'])
    seed = info['file'].replace('model_', '').replace('.pth', '')
    return os.path.exists(os.path.join(d, f'log_{seed}.csv'))


def collect_models(args):
    """收集待测模型列表。"""
    if args.models:
        paths = [s.strip() for s in args.models.split(',') if s.strip()]
        items = []
        for p in paths:
            info = parse_model_path(p, args.diff or None)
            if info is None:
                print(f"  [跳过] 无法解析路径: {p}")
                continue
            items.append(info)
        return items

    roots = glob.glob(os.path.join(args.root, '*/model_*.pth'))
    if not roots:
        roots = glob.glob(os.path.join(args.root, 'model_*.pth'))
    # 排除三阶段训练的中间产物（model_<种子>_stage1.pth / _stage2.pth）
    roots = [p for p in roots if '_stage' not in os.path.basename(p)]
    roots = sorted(roots)

    only = [s.strip() for s in args.only.split(',') if s.strip()]
    by_dir = {}
    for p in roots:
        info = parse_model_path(p, args.diff or None)
        if info is None:
            continue
        if only and not any(s in info['group'] for s in only):
            continue
        by_dir.setdefault(info['group'], []).append(info)

    items = []
    for group in sorted(by_dir):
        # 优先挑选有配套训练日志的模型（排除中断产生的孤儿模型），再按文件名排序
        models = sorted(by_dir[group],
                        key=lambda x: (not _has_training_log(x), x['file']))
        if args.require_log:
            models = [m for m in models if _has_training_log(m)]
        if args.limit_per_dir > 0:
            models = models[:args.limit_per_dir]
        items.extend(models)
    return items


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    k_list = sorted({int(s) for s in args.k_list.split(',') if s.strip()})

    print('=' * 100)
    print('批量模型测试（自动解析路径 + 自动匹配测试数据）')
    print('=' * 100)
    print(f"设备: {device} | 测试K: {k_list} | 每K样本数: {args.test_samples} | "
          f"数据种子: {base.TEST_DATA_SEED}")

    infos = collect_models(args)
    if not infos:
        print('未找到可测试的模型。')
        return

    print(f"\n共 {len(infos)} 个模型：")
    for it in infos:
        print(f"  [{it['group']}] {it['file']}  "
              f"→ {it['rounds']}轮, 训练k={it['train_k']}, "
              f"差分={hex(it['diff'][0])}/{hex(it['diff'][1])}, "
              f"{'加宽无注意力' if it['wide'] else ('有注意力' if it['use_attention'] else '无注意力')}")

    # ---- 加载所有模型（失败则记录并跳过） ----
    models = {}
    for it in infos:
        key = it['path']
        try:
            m = build_model(it, args)
            state = torch.load(it['path'], map_location='cpu', weights_only=False)
            m.load_state_dict(state)
            models[key] = m.to(device).eval()
        except Exception as e:
            print(f"  [加载失败] {it['path']}: {e}")

    if not models:
        print('没有成功加载的模型。')
        return

    # ---- 按 (轮数, 差分) 分组，逐 K 生成一次测试集 ----
    groups = {}
    for it in infos:
        if it['path'] in models:
            groups.setdefault((it['rounds'], it['diff']), []).append(it)

    rows = []
    t_all = time.time()
    for (rounds, diff), group_items in sorted(groups.items()):
        print(f"\n{'=' * 100}")
        print(f"测试组: {rounds} 轮 | 差分 {hex(diff[0])}/{hex(diff[1])} | "
              f"{len(group_items)} 个模型")
        print('=' * 100)
        for K in k_list:
            t0 = time.time()
            print(f"\n[K={K}] 生成固定测试集 ({args.test_samples} 样本) ...")
            pairs, labels = base.generate_test_set(
                args.test_samples, rounds, diff, K)
            print(f"      shape={pairs.shape}, 生成耗时 {time.time() - t0:.1f}s")

            for it in group_items:
                m = models[it['path']]
                acc, tpr, tnr = base.evaluate_model_on(
                    m, pairs, labels, args.batch_size, device)
                rows.append(dict(
                    group=it['group'],
                    model=it['file'],
                    path=it['path'],
                    rounds=it['rounds'],
                    train_k=it['train_k'],
                    mode=it['mode'],
                    attention=('wide-no-attn' if it['wide']
                               else ('attn' if it['use_attention'] else 'no-attn')),
                    diff=f"{it['diff'][0]:04x}/{it['diff'][1]:04x}",
                    test_K=K,
                    acc=acc, tpr=tpr, tnr=tnr,
                ))
                print(f"      {it['group']:<34} {it['file']:<26} "
                      f"acc={acc:.4f} (TPR={tpr:.4f}, TNR={tnr:.4f})")
            del pairs, labels

    # ---- 输出 ----
    df = pd.DataFrame(rows)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(args.output_root, ts)
    os.makedirs(out_dir, exist_ok=True)

    csv_p = os.path.join(out_dir, 'results.csv')
    df.to_csv(csv_p, index=False)
    with open(os.path.join(out_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(config=dict(k_list=k_list, test_samples=args.test_samples,
                                   data_seed=base.TEST_DATA_SEED),
                       rows=rows), f, ensure_ascii=False, indent=2)

    # 控制台：按模型透视 acc / TPR / TNR
    def pivot_table(metric, title):
        if df.empty:
            return
        pt = df.pivot_table(index=['group', 'model'], columns='test_K',
                            values=metric, aggfunc='first')
        print(f"\n{'=' * 100}")
        print(title)
        print('=' * 100)
        print(pt.to_string(float_format=lambda v: f'{v:.4f}'))

    pivot_table('acc', '准确率 acc（行=模型，列=测试K）')
    pivot_table('tpr', 'TPR（正类准确率）')
    pivot_table('tnr', 'TNR（负类准确率）')

    print(f"\n总耗时 {time.time() - t_all:.1f}s")
    print(f"结果已保存: {csv_p}")
    print(f"             {os.path.join(out_dir, 'results.json')}")


if __name__ == '__main__':
    main()
