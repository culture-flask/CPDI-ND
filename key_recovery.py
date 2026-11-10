"""
CPDI-ND2 密钥恢复攻击
======================================================================
基于 n 轮神经区分器的 (1+n+1) 轮密钥恢复攻击。

攻击原理:
    区分器能判断 n 轮密文对是否来自目标差分 Δ。
    如果我们猜测最后一轮的轮密钥 sk，剥掉最后一轮后，
    正确的 sk 会让密文退化为 n 轮加密，区分器输出高分；
    错误的 sk 会产生随机效果，区分器输出接近 0.5。

攻击流程 (1+n+1 轮):
    1. 选择 c 组 k 对明文对，差分为 Δ = (0x2800, 0x0010)
    2. 用 (0x0000, 0x0000) 作为密钥解密一轮所有明文（预白化）
    3. 用同一真实密钥加密 (n+2) 轮，得到密文对
    4. 对每个猜测密钥 sk ∈ [0, 65535]:
       a. dec_one_round(密文, sk)  ← 剥掉最后一轮
       b. 对 c 组密文对分别计算交叉差分
       c. 喂给 n 轮 CPDI-ND2 区分器
       d. c 组分数取平均
    5. 分数最高的 sk 即为猜测密钥
    6. 重复 num-trials 次实验，统计成功率

参数说明:
    --c:           同一次攻击中使用的 k 组数（共享同一密钥）
                   c 越大，单次攻击的区分度越高
    --num-trials:  独立实验次数（每次使用不同的随机密钥）

使用方法:
    # 7轮区分器 → 9轮密钥恢复攻击，10组，重复5次
    python key_recovery.py --model-path save_model_exp1/exp1_model.pth \
        --c 10 --num-trials 5 --gpu 0

    # 指定更多参数
    python key_recovery.py --model-path save_model_exp1/exp1_model.pth \
        --num-rounds 7 --c 50 --k 32 --num-trials 10 --gpu 0
"""

import os
import argparse
import json
import time
import numpy as np
import torch
from cpdi_nd_model import CPDI_ND
import speck as sp

_log_file = None


def log_print(msg=''):
    print(msg)
    if _log_file is not None:
        _log_file.write(msg + '\n')
        _log_file.flush()


def parse_args():
    parser = argparse.ArgumentParser(description='CPDI-ND2 Key Recovery Attack')
    parser.add_argument('--model-path', type=str, required=True,
                        help='n 轮区分器的模型权重路径 (.pth)')
    parser.add_argument('--gpu', type=str, default='2',
                        help='GPU 设备 ID')
    parser.add_argument('--num-rounds', type=int, default=8,
                        help='区分器训练的轮数 n（攻击总轮数为 n+2）')
    parser.add_argument('--c', type=int, default=10,
                        help='同一次攻击中使用的 (k,m) 组数（共享同一密钥，c 越大区分度越高）')
    parser.add_argument('--num-trials', type=int, default=5,
                        help='独立实验次数（每次使用不同的随机密钥）')
    parser.add_argument('--k', type=int, default=32,
                        help='每个样本的密文对数量')
    parser.add_argument('--diff', type=str, default='0x2800,0x0010',
                        help='正样本差分，格式: hex_high,hex_low')
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--enc-blocks', type=int, default=3)
    parser.add_argument('--num-filters', type=int, default=32)
    parser.add_argument('--no-attention', action='store_true',
                        help='禁用自注意力模块（消融实验用，默认开启）')
    parser.add_argument('--batch-size', type=int, default=16384,
                        help='密钥猜测的批处理大小（控制显存占用）')
    parser.add_argument('--top-k', type=int, default=10,
                        help='攻击成功阈值：正确密钥排名 <= top-k 即认为攻击成功')
    parser.add_argument('--log-dir', type=str, default='result/key_recovery_logs',
                        help='攻击结果日志保存目录')
    return parser.parse_args()


def parse_diff(diff_str):
    parts = diff_str.split(',')
    return (int(parts[0], 16), int(parts[1], 16))


def generate_one_trial(c, k, diff, total_rounds):
    """生成一次密钥恢复实验所需的密文对数据。

    c 组密文对共享同一个随机密钥。

    预白化逻辑（与 HND_pt 对齐）：
      1. 随机生成 1 轮加密后的输出 C0
      2. 令 C1 = C0 ⊕ Δ，确保 1 轮加密后差分严格等于 Δ
      3. 用 k=0 反向解密得到明文 P0, P1
      4. 这样 encrypt(P0, k=0) ⊕ encrypt(P1, k=0) = C0 ⊕ C1 = Δ ✓

    参数:
        c:            同一次攻击的 (k,m) 组数
        k:            每组的密文对数量
        diff:         输入差分 (Δ_high, Δ_low)
        total_rounds: 总加密轮数 (n+2)

    返回:
        ct0l, ct0r: 形状 (c, k) 的密文 C 整数数组
        ct1l, ct1r: 形状 (c, k) 的密文 C' 整数数组
        last_round_key: 最后一轮轮密钥（标量）
    """
    N = c * k

    # 先随机生成 1 轮加密后的输出，再构造差分
    after_round1_0l = np.frombuffer(sp.urandom(2 * N), dtype=np.uint16).reshape(c, k)
    after_round1_0r = np.frombuffer(sp.urandom(2 * N), dtype=np.uint16).reshape(c, k)
    after_round1_1l = after_round1_0l ^ diff[0]
    after_round1_1r = after_round1_0r ^ diff[1]

    # 用 k=0 反向解密一轮，得到真正的明文
    # 这样 encrypt(P0, k=0) = after_round1_0, encrypt(P1, k=0) = after_round1_1
    # 且 after_round1_0 ⊕ after_round1_1 = Δ ✓
    zero_key = np.zeros((4, N), dtype=np.uint16)
    ks_pre = sp.expand_key(zero_key, 1)
    p0l_flat = after_round1_0l.reshape(-1)
    p0r_flat = after_round1_0r.reshape(-1)
    p1l_flat = after_round1_1l.reshape(-1)
    p1r_flat = after_round1_1r.reshape(-1)
    p0l_flat, p0r_flat = sp.decrypt((p0l_flat, p0r_flat), ks_pre)
    p1l_flat, p1r_flat = sp.decrypt((p1l_flat, p1r_flat), ks_pre)

    # 生成 1 个随机密钥，所有 c 组共享
    keys = np.frombuffer(sp.urandom(8), dtype=np.uint16).reshape(4, 1)
    ks = sp.expand_key(keys, total_rounds)
    last_round_key = int(ks[-1][0])

    # 将同一个密钥扩展到 c*k 对明文
    keys_expanded = np.repeat(keys, N, axis=1)
    ks_expanded = sp.expand_key(keys_expanded, total_rounds)

    ct0l, ct0r = sp.encrypt((p0l_flat, p0r_flat), ks_expanded)
    ct1l, ct1r = sp.encrypt((p1l_flat, p1r_flat), ks_expanded)

    ct0l = ct0l.reshape(c, k)
    ct0r = ct0r.reshape(c, k)
    ct1l = ct1l.reshape(c, k)
    ct1r = ct1r.reshape(c, k)

    return ct0l, ct0r, ct1l, ct1r, last_round_key


def compute_scores_for_guesses(model, ct0l, ct0r, ct1l, ct1r, k, device, batch_size):
    """对所有 2^16 个猜测密钥计算区分器分数。

    c 组密文对共享同一密钥，对每个猜测 sk 分别处理 c 组后取平均分数。

    参数:
        model:        CPDI-ND 区分器
        ct0l...ct1r:  形状 (c, k) 的密文整数数组
        k:            密文对数量
        device:       计算设备
        batch_size:   批处理大小

    返回:
        scores:     形状 (65536,) 的分数数组
        time_breakdown: 字典，包含 decrypt_time/convert_time/infer_time 三个分项耗时
    """
    c = ct0l.shape[0]
    WORD = sp.WORD_SIZE()
    all_sk = np.arange(65536, dtype=np.uint16)

    scores = np.zeros(65536, dtype=np.float32)
    decrypt_time = 0.0
    convert_time = 0.0
    infer_time = 0.0

    model.eval()
    with torch.no_grad():
        for start in range(0, 65536, batch_size):
            end = min(start + batch_size, 65536)
            sk_batch = all_sk[start:end]
            b = end - start

            batch_scores = np.zeros(b, dtype=np.float32)

            for ci in range(c):
                ct0l_exp = np.tile(ct0l[ci], (b, 1))
                ct0r_exp = np.tile(ct0r[ci], (b, 1))
                ct1l_exp = np.tile(ct1l[ci], (b, 1))
                ct1r_exp = np.tile(ct1r[ci], (b, 1))
                sk_exp = np.tile(sk_batch.reshape(b, 1), (1, k))

                # dec_one_round 向量化
                t0 = time.time()
                d0l = ct0l_exp ^ sk_exp
                d0r = ct0r_exp.copy()
                d0r = d0r ^ ct0l_exp
                d0r = sp.ror(d0r, sp.BETA())
                d0l = (d0l - d0r) & sp.MASK_VAL
                d0l = sp.rol(d0l, sp.ALPHA())

                d1l = ct1l_exp ^ sk_exp
                d1r = ct1r_exp.copy()
                d1r = d1r ^ ct1l_exp
                d1r = sp.ror(d1r, sp.BETA())
                d1l = (d1l - d1r) & sp.MASK_VAL
                d1l = sp.rol(d1l, sp.ALPHA())
                decrypt_time += time.time() - t0

                # 通道构造 + 二值转换（4 通道 [cL, cR, cL', cR']）
                t0 = time.time()
                channels = [d0l, d0r, d1l, d1r]

                pair_data = sp.convert_to_binary([ch.reshape(-1) for ch in channels])
                pair_data = pair_data.reshape(b, k, 4 * WORD)
                convert_time += time.time() - t0

                # 模型推理
                t0 = time.time()
                pair_t = torch.from_numpy(pair_data).float().to(device)
                output = model(pair_t)
                probs = output.cpu().numpy().squeeze()
                probs = np.clip(probs, 1e-6, 1 - 1e-6)
                batch_scores += np.log2(probs / (1 - probs))
                infer_time += time.time() - t0

            scores[start:end] = batch_scores / c

    return scores, {'decrypt': decrypt_time, 'convert': convert_time, 'infer': infer_time}


def main():
    global _log_file

    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    delta_target = parse_diff(args.diff)
    total_rounds = args.num_rounds + 2

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f'k{args.k}_attack_log_{timestamp}.log')
    _log_file = open(log_path, 'w', encoding='utf-8')

    log_print(f'攻击实验日志 - {timestamp}')
    log_print('=' * 80)
    log_print(f'加载模型：{args.model_path}')

    model = CPDI_ND(
        k=args.k, d_model=args.d_model, n_heads=args.n_heads,
        dropout=args.dropout, enc_blocks=args.enc_blocks,
        num_filters=args.num_filters,
        use_attention=not args.no_attention
    )

    state_dict = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_print(f'模型加载完成！参数量: {num_params:,}')

    config = {
        'experiment_id': timestamp,
        'attack_rounds': total_rounds,
        'distinguisher_rounds': args.num_rounds,
        'model_path': args.model_path,
        'c': args.c,
        'k': args.k,
        'use_attention': not args.no_attention,
        'num_trials': args.num_trials,
        'batch_size': args.batch_size,
        'rank_success_threshold': args.top_k,
        'base_diff': [delta_target[0], delta_target[1]],
        'device': str(device)
    }

    log_print('')
    log_print('=' * 80)
    log_print(f'开始{args.num_trials}次{total_rounds}轮密钥恢复攻击测试...')
    log_print(f'攻击配置：{json.dumps(config, indent=2, ensure_ascii=False)}')
    log_print('=' * 80)

    correct_count = 0
    rank_list = []
    time_list = []
    gen_time_list = []
    decrypt_time_list = []
    convert_time_list = []
    infer_time_list = []

    for trial in range(args.num_trials):
        log_print('')
        log_print(f'第{trial+1}次攻击：')

        gen_start = time.time()
        ct0l, ct0r, ct1l, ct1r, true_key = generate_one_trial(
            args.c, args.k, delta_target, total_rounds
        )
        gen_time = time.time() - gen_start
        log_print(f'真实最后一轮子密钥: {hex(true_key)}')
        log_print(f'数据生成与加密耗时: {gen_time:.3f}s')

        scores, tb = compute_scores_for_guesses(
            model, ct0l, ct0r, ct1l, ct1r,
            args.k, device, args.batch_size
        )
        trial_time = gen_time + tb['decrypt'] + tb['convert'] + tb['infer']

        sorted_indices = np.argsort(-scores)
        rank = int(np.where(sorted_indices == true_key)[0][0])
        guessed_key = int(sorted_indices[0])

        is_success = rank < args.top_k
        if is_success:
            log_print(f'✅ 攻击成功！猜测密钥: {hex(guessed_key)}，正确密钥排名：{rank+1}，'
                      f'总耗时: {trial_time:.1f}s (生成: {gen_time:.2f}s, 解密: {tb["decrypt"]:.1f}s, '
                      f'转换: {tb["convert"]:.1f}s, 推理: {tb["infer"]:.1f}s)')
            correct_count += 1
        else:
            log_print(f'❌ 攻击失败！猜测密钥: {hex(guessed_key)}，正确密钥排名：{rank+1}，'
                      f'总耗时: {trial_time:.1f}s (生成: {gen_time:.2f}s, 解密: {tb["decrypt"]:.1f}s, '
                      f'转换: {tb["convert"]:.1f}s, 推理: {tb["infer"]:.1f}s)')

        rank_list.append(rank + 1)
        time_list.append(trial_time)
        gen_time_list.append(gen_time)
        decrypt_time_list.append(tb['decrypt'])
        convert_time_list.append(tb['convert'])
        infer_time_list.append(tb['infer'])

    success_rate = correct_count / args.num_trials
    avg_rank = np.mean(rank_list)
    median_rank = np.median(rank_list)
    min_rank = min(rank_list)
    max_rank = max(rank_list)

    log_print('')
    log_print('=' * 80)
    log_print(f'攻击测试完成，总次数：{args.num_trials}')
    log_print(f'攻击成功率：{success_rate*100:.2f}%')
    log_print(f'正确密钥平均排名：{avg_rank:.1f}')
    log_print(f'正确密钥中位数排名：{median_rank:.1f}')
    log_print(f'正确密钥最佳排名：{min_rank}')
    log_print(f'正确密钥最差排名：{max_rank}')
    log_print(f'单次攻击平均耗时：{np.mean(time_list):.1f}s '
              f'(生成: {np.mean(gen_time_list):.3f}s, '
              f'解密: {np.mean(decrypt_time_list):.1f}s, '
              f'转换: {np.mean(convert_time_list):.1f}s, '
              f'推理: {np.mean(infer_time_list):.1f}s)')
    log_print(f'总耗时：{sum(time_list):.1f}s '
              f'(生成: {sum(gen_time_list):.1f}s, '
              f'解密: {sum(decrypt_time_list):.1f}s, '
              f'转换: {sum(convert_time_list):.1f}s, '
              f'推理: {sum(infer_time_list):.1f}s)')
    log_print('=' * 80)

    log_print('')
    log_print(f'📝 日志文件：{log_path}')

    _log_file.close()
    _log_file = None


if __name__ == '__main__':
    main()
