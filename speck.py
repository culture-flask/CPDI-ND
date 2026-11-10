"""
Speck32/64 分组密码实现 与 CPDI-ND 多密文对数据生成工具
=================================================================
本文件是 CPDI-ND 神经区分器方案的数据层基础，包含三个层次的功能：

第1层 ─ Speck 密码算法核心
  - 轮函数参数常量（字长、旋转量、掩码）
  - 循环移位工具（ROL / ROR）
  - 单轮加解密（enc_one_round / dec_one_round）
  - 密钥扩展（expand_key）
  - 多轮加解密（encrypt / decrypt）
  - 二进制特征展开（convert_to_binary）

第2层 ─ Gohr 原始数据格式
  - make_train_data：单密文对 + 随机标识的二分类数据
    （此为基线方法的训练数据生成器）

第3层 ─ CPDI-ND 多密文对数据格式（本方案核心新增）
  - make_multi_pair_data：生成 k 个密文对 + m 对交叉差分的样本
  - compute_cross_diff_from_pairs：从已有密文对在线计算交叉差分

数据流示意：
  随机明文P ──┬── encrypt(P, key) ──→ 密文 C
              │
  P ⊕ Δ ───┴── encrypt(P⊕Δ, key) ──→ 密文 C'
              │
              └── ΔC = C ⊕ C'（密文差分）
              
  （重复 k 次，每次使用不同的随机密钥，共用一个 Δ）

  然后对 k 个密文对两两计算交叉差分：
    ΔC_i ⊕ ΔC_j  （i ≠ j，共采样 m 对）
"""

import numpy as np
from os import urandom
import random


# ============================================================================
# 第1层：Speck32/64 密码算法核心常量与工具
# ============================================================================

def WORD_SIZE():
    """字长（比特数）。
    
    Speck 系列支持多种块大小，此处固定为 Speck-32/64 变种的字长 16 位。
    块大小 = 2 × WORD_SIZE = 32 位，密钥长度 = 4 × WORD_SIZE = 64 位。
    """
    return 16

def ALPHA():
    """Speck 轮函数中的右旋量 α = 7。"""
    return 7

def BETA():
    """Speck 轮函数中的左旋量 β = 2。"""
    return 2

MASK_VAL = 2 ** WORD_SIZE() - 1
"""16 位全 1 掩码（0xFFFF），用于将所有运算结果截断至 WORD_SIZE 位。"""


# ============================================================================
# 第1层：工具函数（标签编码、数据洗牌、位运算、二进制展开）
# ============================================================================

def one_hot_encode_binary(y):
    """将二分类标签向量 y 转换为独热编码矩阵 (N, 2)。"""
    y = np.asarray(y).reshape(-1).astype(np.uint8)
    return np.eye(2, dtype=np.uint8)[y]


def shuffle_together(l):
    """对多个数组以相同的随机状态进行同步洗牌。"""
    state = np.random.get_state()
    for x in l:
        np.random.set_state(state)
        np.random.shuffle(x)


def rol(x,k):
    """16 位循环左移。"""
    return(((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)))

def ror(x,k):
    """16 位循环右移。"""
    return((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL))


def convert_to_binary(arr):
    """将多个 16 位整数数组展开为按位二进制特征矩阵（向量化版本）。

    参数:
        arr: 列表，每个元素是形状为 (N,) 的 uint16 数组。
             典型输入：[ct0L, ct0R, ct1L, ct1R]

    返回:
        形状为 (N, len_arr × WORD_SIZE) 的 uint8 矩阵。

    性能: 使用 numpy unpackbits 向量化，比逐位循环快约 50 倍。
    """
    len_arr = len(arr)
    WORD = WORD_SIZE()
    # 将所有分量堆叠为 (N, len_arr) 的 uint16 矩阵
    combined = np.stack(arr, axis=1)  # (N, len_arr) uint16
    # 转为 big-endian uint16 字节视图 → (N, len_arr, 2) uint8
    combined_bytes = combined.astype('>u2').view(np.uint8).reshape(-1, len_arr, 2)
    # numpy unpackbits: 每个 uint8 → 8 个 bit，(N, len_arr, 2, 8) → (N, len_arr*16)
    X = np.unpackbits(combined_bytes, axis=2).reshape(-1, len_arr * WORD)
    return X


# ============================================================================
# 第1层：Speck32/64 轮函数与加解密
# ============================================================================

def enc_one_round(p, k):
    """Speck32/64 单轮加密。

    参数:
        p: 二元组 (x, y)，各为 16 位整数或 numpy 数组
        k: 轮密钥，16 位整数或 numpy 数组

    返回:
        (c0, c1): 加密一轮后的新二元组

    轮函数公式:
        x = ROR(x, 7);  x = (x + y) mod 2^16;  x = x ⊕ k
        y = ROL(y, 2);  y = y ⊕ x

    注意: numpy 的整数加法在 uint16 下自动模 2^16，但 Python 原生 int
    会溢出为更大的整数，因此需要手动 & MASK_VAL。
    当输入为 numpy uint16 数组时，加法自动截断，无需额外掩码。
    """
    c0, c1 = p[0], p[1]
    c0 = ror(c0, ALPHA())
    c0 = (c0 + c1) & MASK_VAL
    c0 = c0 ^ k
    c1 = rol(c1, BETA())
    c1 = c1 ^ c0
    return(c0,c1)


def dec_one_round(c,k):
    """Speck32/64 单轮解密（enc_one_round 的逆）。"""
    c0, c1 = c[0], c[1]
    c1 = c1 ^ c0
    c1 = ror(c1, BETA())
    c0 = c0 ^ k
    c0 = (c0 - c1) & MASK_VAL
    c0 = rol(c0, ALPHA())
    return(c0, c1)


def expand_key(k, t):
    """Speck32/64 密钥扩展：由主密钥 k 生成 t 轮的轮密钥序列。

    参数:
        k: 形状为 (4, N) 的 uint16 数组（N 个密钥同时扩展），
           或 (4,) 的单个密钥
        t: 需要的轮密钥数量 = 加密轮数

    返回:
        长度为 t 的列表，每个元素是形状 (N,) 的轮密钥数组
        （如果输入是单个密钥，则每个元素是标量）

    向量化: 支持同时扩展 N 个密钥，利用 numpy 广播机制。
    """
    ks = [0 for i in range(t)]
    ks[0] = k[len(k)-1]
    l = list(reversed(k[:len(k)-1]))
    for i in range(t-1):
        l[i%3], ks[i+1] = enc_one_round((l[i%3], ks[i]), i)
    return(ks)


def encrypt(p, ks):
    """多轮 Speck 加密（支持 numpy 向量化）。

    参数:
        p: 二元组 (x, y)，各为可广播的 uint16 数组 / 标量
        ks: 轮密钥序列（由 expand_key 生成）

    返回:
        (cx, cy): 加密后的二元组

    向量化: 当输入为数组时，numpy 广播使得 N 个明文可以同时加密，
    无需显式 for 循环遍历样本。这是 make_train_data 高效的关键。
    """
    x, y = p[0], p[1]
    for k in ks:
        x,y = enc_one_round((x,y), k)
    return(x, y)


def decrypt(c, ks):
    """多轮 Speck 解密。"""
    x, y = c[0], c[1]
    for k in reversed(ks):
        x, y = dec_one_round((x,y), k)
    return(x,y)


# ============================================================================
# 第2层：Gohr 原始数据格式 ─ 单密文对二分类训练数据生成
# ============================================================================

def make_train_data(n, nr, diff):
    """生成单密文对格式的训练数据（Gohr CRYPTO 2019 基线）。

    参数:
        n:   总样本数（正 + 负各约一半）
        nr:  加密轮数
        diff: 输入差分元组 (Δ_high, Δ_low)

    返回:
        X: 形状 (n, 112) 的二进制特征矩阵（7 通道 × 16 位）
        Y: 形状 (n, 2) 的独热标签矩阵

    性能关键: 使用 numpy 向量化加密，N 个样本一次性加密完成。
    """
    Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1
    keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1)
    plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16)
    plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16)
    plain1l = plain0l ^ diff[0]
    plain1r = plain0r ^ diff[1]
    num_rand_samples = np.sum(Y==0)
    plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16)
    plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16)
    ks = expand_key(keys, nr)

    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks)

    # 计算前一轮部分信息
    prev_R  = ror(ctdata0l ^ ctdata0r, BETA())
    prev_Rp = ror(ctdata1l ^ ctdata1r, BETA())
    d_l = rol(((ctdata0l - prev_R) & MASK_VAL) ^ ((ctdata1l - prev_Rp) & MASK_VAL), ALPHA())

    X = convert_to_binary([prev_R, prev_Rp, d_l, ctdata0l, ctdata0r, ctdata1l, ctdata1r])

    Y = one_hot_encode_binary(Y)
    return(X,Y)


# ============================================================================
# 第3层：CPDI-ND 多密文对数据格式 ─ 向量化实现
# ============================================================================

def make_multi_pair_data(num_samples, nr, diff, k, m=None, label=1, rng=None,
                         shared_key=False):
    """生成 CPDI-ND 多密文对格式的训练数据（向量化版本）。

    数据格式（4 通道 × 16 位）: [C_l, C_r, C_l', C_r']

    参数:
        num_samples: 要生成的样本（批次）数量
        nr:          加密轮数
        diff:        输入差分元组 (Δ_high, Δ_low)
        k:           每个样本包含的密文对数量（建议 32 或 64）
        m:           交叉差分采样数（已废弃，保留接口兼容）
        label:       标签，1=正样本，0=负样本
        rng:         可选 numpy.random.Generator；指定时用其生成随机明文/密钥
                     （用于固定测试集的可复现生成），默认 None 使用系统熵源 urandom
        shared_key:  密钥策略。False（默认）= 每个密文对独立的随机密钥（本文默认协议）；
                     True = 同一个样本内的 k 个密文对共享同一随机密钥
                     （用于检验训练/攻击密钥分布不一致的影响）

    返回:
        pair_data:      形状 (num_samples, k, 64) 的密文对二进制数据
        cross_diff_data: None（已废弃）
        Y:              形状 (num_samples, 2) 的独热标签

    性能优化:
        与旧版的双重 Python for 循环逐个加密不同，本版本将所有
        num_samples × k 对明文视为 N = num_samples × k 个独立加密任务，
        利用 numpy 向量化一次性完成全部加密，速度提升约 100~500 倍。
    """
    WORD = WORD_SIZE()
    N = num_samples * k  # 总加密次数

    # --- 批量生成随机明文和密钥 ---
    if rng is None:
        plain0l = np.frombuffer(urandom(2 * N), dtype=np.uint16)
        plain0r = np.frombuffer(urandom(2 * N), dtype=np.uint16)
        if shared_key:
            # 每个样本只生成 1 个密钥，再沿 k 维复制
            keys = np.frombuffer(urandom(8 * num_samples), dtype=np.uint16).reshape(4, num_samples)
            keys = np.repeat(keys, k, axis=1)
        else:
            keys = np.frombuffer(urandom(8 * N), dtype=np.uint16).reshape(4, -1)
    else:
        # 指定 rng 时用可播种随机源（固定测试集的可复现生成）
        plain0l = rng.integers(0, 65536, size=N, dtype=np.uint16)
        plain0r = rng.integers(0, 65536, size=N, dtype=np.uint16)
        if shared_key:
            keys = rng.integers(0, 65536, size=(4, num_samples), dtype=np.uint16)
            keys = np.repeat(keys, k, axis=1)
        else:
            keys = rng.integers(0, 65536, size=(4, N), dtype=np.uint16)

    # --- 生成第二个明文 P1 ---
    if label == 0 and diff != (0, 0):
        if rng is None:
            plain1l = np.frombuffer(urandom(2 * N), dtype=np.uint16)
            plain1r = np.frombuffer(urandom(2 * N), dtype=np.uint16)
        else:
            plain1l = rng.integers(0, 65536, size=N, dtype=np.uint16)
            plain1r = rng.integers(0, 65536, size=N, dtype=np.uint16)
    else:
        plain1l = plain0l ^ diff[0]
        plain1r = plain0r ^ diff[1]

    # --- 向量化加密：N 对明文同时加密 ---
    ks = expand_key(keys, nr)
    ct0l, ct0r = encrypt((plain0l, plain0r), ks)
    ct1l, ct1r = encrypt((plain1l, plain1r), ks)

    # --- 输出特征：4 通道 [cL, cR, cL', cR'] ---
    channels = [ct0l, ct0r, ct1l, ct1r]

    # --- 整数密文 → 二进制展开 ---
    pair_data = convert_to_binary(channels)
    pair_data = pair_data.reshape(num_samples, k, 4 * WORD)

    Y = one_hot_encode_binary(np.full(num_samples, label, dtype=np.uint8))
    return pair_data, None, Y


def _compute_cross_diff_vectorized(ct0l, ct0r, ct1l, ct1r, num_samples, k, m, WORD):
    """向量化计算交叉差分。

    直接在密文整数上计算 XOR，再统一做二进制展开。
    比逐对恢复整数值→异或→再展开快约 50~100 倍。

    参数:
        ct0l, ct0r, ct1l, ct1r: 形状 (num_samples*k,) 的密文整数数组
        num_samples, k, m, WORD: 形状参数

    返回:
        形状 (num_samples, m, 4*WORD) 的交叉差分二进制数据
    """
    # reshape 为 (num_samples, k)，方便按样本索引
    ct0l = ct0l.reshape(num_samples, k)
    ct0r = ct0r.reshape(num_samples, k)
    ct1l = ct1l.reshape(num_samples, k)
    ct1r = ct1r.reshape(num_samples, k)

    # 预计算所有 m 对的 (i, j) 索引
    i_indices = np.arange(m) % k
    j_indices = (np.arange(m) + np.arange(m) // k + 1) % k
    # 修复 i == j 的情况
    conflict = i_indices == j_indices
    j_indices[conflict] = (i_indices[conflict] + 1) % k

    # 向量化索引：对所有样本同时取第 i 个和第 j 个密文对
    # ct0l[:, i_indices] 形状为 (num_samples, m)
    cd0l = ct0l[:, i_indices] ^ ct0l[:, j_indices]  # (num_samples, m)
    cd0r = ct0r[:, i_indices] ^ ct0r[:, j_indices]
    cd1l = ct1l[:, i_indices] ^ ct1l[:, j_indices]
    cd1r = ct1r[:, i_indices] ^ ct1r[:, j_indices]

    # 展平为 (num_samples * m,) 再做二进制展开
    cd0l_flat = cd0l.reshape(-1)
    cd0r_flat = cd0r.reshape(-1)
    cd1l_flat = cd1l.reshape(-1)
    cd1r_flat = cd1r.reshape(-1)

    cross_diff_data = convert_to_binary([cd0l_flat, cd0r_flat, cd1l_flat, cd1r_flat])
    cross_diff_data = cross_diff_data.reshape(num_samples, m, 4 * WORD)

    return cross_diff_data


def compute_cross_diff_from_pairs(pair_data, m=None):
    """从已有的密文对二进制数据在线计算交叉差分（向量化版本）。

    参数:
        pair_data: numpy 数组，形状 (batch, k, 4*WORD_SIZE)，uint8 二进制格式
        m:         交叉差分采样数，默认 max(k//4, 16)

    返回:
        cross_diff: numpy 数组，形状 (batch, m, 4*WORD_SIZE)，uint8 二进制格式
    """
    num_samples, k, _ = pair_data.shape
    WORD = WORD_SIZE()
    if m is None:
        m = max(k // 4, 16)

    # 预计算索引对
    i_indices = np.arange(m) % k
    j_indices = (np.arange(m) + np.arange(m) // k + 1) % k
    conflict = i_indices == j_indices
    j_indices[conflict] = (i_indices[conflict] + 1) % k

    # 直接在二进制位上做 XOR（向量化）
    # pair_data[:, i_indices, :] 形状 (batch, m, 64)
    cross_diff = pair_data[:, i_indices, :] ^ pair_data[:, j_indices, :]

    return cross_diff
