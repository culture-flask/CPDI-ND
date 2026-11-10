"""
CPDI-ND 训练与评估模块
======================================================================
本模块是 CPDI-ND 神经区分器方案的训练层，负责模型的训练、验证和微调。

模块组成一览：
  损失函数 ─────────────────────────────────────────────────────────────
    - ContrastiveLoss         : 跨样本对比损失，拉近同类/推远异类
    - DiversityRegularization : 特征多样性正则，防止 k 个密文对学习冗余特征
    - cyclic_lr               : 循环学习率策略（周期性正弦波）

  数据工具 ─────────────────────────────────────────────────────────────
    - CPDI_ND_Dataset        : PyTorch Dataset 封装
    - load_cpdi_data          : 正负样本数据加载入口（调用 speck.py 生成）

  训练函数 ─────────────────────────────────────────────────────────────
    - train_one_epoch         : 单轮训练（BCE + 对比损失 + 多样性正则）
    - evaluate                : 模型评估（BCE + 对比损失 + 多样性正则）
    - train_cpdi_nd          : 标准训练循环（循环学习率 + 模型保存）


依赖：
    numpy, pandas, torch, torch.nn, torch.nn.functional, torch.utils.data, speck
"""

import copy
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import speck as sp


# ============================================================================
# 工具函数：循环学习率
# ============================================================================

def cyclic_lr(num_epochs, high_lr, low_lr):
    """生成循环学习率的 Lambda 函数。
    
    循环学习率（Cyclical Learning Rate）策略让学习率在每个周期内从高到低
    线性衰减，周期结束后重置为最高值。这种策略有助于模型在训练过程中跳出
    局部最优解，找到更好的 loss landscape。

    参数:
        num_epochs: 每个周期的 epoch 数
        high_lr:    周期起始的高学习率值
        low_lr:     周期结束的低学习率值

    返回:
        一个 lambda 函数，输入当前 epoch 索引，输出换算后的 lr_lambda 值。
    """
    res = lambda i: (low_lr + ((num_epochs - 1) - i % num_epochs) / (num_epochs - 1) * (high_lr - low_lr)) / high_lr
    return res


# ============================================================================
# 损失函数 1：对比损失（Contrastive Loss）
# ============================================================================

class ContrastiveLoss(nn.Module):
    """跨样本对比损失 —— 拉近同类样本、推远异类样本。

    密码学直觉：
        正样本（相同输入差分 Δ 的 k 个密文对集合）在表示空间中应形成紧密的簇，
        负样本（不同差分）应远离正样本簇。对比损失显式地优化这个目标。

    实现方式（批次内对比）：
        1. 对每个样本的 k 个密文对特征取均值池化 → 得到 B 个全局特征向量
        2. 计算 B×B 的余弦相似度矩阵，除以温度 τ 得到缩放相似度
        3. 对每个样本 i：
           - 正对：与 i 同类（同标签）的其他样本
           - 负对：与 i 不同类的所有样本
        4. 使用 InfoNCE 风格损失：
           L_i = -log( Σ_{j∈正对} exp(sim_ij/τ) / Σ_{j≠i} exp(sim_ij/τ) )

    参数:
        temperature: 温度参数 τ，控制相似度分布的"锐度"。默认 0.1。
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """计算批次内对比损失。

        参数:
            features: 形状 (B, K, D) 的密码对特征
            labels:   形状 (B, 2) 的独热标签 或 (B,) 的整数标签

        返回:
            标量损失值
        """
        B, K, D = features.shape

        # 对每个样本的 K 个密文对做均值池化
        f_pooled = features.mean(dim=1)

        # L2 归一化
        f_norm = F.normalize(f_pooled, dim=1)

        # 计算缩放相似度矩阵
        sim = torch.mm(f_norm, f_norm.t()) / self.temperature

        # 构建正负标签掩码
        if labels.dim() > 1:
            labels = labels.argmax(dim=1)
        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_mask = pos_mask.float()
        pos_mask.fill_diagonal_(0.0)

        # 计算 InfoNCE 损失
        sim_exp = torch.exp(sim)
        sim_exp = sim_exp - sim_exp.diag().diag()

        pos_sum = (sim_exp * pos_mask).sum(dim=1)
        all_sum = sim_exp.sum(dim=1)

        loss = -torch.log((pos_sum + 1e-8) / (all_sum + 1e-8))

        loss = loss[all_sum > 1e-8]
        if loss.numel() == 0:
            return torch.tensor(0.0, device=features.device)
        return loss.mean()


# ============================================================================
# 损失函数 2：特征多样性正则（Diversity Regularization）
# ============================================================================

class DiversityRegularization(nn.Module):
    """特征多样性正则 —— 鼓励 k 个密文对学习互补的而非冗余的特征。

    L_div = - mean_{i≠j} ||f_i - f_j||_2

    负号是因为我们要最大化距离，而优化器做最小化。

    实现说明（显存关键）：
        直接构造 (B, K, K, D) 的差分张量会随 K 以 O(B·K²·D) 增长——
        B=5000、K=64 时单个张量就约 5.2 GB，加上反向传播的中间量可达十几 GB，
        是 exp3 在 k=64/batch=5000 下爆显存的根因。
        由于特征已 L2 归一化，有恒等式
            ||f_i - f_j||² = 2 - 2·(f_i·f_j)
        改用 (B,K,K) 的点积矩阵计算距离，显存降为 O(B·K²)（约 1/64），
        数值上与逐对相减再取范数等价（float32 误差 ~1e-5）。
    """

    def __init__(self):
        super().__init__()

    def forward(self, features):
        """计算多样性正则损失。

        参数:
            features: 形状 (B, K, D) 的密码对特征

        返回:
            标量正则损失值
        """
        B, K, D = features.shape
        n_pairs = K * (K - 1)
        if n_pairs <= 1:
            return torch.tensor(0.0, device=features.device)

        f = F.normalize(features, dim=2)

        # 归一化后 ||f_i - f_j||² = 2 - 2·cos(f_i, f_j)：只构造 (B,K,K) 矩阵。
        # 注意：对角元 f_i·f_i = 1 使 2-2·sim 恰为 0，而 sqrt 在 0 处梯度为 inf，
        # 会导致反向传播出现 NaN/inf（旧实现用 torch.norm，在 0 处梯度被定义为 0，
        # 因此不会触发）。这里用 1e-12 下限保护，对 loss 的影响可忽略
        # （对角元贡献 1e-6/对，远小于 float32 精度量级）。
        sim = torch.bmm(f, f.transpose(1, 2))
        dist = torch.sqrt(torch.clamp(2.0 - 2.0 * sim, min=1e-12))

        loss = -dist.sum() / (B * n_pairs)
        return loss


# ============================================================================
# 数据工具：PyTorch Dataset 封装
# ============================================================================

class CPDI_ND_Dataset(Dataset):
    """CPDI-ND 数据的 PyTorch Dataset 封装。

    参数:
        pair_data: 形状 (N, K, D) 的密文对数据（numpy uint8）
        labels:    形状 (N, 2) 的独热标签（numpy uint8）
    """

    def __init__(self, pair_data, labels):
        # 以 uint8 存储（float32 的 1/4 内存：k=32、100万样本约 2GB 而非 8GB），
        # float 转换由 DataLoader worker 在 __getitem__ 中并行完成
        self.pair_data = torch.as_tensor(pair_data, dtype=torch.uint8)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.pair_data)

    def __getitem__(self, idx):
        return self.pair_data[idx].float(), self.labels[idx]


# ============================================================================
# 数据加载入口：生成正负混合数据集
# ============================================================================

def load_cpdi_data(num_samples, num_rounds, diff, k=32,
                   diff_neg=(0x0200, 0x0000), positive_ratio=0.5, random_neg=False,
                   shuffle=True, rng=None, shared_key=False):
    """加载 CPDI-ND 格式的训练数据：正样本 + 负样本（可选数组级打乱）。

    参数:
        num_samples:    总样本数
        num_rounds:     Speck 加密轮数
        diff:           正样本输入差分 (Δ_high, Δ_low)
        k:              每个样本的密文对数量，默认 32
        diff_neg:       负样本的固定差分，默认 (0x0200, 0x0000)
        positive_ratio: 正样本占比，默认 0.5
        random_neg:     是否使用随机差分负样本，默认 False
        shuffle:        是否在数组级打乱样本顺序，默认 True。在线分块生成时必须
                        打开（保证每个 batch 内正负样本混合）；全量加载并配合
                        DataLoader(shuffle=True) 时应设为 False，避免对全量
                        数组做一次冗余的完整拷贝（100万样本约多占2GB、耗时1秒+）。
        rng:            可选 numpy.random.Generator。指定时明文/密钥生成与
                        打乱均由该随机源派生（同 rng 跨运行逐位一致，用于可复现
                        训练）；默认 None 使用系统熵源 urandom（每次独立）。
        shared_key:     密钥策略。False（默认）= 每个密文对独立密钥；True = 样本内
                        k 个密文对共享同一密钥（密钥分布对照实验用）。

    返回:
        all_pairs:  (num_samples, k, 64) 的密文对数据
        all_y:      (num_samples, 2) 的独热标签
    """
    print(f"Generating CPDI-ND data: {num_samples} samples, rounds={num_rounds}, k={k}"
          f"{' [shared-key]' if shared_key else ''}")
    print(f"  Positive diff: {hex(diff[0])}/{hex(diff[1])}")
    if random_neg:
        print(f"  Negative diff: RANDOM (P' is completely random)")
    else:
        print(f"  Negative diff: {hex(diff_neg[0])}/{hex(diff_neg[1])}")

    n_pos = int(num_samples * positive_ratio)
    n_neg = num_samples - n_pos

    # 生成正样本：m=None 不生成交叉差分
    print(f"  Generating {n_pos} positive samples...")
    pos_pairs, _, pos_y = sp.make_multi_pair_data(n_pos, num_rounds, diff, k=k, m=None,
                                                  label=1, rng=rng, shared_key=shared_key)

    # 生成负样本
    if random_neg:
        print(f"  Generating {n_neg} negative samples (RANDOM diff)...")
        neg_pairs, _, _ = sp.make_multi_pair_data(n_neg, num_rounds, diff, k=k, m=None,
                                                  label=0, rng=rng, shared_key=shared_key)
    else:
        print(f"  Generating {n_neg} negative samples (fixed diff {hex(diff_neg[0])}/{hex(diff_neg[1])})...")
        neg_pairs, _, _ = sp.make_multi_pair_data(n_neg, num_rounds, diff_neg, k=k, m=None,
                                                  label=1, rng=rng, shared_key=shared_key)
    neg_y = sp.one_hot_encode_binary(np.full(n_neg, 0, dtype=np.uint8))

    # 拼接（正样本在前、负样本在后）
    all_pairs = np.concatenate([pos_pairs, neg_pairs], axis=0)
    all_y = np.concatenate([pos_y, neg_y], axis=0)

    # 数组级打乱：仅在线分块生成时需要（保证每个 batch 内正负混合）。
    # 全量加载时由 DataLoader(shuffle=True) 在批级别洗牌，跳过此处可避免
    # 对全量数组的一次冗余完整拷贝。
    if shuffle:
        if rng is not None:
            indices = rng.permutation(num_samples)
        else:
            indices = np.random.permutation(num_samples)
        all_pairs = all_pairs[indices]
        all_y = all_y[indices]

    print(f"  Data shape - pairs: {all_pairs.shape}, labels: {all_y.shape}")
    return all_pairs, all_y


# ============================================================================
# 训练与评估函数
# ============================================================================

def train_one_epoch(model, dataloader, optimizer, criterion, device,
                    lam1=0.1, lam2=0.05,
                    use_contrastive=True, use_diversity=False):
    """执行一个 epoch 的训练。

    损失构成：
        L_total = L_BCE + λ1 × L_contrast + λ2 × L_div

    返回:
        (train_loss, train_acc): 平均总损失和准确率
    """
    model.train()
    train_loss = 0.0
    train_corrects = 0
    train_num = 0

    contrast_fn = ContrastiveLoss(temperature=0.1) if use_contrastive else None
    diversity_fn = DiversityRegularization() if use_diversity else None

    for b_x, b_y in dataloader:
        b_x, b_y = b_x.to(device), b_y.to(device)

        # 前向传播
        output, features = model(b_x, return_features=True)

        # 计算总损失
        bce = criterion(output, b_y[:, 1:2])
        loss = bce

        if contrast_fn is not None:
            contrast_loss = contrast_fn(features, b_y)
            loss = loss + lam1 * contrast_loss

        if diversity_fn is not None:
            div_loss = diversity_fn(features)
            loss = loss + lam2 * div_loss

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计训练指标
        train_loss += loss.item() * b_x.size(0)
        preds = (output > 0.5).float().squeeze()
        true_labels = b_y[:, 1]
        train_corrects += (preds == true_labels).sum().item()
        train_num += b_x.size(0)

    train_loss = train_loss / train_num
    train_acc = train_corrects / train_num
    return train_loss, train_acc


def evaluate(model, dataloader, criterion, device,
             use_contrastive=False, use_diversity=False,
             lam1=0.1, lam2=0.05):
    """评估模型在验证集上的性能。

    返回:
        (val_loss_bce, val_acc, val_contrast, val_div, val_total)
    """
    model.eval()
    contrast_fn = ContrastiveLoss(temperature=0.1) if use_contrastive else None
    diversity_fn = DiversityRegularization() if use_diversity else None

    val_loss_bce = 0.0
    val_contrast = 0.0
    val_div = 0.0
    val_corrects = 0
    val_num = 0

    with torch.no_grad():
        for b_x, b_y in dataloader:
            b_x, b_y = b_x.to(device), b_y.to(device)

            if use_contrastive or use_diversity:
                output, features = model(b_x, return_features=True)
            else:
                output = model(b_x)
                features = None

            bce = criterion(output, b_y[:, 1:2])
            val_loss_bce += bce.item() * b_x.size(0)

            if contrast_fn is not None and features is not None:
                contrast = contrast_fn(features, b_y)
                val_contrast += contrast.item() * b_x.size(0)

            if diversity_fn is not None and features is not None:
                div = diversity_fn(features)
                val_div += div.item() * b_x.size(0)

            preds = (output > 0.5).float().squeeze()
            true_labels = b_y[:, 1]
            val_corrects += (preds == true_labels).sum().item()
            val_num += b_x.size(0)

    val_loss_bce /= val_num
    val_contrast /= val_num
    val_div /= val_num
    val_acc = val_corrects / val_num
    val_total = val_loss_bce + (lam1 * val_contrast if use_contrastive else 0.0) \
                              + (lam2 * val_div if use_diversity else 0.0)
    return val_loss_bce, val_acc, val_contrast, val_div, val_total


# ============================================================================
# 标准训练循环
# ============================================================================

def train_cpdi_nd(model, train_loader, val_loader, num_epochs=100,
                   batch_size=100, cycle_length=10, high_lr=2e-3, low_lr=1e-4,
                   lam1=0.1, lam2=0.05, save_path='save_model/cpdi_nd_best.pth',
                   log_path='save_model/cpdi_nd_train_log.csv'):
    """CPDI-ND 标准训练循环。

    返回:
        (model, train_process): 加载了最佳权重的模型 + 训练日志 DataFrame
    """
    import os
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=high_lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())

    lr_lambda = cyclic_lr(num_epochs=cycle_length, high_lr=high_lr, low_lr=low_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    use_contrastive = lam1 > 0
    use_diversity = lam2 > 0

    best_acc = 0.0
    train_loss_all, train_acc_all = [], []
    val_loss_all, val_acc_all = [], []
    since = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs - 1}")
        print("-" * 40)
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        epoch_start = time.time()

        # 训练阶段
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            lam1=lam1, lam2=lam2,
            use_contrastive=use_contrastive, use_diversity=use_diversity
        )

        # 验证阶段
        val_loss_bce, val_acc, val_contrast, val_div, val_total = evaluate(
            model, val_loader, criterion, device,
            use_contrastive=use_contrastive, use_diversity=use_diversity,
            lam1=lam1, lam2=lam2
        )

        # 记录指标（统一保存总损失）
        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)
        val_loss_all.append(val_total)
        val_acc_all.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_total:.4f}  Val   Acc: {val_acc:.4f}", end="")
        if use_contrastive:
            print(f"  Contrast: {val_contrast:.4f}", end="")
        if use_diversity:
            print(f"  Div: {val_div:.4f}", end="")
        print(f"  BCE: {val_loss_bce:.4f}")
        print(f"Epoch time: {time.time() - epoch_start:.2f}s  Total: {time.time() - since:.2f}s")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, save_path)
            print(f"  => Best model saved (acc: {best_acc:.4f})")

        scheduler.step()

    print(f"\nTraining complete. Best Val Acc: {best_acc:.4f}")

    # 保存训练日志
    train_process = pd.DataFrame(data={
        "epoch": range(num_epochs),
        "train_loss": train_loss_all,
        "train_acc": train_acc_all,
        "val_loss": val_loss_all,
        "val_acc": val_acc_all,
    })
    train_process.to_csv(log_path, index=False)
    print(f"Training log saved to {log_path}")

    model.load_state_dict(best_model_wts)
    return model, train_process
