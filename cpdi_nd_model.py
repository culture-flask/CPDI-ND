"""
CPDI-ND 神经区分器模型
======================================================================
三阶段流水线：
  Stage 1 → 独立特征提取（CipherPairEncoder）
  Stage 2 → 跨对注意力交互（Self-Attention）
  Stage 3 → 全局池化与融合（AvgPool + MaxPool → Concat）
  Stage 4 → 分类头（MLP → Sigmoid）

关键设计决策：
  - 编码器使用 Flatten + Linear 投影（而非 GAP），保留比特位置信息
  - Speck 的差分偏差是位置相关的，GAP 会平均掉这些关键信号
  - 基线 Gohr 模型也使用 Flatten 保留位置信息

数据流全景（以 B=4, K=32 为例）:

  pair_data (B, K, 64)
       │ view(B,K,4,16)
       ▼
  ┌──────────────────┐
  │ CipherPairEncoder│
  │ 3×InceptionRes   │
  │ Flatten+Linear   │
  └────────┬─────────┘
           │ (B, K, 64)
           ▼
  ┌──────────────────┐
  │  Self-Attention  │
  │  Post-LN         │
  └────────┬─────────┘
           │ (B, K, 64)
           ▼
  ┌──────────────────┐
  │ AvgPool + MaxPool│
  │ → Concat         │
  └────────┬─────────┘
           │ (B, 128)
           ▼
  ┌──────────────────┐
  │  Classifier (MLP)│
  │ 128→128→64→1    │
  │ + Sigmoid        │
  └────────┬─────────┘
           ▼
      P(Y=1) ∈ [0,1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionResBlock(nn.Module):
    """Inception 风格残差块：多尺度卷积（k=1,3,5）+ 瓶颈结构 + pre-activation。

    结构（以 channels=32, bottleneck=16 为例）:

        输入 x: (N, 32, L)
            │
            ▼ BN1 → ReLU                    ← pre-activation：先归一化再卷积
            ▼ Conv1d(32→16, k=1)            ← 瓶颈压缩：32通道→16通道
            ▼ BN2 → ReLU
            ├── Conv1d(16→16, k=1)  ─┐       ← 单比特独立偏差
            ├── Conv1d(16→16, k=3)  ─┼─ cat  ← 相邻3位局部模式（进位链）
            └── Conv1d(16→16, k=5)  ─┘       ← 更宽的5位相关性（模加传播）
                    │
                    ▼ BN_out → ReLU
                    ▼ Conv1d(48→32, k=1)      ← 投影回原通道数
                    ▼ + shortcut (输入直通)    ← 残差连接

    密码学直觉：
        Speck 的差分特征出现在不同尺度上——单比特偏差(k=1)、
        进位链效应(k=3)、模加法传播(k=5)。多尺度并行确保不遗漏任何尺度的特征。
    """

    def __init__(self, channels, bottleneck=16):
        super().__init__()
        # pre-activation 第一层：先归一化+激活，再做瓶颈压缩
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv1 = nn.Conv1d(channels, bottleneck, kernel_size=1, bias=False)
        # pre-activation 第二层：瓶颈空间内的归一化+激活
        self.bn2 = nn.BatchNorm1d(bottleneck)
        self.relu = nn.ReLU(inplace=True)
        # 多尺度并行卷积：在瓶颈空间(16通道)内分别用 k=1,3,5 提取不同感受野的特征
        self.conv_k1 = nn.Conv1d(bottleneck, bottleneck, kernel_size=1, bias=False)
        self.conv_k3 = nn.Conv1d(bottleneck, bottleneck, kernel_size=3, padding=1, bias=False)
        self.conv_k5 = nn.Conv1d(bottleneck, bottleneck, kernel_size=5, padding=2, bias=False)
        # concat 后的归一化 + 投影回原通道数
        self.bn_out = nn.BatchNorm1d(bottleneck * 3)
        self.proj = nn.Conv1d(bottleneck * 3, channels, kernel_size=1, bias=False)

    def forward(self, x):
        shortcut = x
        # pre-activation: BN → ReLU → Conv
        out = self.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.relu(self.bn2(out))
        # 多尺度并行卷积 + 拼接
        k1 = self.conv_k1(out)
        k3 = self.conv_k3(out)
        k5 = self.conv_k5(out)
        out = torch.cat([k1, k3, k5], dim=1)  # (N, 48, L)
        # 投影回原通道数 + 残差连接
        out = self.proj(self.relu(self.bn_out(out)))
        return out + shortcut


class CipherPairEncoder(nn.Module):
    """密文对特征编码器 (E_c)。

    将每个密文对的 64 位（4 通道×16 位）二进制表示编码为 d_model 维特征向量。

    输入: (N, 4, 16) — N 个密文对，4 通道 [cL, cR, cL', cR']
    输出: (N, d_model) — 每个密文对的 d_model 维特征

    结构:
        Conv1d(4→32, k=1) + BN + ReLU               ← 通道混合
        InceptionResBlock × num_blocks                 ← 多尺度特征提取
        Flatten                                        ← (32, 16) → 512 维，保留位置信息
        Linear(512 → d_model)                          ← 投影到低维特征空间

    为什么用 Flatten + Linear 而不是 Global Average Pooling:
        Speck 的差分偏差是比特位置相关的——某些位比其他位更偏。
        GAP 把 16 个位置平均掉，位置信息全部丢失，模型无法学习。
        基线 Gohr 模型也使用 Flatten 保留位置信息。
    """

    def __init__(self, num_filters=32, num_blocks=3, d_model=64, seq_len=16):
        super().__init__()
        # 初始 1×1 卷积：通道混合
        self.initial = nn.Sequential(
            nn.Conv1d(4, num_filters, kernel_size=1, bias=False),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(inplace=True)
        )
        # Inception 残差块堆叠：多尺度特征提取
        blocks = []
        for _ in range(num_blocks):
            blocks.append(InceptionResBlock(num_filters))
        self.res_blocks = nn.Sequential(*blocks)
        # Flatten + Linear：保留比特位置信息（而非 GAP 丢弃位置信息）
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(num_filters * seq_len, d_model)

    def forward(self, x):
        # x: (N, 4, 16) → (N, 32, 16)
        x = self.initial(x)
        # (N, 32, 16) → (N, 32, 16)，多尺度特征提取
        x = self.res_blocks(x)
        # (N, 32, 16) → (N, 512) → (N, d_model)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class CPDI_ND(nn.Module):
    """CPDI-ND 完整模型。

    参数:
        k:              每个样本的密文对数量（默认 32）
        d_model:        特征维度（默认 64）
        n_heads:        注意力头数（默认 4，每头 16 维）
        dropout:        Dropout 概率（默认 0.3）
        enc_blocks:     密文对编码器残差块数（默认 3，接近基线复杂度）
        num_filters:    编码器卷积通道数（默认 32，与基线一致）
        attn_layers:    Self-Attention 层数（默认 3）
        use_attention:  是否启用自注意力模块（默认 True，设为 False 跳过以消融）

    forward 数据流:
        pair_data (B, K, 64)
          → view(B, K, 4, 16)                  拆分为通道×16位
          → reshape(B*K, 4, 16)                展平为独立密文对
          → CipherPairEncoder                   逐对编码 → (B*K, 64)
          → view(B, K, 64)                     恢复样本维度

        Self-Attention × attn_layers:
          Q=K=V=(B, K, 64) → (B, K, 64)    K个密文对互相看

        Pooling:
          AvgPool: mean(dim=1) → (B, 64)              全局平均趋势
          MaxPool: max(dim=1)  → (B, 64)              最显著信号
          Concat → (B, 128)

        Classifier:
          128→128→64→1 + Sigmoid → P(Y=1) ∈ [0,1]
    """

    def __init__(self, k=32, d_model=64, n_heads=4, dropout=0.3,
                 enc_blocks=3, num_filters=32, attn_layers=3,
                 use_attention=True):
        super().__init__()
        self.k = k
        self.in_channels = 4  # 固定 4 通道输入 [cL, cR, cL', cR']
        self.use_attention = use_attention

        # Stage 1: 密文对编码器 — 3个InceptionResBlock
        self.encoder_c = CipherPairEncoder(
            num_filters=num_filters,
            num_blocks=enc_blocks, d_model=d_model
        )

        # Stage 2: Self-Attention — 多层 Post-LN（use_attention=False 时不创建，真正减少参数量）
        if self.use_attention:
            self.attn_layers = nn.ModuleList([
                nn.ModuleDict({
                    'self_attn': nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                    'norm': nn.LayerNorm(d_model),
                    'dropout': nn.Dropout(dropout),
                })
                for _ in range(attn_layers)
            ])

        # Stage 3+4: 全局池化融合 + 分类头
        # 输入维度 = d_model * 2（AvgPool 和 MaxPool 各 d_model 维拼接）
        # 前两层 Linear 使用 bias=False：后面紧跟 BN，BN 自带偏移参数，bias 冗余
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, pair_data, return_features=False):
        # ==================== Stage 1: 密文对编码 ====================
        # pair_data 输入形状: (B, K, 64) 或 (B, K, 4, 16)
        if pair_data.dim() == 3:
            B, K, D = pair_data.shape
            C, L = self.in_channels, D // self.in_channels
            pair_data = pair_data.view(B, K, C, L)
        B, K, C, L = pair_data.shape

        # 展平为 B*K 个独立密文对，送入编码器
        pairs_flat = pair_data.reshape(B * K, C, L)  # (B*K, 4, 16)
        pair_feats = self.encoder_c(pairs_flat)        # (B*K, d_model)
        pair_feats = pair_feats.view(B, K, -1)         # (B, K, d_model)

        # ==================== Stage 2: Self-Attention × attn_layers ====================
        # K个密文对互相关注，捕捉对间统计相关性
        if self.use_attention:
            x = pair_feats
            for layer in self.attn_layers:
                # need_weights=False: 训练/评估均无需返回注意力矩阵，避免物化
                # (B,K,K) 权重张量（大K评估时每层省约0.5GB显存与softmax开销）
                attn_out, _ = layer['self_attn'](x, x, x, need_weights=False)
                x = layer['norm'](x + layer['dropout'](attn_out))  # Post-LN residual
            attended = x
        else:
            attended = pair_feats

        # ==================== Stage 3: 全局池化与融合 ====================
        # AvgPool: K个密文对的平均特征 → 捕获全局统计趋势（对噪声鲁棒）
        f_g = attended.mean(dim=1)   # (B, d_model)
        # MaxPool: K个密文对的最大特征 → 捕获最显著的偏差信号（对强信号敏感）
        f_m = attended.max(dim=1)[0]  # (B, d_model)
        # 拼接：两者互补
        fused = torch.cat([f_g, f_m], dim=1)  # (B, d_model*2)

        # ==================== Stage 4: 分类头 ====================
        output = self.classifier(fused)  # (B, 1)

        if return_features:
            # 返回注意力后的特征，用于计算对比损失和多样性正则
            return output, attended
        return output

    def get_pair_features(self, pair_data):
        """仅提取密文对特征（不经过注意力），用于推理或特征可视化。"""
        if pair_data.dim() == 3:
            B, K, D = pair_data.shape
            pair_data = pair_data.view(B, K, self.in_channels, D // self.in_channels)
        B, K, C, L = pair_data.shape
        pairs_flat = pair_data.view(B * K, C, L)
        pair_feats = self.encoder_c(pairs_flat)
        return pair_feats.view(B, K, -1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = CPDI_ND(k=32, d_model=64).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    B, K = 4, 32
    pair_input = torch.randn(B, K, 64).to(device)
    output, feats = model(pair_input, return_features=True)
    print(f"Output shape: {output.shape}")
    print(f"Features shape: {feats.shape}")
