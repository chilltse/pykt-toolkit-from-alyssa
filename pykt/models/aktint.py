import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_sgap(concepts, max_gap=300):
    """
    计算到下一个相同概念出现的间隔（Successor Gap, sgap）
    
    这个函数用于计算每个位置到下一个相同概念出现的距离。
    sgap反映了"未来重复的临近性"：如果相同概念很快会再次出现，sgap值较小，
    说明该概念的记忆可能因为即将重复而得到强化。
    
    例如，对于序列 [A, B, A, C, B]：
    - 位置0的A：下一个A在位置2，sgap = 2
    - 位置1的B：下一个B在位置4，sgap = 3
    - 位置2的A：没有下一个A，sgap = max_gap-1
    
    参数:
        concepts: 形状为 [batch_size, seq_len] 的张量，包含概念ID
                  每个元素代表一个学习项目对应的概念
        max_gap: 最大间隔值，用于限制sgap的上限（默认300）
                 如果到下一个相同概念的距离超过max_gap，则设为max_gap-1
    
    返回:
        sgap: 形状为 [batch_size, seq_len] 的张量，包含每个位置的间隔值
              值越大表示距离下一个相同概念越远（或不存在）
    
    算法说明:
        1. 初始化sgap矩阵，默认值为max_gap-1（表示没有找到下一个相同概念）
        2. 对每个批次和每个位置，查找后续序列中第一个相同概念的位置
        3. 计算间隔距离，并限制在max_gap-1以内
    """
    """
    For each position, calculate gap to NEXT occurrence of same concept.

    Args:
        concepts: Tensor of shape [batch_size, seq_len] containing concept IDs
        max_gap: Maximum gap value to cap at (default 300)

    Returns:
        sgap: Tensor of shape [batch_size, seq_len] with gap values
    """
    batch_size, seq_len = concepts.shape
    sgap = torch.full((batch_size, seq_len), max_gap - 1, dtype=torch.long, device=concepts.device)

    for b in range(batch_size):
        for i in range(seq_len):
            concept = concepts[b, i].item()
            # Find next occurrence of same concept
            for j in range(i + 1, seq_len):
                if concepts[b, j].item() == concept:
                    sgap[b, i] = min(j - i, max_gap - 1)
                    break

    return sgap


def calculate_pcount(concepts):
    """
    计算自上次相同概念出现以来的项目数（Predecessor Count, pcount）
    
    这个函数用于计算每个位置自上次相同概念出现以来，中间经历了多少个不同的学习项目。
    pcount反映了"干扰程度"：如果自上次出现以来经历了很多其他概念，pcount值较大，
    说明该概念的记忆可能因为干扰而减弱。
    
    例如，对于序列 [A, B, C, A, D, B]：
    - 位置0的A：首次出现，pcount = 0
    - 位置1的B：首次出现，pcount = 0
    - 位置2的C：首次出现，pcount = 0
    - 位置3的A：上次在位置0，中间有2个项目（B, C），pcount = 3 - 0 = 3
    - 位置4的D：首次出现，pcount = 0
    - 位置5的B：上次在位置1，中间有3个项目（C, A, D），pcount = 5 - 1 = 4
    
    参数:
        concepts: 形状为 [batch_size, seq_len] 的张量，包含概念ID
    
    返回:
        pcount: 形状为 [batch_size, seq_len] 的张量，包含每个位置的计数
                0表示该概念首次出现，大于0表示自上次出现以来的项目数
    
    算法说明:
        1. 使用字典记录每个概念最后出现的位置
        2. 对每个位置，如果该概念之前出现过，计算当前位置与上次位置的差值
        3. 如果首次出现，则pcount为0
    """
    """
    For each position, count items since LAST occurrence of same concept.

    Args:
        concepts: Tensor of shape [batch_size, seq_len] containing concept IDs

    Returns:
        pcount: Tensor of shape [batch_size, seq_len] with count values
    """
    batch_size, seq_len = concepts.shape
    pcount = torch.zeros((batch_size, seq_len), dtype=torch.long, device=concepts.device)

    for b in range(batch_size):
        concept_last_pos = {}
        for i in range(seq_len):
            concept = concepts[b, i].item()
            if concept in concept_last_pos:
                pcount[b, i] = i - concept_last_pos[concept]
            else:
                pcount[b, i] = 0  # First occurrence
            concept_last_pos[concept] = i

    return pcount


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class InterferenceAddNorm(nn.Module):
    """
    干扰信息的残差连接和层归一化模块
    
    将干扰信息（sgap, pcount）编码为特征，然后通过残差连接添加到注意力输出中。
    这样即使干扰信息无效，模型也能回退到原始行为。
    
    类似AddNorm的设计，确保模型加上干扰机制后不会变得更差。
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)
        
        # 将干扰指标编码为特征
        # sgap和pcount都是标量，需要映射到d_model维度
        self.interference_proj = nn.Sequential(
            nn.Linear(2, d_model // 4),  # sgap + pcount -> d_model//4
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model),  # -> d_model
            nn.Dropout(dropout)
        )

        # 门控网络：决定是否使用干扰信息
        # 输出值在[0,1]，0表示不使用，1表示完全使用
        # self.gate_network = nn.Sequential(
        #     nn.Linear(2, d_model // 8),
        #     nn.ReLU(),
        #     nn.Linear(d_model // 8, 1),
        #     nn.Sigmoid()  # 确保门控值在[0,1]
        # )
        
        # 可学习的缩放因子，初始化为接近0，让模型可以学习是否使用干扰信息
        self.interference_scale = nn.Parameter(torch.zeros(1))
        
    def forward(self, X, interference_info=None):
        """
        参数:
            X: 原始注意力输出 [batch_size, seq_len, d_model]
            interference_info: 干扰信息字典，包含 'sgap' 和 'pcount'
                             [batch_size, seq_len]
        
        返回:
            output: 增强后的输出 [batch_size, seq_len, d_model]
        """
        if interference_info is None:
            return self.ln(X) # 如果没有干扰信息，直接返回归一化的原始输出
        # if True: # 测试，原始模型
        #     return X
        
        sgap = interference_info.get("sgap", None)
        pcount = interference_info.get("pcount", None)
        
        if sgap is None or pcount is None:
            return self.ln(X)
        
        # 确保sgap和pcount是2D张量 [batch_size, seq_len]
        if sgap.dim() == 1:
            sgap = sgap.unsqueeze(0)
        if pcount.dim() == 1:
            pcount = pcount.unsqueeze(0)
        
        # 确保维度匹配X的序列长度
        batch_size, seq_len, d_model = X.shape
        if sgap.size(1) != seq_len:
            # 如果维度不匹配，截断或填充
            if sgap.size(1) > seq_len:
                sgap = sgap[:, :seq_len]
                pcount = pcount[:, :seq_len]
            else:
                # 填充（使用最后一个值）
                pad_len = seq_len - sgap.size(1)
                sgap = torch.cat([sgap, sgap[:, -1:].expand(-1, pad_len)], dim=1)
                pcount = torch.cat([pcount, pcount[:, -1:].expand(-1, pad_len)], dim=1)
        
        # 归一化干扰指标
        # sgap_max = sgap.max()
        # pcount_max = pcount.max()
        # sgap_norm = sgap.float() / (sgap_max + 1e-6) if sgap_max > 0 else sgap.float()
        # pcount_norm = pcount.float() / (pcount_max + 1e-6) if pcount_max > 0 else pcount.float()
        # 改进的归一化：使用更稳定的方式
        # 使用tanh归一化，将值映射到[-1,1]，然后缩放到[0,1]
        sgap_mean = sgap.float().mean()
        sgap_std = sgap.float().std() + 1e-6
        sgap_norm = (sgap.float() - sgap_mean) / sgap_std
        sgap_norm = (torch.tanh(sgap_norm) + 1) / 2  # 映射到[0,1]
        
        pcount_mean = pcount.float().mean()
        pcount_std = pcount.float().std() + 1e-6
        pcount_norm = (pcount.float() - pcount_mean) / pcount_std
        pcount_norm = (torch.tanh(pcount_norm) + 1) / 2  # 映射到[0,1]
        
        # 拼接干扰指标 [batch_size, seq_len, 2]
        interference_input = torch.stack([sgap_norm, pcount_norm], dim=-1)

        # 计算门控值：基于干扰信息本身决定是否使用
        # gate_value = self.gate_network(interference_input)  # [batch_size, seq_len, 1]
        
        # 编码为特征 [batch_size, seq_len, d_model]
        interference_features = self.interference_proj(interference_input)
        
        # 双路径融合：
        # 主路径：X（保持不变）
        # 融合：X + interference_scale * interference_features
        Y = X + self.interference_scale * self.dropout(interference_features)
        
        # 层归一化
        return self.ln(Y)


class AKTInt(nn.Module):
    """
    AKT with Interference-based forgetting (AKTInt) - 基于干扰遗忘的AKT模型
    
    这是AKT（Adaptive Knowledge Tracing）模型的扩展版本，引入了干扰遗忘机制。
    模型能够同时考虑时间衰减和干扰衰减，更准确地模拟人类学习中的遗忘过程。
    
    核心创新:
        1. 时间衰减（Temporal Decay）：基于时间距离的遗忘
           - 使用gamma参数控制时间衰减率
           - 距离越远的历史信息，权重衰减越大
        
        2. 干扰衰减（Interference Decay）：基于概念间干扰的遗忘
           - sgap: 到下一个相同概念出现的间隔（反映未来重复的临近性）
           - pcount: 自上次相同概念出现以来的项目数（反映干扰程度）
           - 使用beta参数控制干扰衰减率
    
    模型架构:
        1. 嵌入层：将问题ID、答案等转换为向量表示
        2. Transformer编码器：使用多头注意力机制处理序列
        3. 输出层：多层全连接网络进行最终预测
    
    输入:
        - q_data: 问题/概念ID序列 [batch_size, seq_len]
        - target: 学生回答（正确/错误）序列 [batch_size, seq_len]
        - pid_data: 问题ID序列（可选）[batch_size, seq_len]
        - dgaps: 干扰数据字典，包含sgaps和pcounts
    
    输出:
        - preds: 预测概率 [batch_size, seq_len]
        - c_reg_loss: 正则化损失（用于控制问题难度参数的复杂度）
    """
    """AKT with Interference-based forgetting (AKTInt)

    This model extends AKT with interference decay that captures:
    - sgap: Gap to next occurrence of same concept (recency of future repetition)
    - pcount: Count since last occurrence (interference amount)
    """
    def __init__(self, n_question, n_pid, d_model, n_blocks, dropout, d_ff=256,
            kq_same=1, final_fc_dim=512, num_attn_heads=8, separate_qa=False, l2=1e-5, emb_type="qid", emb_path="", pretrain_dim=768):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        self.model_name = "aktint"
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l2 = l2
        self.model_type = "akt"  # Use akt architecture type
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        embed_l = d_model
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid+1, 1)
            self.q_embed_diff = nn.Embedding(self.n_question+1, embed_l)
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l)

        if emb_type.startswith("qid"):
            self.q_embed = nn.Embedding(self.n_question, embed_l)
            if self.separate_qa:
                self.qa_embed = nn.Embedding(2*self.n_question+1, embed_l)
            else:
                self.qa_embed = nn.Embedding(2, embed_l)

        # Architecture Object
        self.model = ArchitectureInt(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff, kq_same=self.kq_same, model_type=self.model_type, emb_type=self.emb_type)

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )
        # 添加可学习的输出偏置，用于校准概率分布
        # 初始化为0，模型可以通过训练自动学习最优的偏置值
        # 这有助于解决AUC提高但ACC降低的问题（概率分布偏移）
        self.output_bias = nn.Parameter(torch.zeros(1))
        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid+1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)
        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_embed_data = self.qa_embed(target)+q_embed_data
        return q_embed_data, qa_embed_data

    def forward(self, q_data, target, pid_data=None, qtest=False, dgaps=None):
        """
        Forward pass with interference-based forgetting.

        Args:
            q_data: Question/concept IDs [bs, seq_len]
            target: Response labels [bs, seq_len]
            pid_data: Problem IDs (optional) [bs, seq_len]
            qtest: If True, return additional concat_q output
            dgaps: Dictionary with interference data:
                   {'sgaps': tensor, 'pcounts': tensor}
                   - sgaps: Gap to next occurrence of same concept [bs, seq_len]
                   - pcounts: Count of items since last occurrence [bs, seq_len]

        Returns:
            If qtest=False: (preds, c_reg_loss)
            If qtest=True: (preds, c_reg_loss, concat_q)
        """
        emb_type = self.emb_type
        if emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)

        pid_embed_data = None
        if self.n_pid > 0:
            q_embed_diff_data = self.q_embed_diff(q_data)
            pid_embed_data = self.difficult_param(pid_data)
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data

            qa_embed_diff_data = self.qa_embed_diff(target)
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * (qa_embed_diff_data+q_embed_diff_data)
            c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2
        else:
            c_reg_loss = 0.

        # Extract interference data
        sgap = dgaps.get("sgaps", None) if dgaps is not None else None
        pcount = dgaps.get("pcounts", None) if dgaps is not None else None

        d_output = self.model(q_embed_data, qa_embed_data, pid_embed_data,
                              sgap=sgap, pcount=pcount)

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        # 添加可学习的偏置来校准概率分布
        # 这可以帮助模型自动调整输出，使预测概率分布更合理
        # 从而在保持高AUC的同时提高ACC
        output = output + self.output_bias
        m = nn.Sigmoid()
        preds = m(output)
        if not qtest:
            return preds, c_reg_loss
        else:
            return preds, c_reg_loss, concat_q


class ArchitectureInt(nn.Module):
    def __init__(self, n_question, n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type, emb_type):
        super().__init__()
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {'akt'}:
            self.blocks_1 = nn.ModuleList([
                TransformerLayerInt(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, emb_type=emb_type)
                for _ in range(n_blocks)
            ])
            self.blocks_2 = nn.ModuleList([
                TransformerLayerInt(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, emb_type=emb_type)
                for _ in range(n_blocks*2)
            ])

    def forward(self, q_embed_data, qa_embed_data, pid_embed_data,
                sgap=None, pcount=None):
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        # encoder
        for block in self.blocks_1:
            y = block(mask=1, query=y, key=y, values=y, pdiff=pid_embed_data,
                     sgap=sgap, pcount=pcount)
        flag_first = True
        for block in self.blocks_2:
            if flag_first:
                x = block(mask=1, query=x, key=x,
                          values=x, apply_pos=False, pdiff=pid_embed_data,
                          sgap=sgap, pcount=pcount)
                flag_first = False
            else:
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True, pdiff=pid_embed_data,
                          sgap=sgap, pcount=pcount)
                flag_first = True
        return x

class TransformerLayerInt(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, emb_type):
        super().__init__()
        kq_same = kq_same == 1
        self.masked_attn_head = MultiHeadAttentionInt(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same, emb_type=emb_type)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True, pdiff=None,
                sgap=None, pcount=None):
        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True, pdiff=pdiff,
                sgap=sgap, pcount=pcount)
        else:
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False, pdiff=pdiff,
                sgap=sgap, pcount=pcount)

        query = query + self.dropout1((query2))
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)
        return query


class MultiHeadAttentionInt(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True, emb_type="qid"):
        super().__init__()
        self.d_model = d_model
        self.emb_type = emb_type
        if emb_type.endswith("avgpool"):
            pool_size = 3
            self.pooling = nn.AvgPool1d(pool_size, stride=1, padding=pool_size//2, count_include_pad=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        elif emb_type.endswith("linear"):
            self.linear = nn.Linear(d_model, d_model, bias=bias)
            self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        elif emb_type.startswith("qid"):
            self.d_k = d_feature
            self.h = n_heads
            self.kq_same = kq_same

            self.v_linear = nn.Linear(d_model, d_model, bias=bias)
            self.k_linear = nn.Linear(d_model, d_model, bias=bias)
            if kq_same is False:
                self.q_linear = nn.Linear(d_model, d_model, bias=bias)
            self.dropout = nn.Dropout(dropout)
            self.proj_bias = bias
            self.out_proj = nn.Linear(d_model, d_model, bias=bias)
            self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
            torch.nn.init.xavier_uniform_(self.gammas)

            # 添加干扰残差模块（替代原来的beta参数）
            self.interference_addnorm = InterferenceAddNorm(d_model, dropout)

            self._reset_parameters()


    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad, pdiff=None, sgap=None, pcount=None):
        bs = q.size(0)

        if self.emb_type.endswith("avgpool"):
            scores = self.pooling(v)
            concat = self.pad_zero(scores, bs, scores.shape[2], zero_pad)
        elif self.emb_type.endswith("linear"):
            scores = self.linear(v)
            concat = self.pad_zero(scores, bs, scores.shape[2], zero_pad)
        elif self.emb_type.startswith("qid"):
            k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
            if self.kq_same is False:
                q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
            else:
                q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
            v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

            k = k.transpose(1, 2)
            q = q.transpose(1, 2)
            v = v.transpose(1, 2)
            gammas = self.gammas
            if self.emb_type.find("pdiff") == -1:
                pdiff = None
            # 计算注意力（不应用干扰衰减，只应用时间衰减）
            scores = attention_int(q, k, v, self.d_k,
                            mask, self.dropout, zero_pad, gammas, pdiff,
                            sgap=None, pcount=None, beta=None)

            concat = scores.transpose(1, 2).contiguous()\
                .view(bs, -1, self.d_model)
            
            # 应用干扰信息的残差连接（AddNorm形式）
            interference_info = None
            if sgap is not None and pcount is not None:
                # 确保sgap和pcount的维度匹配
                seq_len = concat.size(1)
                if sgap.dim() == 2 and sgap.size(1) == seq_len:
                    interference_info = {"sgap": sgap, "pcount": pcount}
            concat = self.interference_addnorm(concat, interference_info)

        output = self.out_proj(concat)

        return output

    def pad_zero(self, scores, bs, dim, zero_pad):
        if zero_pad:
            pad_zero = torch.zeros(bs, 1, dim).to(device)
            scores = torch.cat([pad_zero, scores[:, 0:-1, :]], dim=1)
        return scores


def attention_int(q, k, v, d_k, mask, dropout, zero_pad, gamma=None, pdiff=None,
              sgap=None, pcount=None, beta=None):
    """
    注意力计算（只应用时间衰减，干扰信息在输出层通过残差连接添加）
    
    注意：sgap, pcount, beta 参数保留以保持接口兼容性，但不再在此函数中使用。
    干扰信息现在通过 InterferenceAddNorm 模块在 MultiHeadAttentionInt 中处理。

    Args:
        q, k, v: Query, Key, Value tensors
        d_k: Dimension per head
        mask: Attention mask
        dropout: Dropout layer
        zero_pad: Whether to zero-pad first position
        gamma: Learnable temporal decay parameter
        pdiff: Problem difficulty (optional)
        sgap: Gap to next occurrence of same concept [bs, seq_len] (已废弃，保留以兼容)
        pcount: Count of items since last occurrence [bs, seq_len] (已废弃，保留以兼容)
        beta: Learnable interference decay parameter [n_heads, 1, 1] (已废弃，保留以兼容)
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    # Temporal decay calculation
    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32) # 凡是 condition == True 的位置，用 value 覆盖
        scores_ = F.softmax(scores_, dim=-1)
        scores_ = scores_ * mask.float().to(device)
        distcum_scores = torch.cumsum(scores_, dim=-1)
        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
        position_effect = torch.abs(x1-x2)[None, None, :, :].type(torch.FloatTensor).to(device)
        dist_scores = torch.clamp((disttotal_scores-distcum_scores)*position_effect, min=0.)
        dist_scores = dist_scores.sqrt().detach()

    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)

    if pdiff == None:
        temporal_effect = torch.clamp(torch.clamp(
            (dist_scores*gamma).exp(), min=1e-5), max=1e5)
    else:
        diff = pdiff.unsqueeze(1).expand(pdiff.shape[0], dist_scores.shape[1], pdiff.shape[1], pdiff.shape[2])
        diff = diff.sigmoid().exp()
        temporal_effect = torch.clamp(torch.clamp(
            (dist_scores*gamma*diff).exp(), min=1e-5), max=1e5)

    # 只应用时间衰减，不应用干扰衰减
    # 干扰信息将在输出层通过残差连接添加（InterferenceAddNorm）
    scores = scores * temporal_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]
