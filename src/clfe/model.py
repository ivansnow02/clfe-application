import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    BertModel,
    BertTokenizer,
    RobertaModel,
    RobertaTokenizer,
    AutoTokenizer,
    AutoModel,
)
from torch import einsum
from einops import rearrange, repeat
from types import SimpleNamespace


def save_model(save_path, epoch, model, optimizer, filename):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if filename is None:
        filename = f"epoch_{epoch}.pth"
    save_file_path = os.path.join(save_path, filename)
    states = {
        "epoch": epoch + 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(states, save_file_path)
    return save_file_path


def dict_to_namespace(d):
    """Recursively converts a dictionary and its nested dictionaries to a Namespace."""
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return SimpleNamespace(**d)


# ======================================================================
# 学习率调度器 (来自 core/scheduler.py)
# ======================================================================
class GradualWarmupScheduler(_LRScheduler):
    """Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.0:
            raise ValueError("multiplier should be greater thant or equal to 1.")
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [
                base_lr * (float(self.last_epoch) / self.total_epoch)
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = (
            epoch if epoch != 0 else 1
        )  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [
                base_lr
                * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group["lr"] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


def get_scheduler(optimizer, args):
    scheduler_steplr = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=0.9 * args.base.n_epochs
    )
    scheduler_warmup = GradualWarmupScheduler(
        optimizer,
        multiplier=1,
        total_epoch=0.1 * args.base.n_epochs,
        after_scheduler=scheduler_steplr,
    )

    return scheduler_warmup


# ======================================================================
# 评估指标 (来自 core/metric.py)
# ======================================================================


class MetricsTop:
    def __init__(self):
        self.metrics_dict = {
            "MOSI": self.__eval_mosi_regression,
            "MOSEI": self.__eval_mosei_regression,
            "SIMS": self.__eval_sims_regression,
        }

    def __multiclass_acc(self, y_pred, y_true):
        """
        Compute the multiclass accuracy w.r.t. groundtruth
        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

    def __eval_mosei_regression(self, y_pred, y_true, exclude_zero=False):
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()

        test_preds_a7 = np.clip(test_preds, a_min=-3.0, a_max=3.0)
        test_truth_a7 = np.clip(test_truth, a_min=-3.0, a_max=3.0)
        test_preds_a5 = np.clip(test_preds, a_min=-2.0, a_max=2.0)
        test_truth_a5 = np.clip(test_truth, a_min=-2.0, a_max=2.0)
        test_preds_a3 = np.clip(test_preds, a_min=-1.0, a_max=1.0)
        test_truth_a3 = np.clip(test_truth, a_min=-1.0, a_max=1.0)

        mae = np.mean(
            np.absolute(test_preds - test_truth)
        )  # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a7 = self.__multiclass_acc(test_preds_a7, test_truth_a7)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)

        non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
        non_zeros_binary_truth = test_truth[non_zeros] > 0
        non_zeros_binary_preds = test_preds[non_zeros] > 0

        non_zeros_acc2 = accuracy_score(non_zeros_binary_preds, non_zeros_binary_truth)
        non_zeros_f1_score = f1_score(
            non_zeros_binary_truth, non_zeros_binary_preds, average="weighted"
        )

        binary_truth = test_truth >= 0
        binary_preds = test_preds >= 0
        acc2 = accuracy_score(binary_preds, binary_truth)
        f_score = f1_score(binary_truth, binary_preds, average="weighted")

        eval_results = {
            "Has0_acc_2": round(acc2, 4),
            "Has0_F1_score": round(f_score, 4),
            "Non0_acc_2": round(non_zeros_acc2, 4),
            "Non0_F1_score": round(non_zeros_f1_score, 4),
            "Mult_acc_5": round(mult_a5, 4),
            "Mult_acc_7": round(mult_a7, 4),
            "MAE": round(mae, 4),
            "Corr": round(corr, 4),
        }
        return eval_results

    def __eval_mosi_regression(self, y_pred, y_true):
        return self.__eval_mosei_regression(y_pred, y_true)

    def __eval_sims_regression(self, y_pred, y_true):
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()
        test_preds = np.clip(test_preds, a_min=-1.0, a_max=1.0)
        test_truth = np.clip(test_truth, a_min=-1.0, a_max=1.0)

        # two classes{[-1.0, 0.0], (0.0, 1.0]}
        ms_2 = [-1.01, 0.0, 1.01]
        test_preds_a2 = test_preds.copy()
        test_truth_a2 = test_truth.copy()
        for i in range(2):
            test_preds_a2[
                np.logical_and(test_preds > ms_2[i], test_preds <= ms_2[i + 1])
            ] = i
        for i in range(2):
            test_truth_a2[
                np.logical_and(test_truth > ms_2[i], test_truth <= ms_2[i + 1])
            ] = i

        # three classes{[-1.0, -0.1], (-0.1, 0.1], (0.1, 1.0]}
        ms_3 = [-1.01, -0.1, 0.1, 1.01]
        test_preds_a3 = test_preds.copy()
        test_truth_a3 = test_truth.copy()
        for i in range(3):
            test_preds_a3[
                np.logical_and(test_preds > ms_3[i], test_preds <= ms_3[i + 1])
            ] = i
        for i in range(3):
            test_truth_a3[
                np.logical_and(test_truth > ms_3[i], test_truth <= ms_3[i + 1])
            ] = i

        # five classes{[-1.0, -0.7], (-0.7, -0.1], (-0.1, 0.1], (0.1, 0.7], (0.7, 1.0]}
        ms_5 = [-1.01, -0.7, -0.1, 0.1, 0.7, 1.01]
        test_preds_a5 = test_preds.copy()
        test_truth_a5 = test_truth.copy()
        for i in range(5):
            test_preds_a5[
                np.logical_and(test_preds > ms_5[i], test_preds <= ms_5[i + 1])
            ] = i
        for i in range(5):
            test_truth_a5[
                np.logical_and(test_truth > ms_5[i], test_truth <= ms_5[i + 1])
            ] = i

        mae = np.mean(
            np.absolute(test_preds - test_truth)
        )  # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a2 = self.__multiclass_acc(test_preds_a2, test_truth_a2)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        f_score = f1_score(test_truth_a2, test_preds_a2, average="weighted")

        eval_results = {
            "Mult_acc_2": mult_a2,
            "Mult_acc_3": mult_a3,
            "Mult_acc_5": mult_a5,
            "F1_score": f_score,
            "MAE": mae,
            "Corr": corr,  # Correlation Coefficient
        }
        return eval_results

    def getMetics(self, datasetName):
        return self.metrics_dict[datasetName.upper()]


# ======================================================================
# 数据集加载 (来自 core/dataset.py)
# ======================================================================


class MMDataset(Dataset):
    def __init__(self, args, mode="train"):
        self.mode = mode
        # 这里的 self.args 已经是完整的args，无需改动
        self.args = args.dataset
        DATA_MAP = {
            "mosi": self.__init_mosi,
            "mosei": self.__init_mosei,
            "sims": self.__init_sims,
        }
        DATA_MAP[self.args.datasetName]()

    def __init_mosi(self):
        # 这是原始的、高效的加载方式
        with open(self.args.dataPath, "rb") as f:
            data = pickle.load(f)

        self.text = data[self.mode]["text_bert"].astype(
            np.float32
        )  # 直接加载处理好的'text_bert'
        self.vision = data[self.mode]["vision"].astype(np.float32)
        self.audio = data[self.mode]["audio"].astype(np.float32)

        print(
            f"----------------- {self.args.datasetName} {self.mode} -----------------"
        )
        print(f"Language shape: {self.text.shape}")
        print(f"Vision shape: {self.vision.shape}")
        print(f"Audio shape: {self.audio.shape}")
        print(
            "-------------------------------------------------------------------------"
        )

        self.rawText = data[self.mode]["raw_text"]
        self.ids = data[self.mode]["id"]
        self.labels = {
            "M": data[self.mode][self.args.train_mode + "_labels"].astype(np.float32)
        }
        if self.args.datasetName == "sims":
            for m in "TAV":
                self.labels[m] = data[self.mode][self.args.train_mode + "_labels_" + m]

        self.audio[self.audio == -np.inf] = 0

    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()

    def __len__(self):
        return len(self.labels["M"])

    def __getitem__(self, index):
        # 这也是原始的、高效的__getitem__
        sample = {
            "raw_text": self.rawText[index],
            "text": torch.Tensor(self.text[index]),  # 直接从 self.text 中取张量
            "audio": torch.Tensor(self.audio[index]),
            "vision": torch.Tensor(self.vision[index]),
            "index": index,
            "id": self.ids[index],
            "labels": {
                k: torch.Tensor(v[index].reshape(--1)) for k, v in self.labels.items()
            },
        }
        return sample


def MMDataLoader(args):
    datasets = {
        "train": MMDataset(args, mode="train"),
        "valid": MMDataset(args, mode="valid"),
        "test": MMDataset(args, mode="test"),
    }

    dataLoader = {
        ds: DataLoader(
            datasets[ds],
            batch_size=args.base.batch_size,
            num_workers=args.base.num_workers,
            shuffle=True if ds == "train" else False,
        )
        for ds in datasets.keys()
    }

    return dataLoader


# ======================================================================
# 模型层定义 (来自 models/bert.py 和 models/almt_layer.py)
# ======================================================================

# ----------------------------------------------------
# 文本编码器 (来自 models/bert.py)
# ----------------------------------------------------
TRANSFORMERS_MAP = {
    "bert": (BertModel, BertTokenizer),
    "roberta": (RobertaModel, RobertaTokenizer),
    "deberta": (AutoModel, AutoTokenizer),
}


class BertTextEncoder(nn.Module):
    def __init__(self, use_finetune=False, pretrained="bert-base-uncased"):
        super().__init__()

        model_type = "bert"  # Default
        if "roberta" in pretrained.lower():
            model_type = "roberta"
        else:
            model_type = "deberta"

        tokenizer_class = TRANSFORMERS_MAP[model_type][1]
        model_class = TRANSFORMERS_MAP[model_type][0]

        self.tokenizer = tokenizer_class.from_pretrained(pretrained)
        self.model = model_class.from_pretrained(pretrained)
        self.use_finetune = use_finetune

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, text):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        input_ids, input_mask, segment_ids = (
            text[:, 0, :].long(),
            text[:, 1, :].float(),
            text[:, 2, :].long(),
        )
        if self.use_finetune:
            last_hidden_states = self.model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
            )[0]  # Models outputs are now tuples
        else:
            with torch.no_grad():
                last_hidden_states = self.model(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    token_type_ids=segment_ids,
                )[0]  # Models outputs are now tuples
        return last_hidden_states


# ----------------------------------------------------
# ALMT 核心层 (来自 models/almt_layer.py)
# ----------------------------------------------------
class PreNormForward(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNormAttention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, q, k, v, **kwargs):
        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)

        return self.fn(q, k, v)


class PreNormAHL(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, h_t, h_a, h_v, h_hyper):
        h_t = self.norm1(h_t)
        h_a = self.norm2(h_a)
        h_v = self.norm3(h_v)
        h_hyper = self.norm4(h_hyper)

        return self.fn(h_t, h_a, h_v, h_hyper)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, q, k, v):
        b, n, _, h = *q.shape, self.heads
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = self.attend(dots)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)


# class HhyperLearningLayer(nn.Module):
#     def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
#         super().__init__()
#         inner_dim = dim_head *  heads
#         project_out = not (heads == 1 and dim_head == dim)

#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.attend = nn.Softmax(dim = -1)
#         self.to_q = nn.Linear(dim, inner_dim, bias=False)
#         self.to_k_ta = nn.Linear(dim, inner_dim, bias=False)
#         self.to_k_tv = nn.Linear(dim, inner_dim, bias=False)
#         self.to_v_ta = nn.Linear(dim, inner_dim, bias=False)
#         self.to_v_tv = nn.Linear(dim, inner_dim, bias=False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim, bias=True),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()

#     def forward(self, h_t, h_a, h_v, h_hyper):
#         b, n, _, h = *h_t.shape, self.heads

#         q = self.to_q(h_t)
#         k_ta = self.to_k_ta(h_a)
#         k_tv = self.to_k_tv(h_v)
#         v_ta = self.to_v_ta(h_a)
#         v_tv = self.to_v_tv(h_v)

#         q, k_ta, k_tv, v_ta, v_tv = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k_ta, k_tv, v_ta, v_tv))

#         dots_ta = einsum('b h i d, b h j d -> b h i j', q, k_ta) * self.scale
#         attn_ta = self.attend(dots_ta)
#         out_ta = einsum('b h i j, b h j d -> b h i d', attn_ta, v_ta)
#         out_ta = rearrange(out_ta, 'b h n d -> b n (h d)')

#         dots_tv = einsum('b h i d, b h j d -> b h i j', q, k_tv) * self.scale
#         attn_tv = self.attend(dots_tv)
#         out_tv = einsum('b h i j, b h j d -> b h i d', attn_tv, v_tv)
#         out_tv = rearrange(out_tv, 'b h n d -> b n (h d)')

#         h_hyper_shift = self.to_out(out_ta + out_tv)
#         h_hyper += h_hyper_shift

#         return h_hyper


# class HhyperLearningEncoder(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, dropout = 0.):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 PreNormAHL(dim, HhyperLearningLayer(dim, heads = heads, dim_head = dim_head, dropout = dropout))
#             ]))

#     def forward(self, h_t_list, h_a, h_v, h_hyper):
#         for i, attn in enumerate(self.layers):
#             h_hyper = attn[0](h_t_list[i], h_a, h_v, h_hyper)
#         return h_hyper


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNormAttention(
                        dim,
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                    ),
                    PreNormForward(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                ])
            )

    def forward(self, x, save_hidden=False):
        if save_hidden == True:
            hidden_list = []
            hidden_list.append(x)
            for attn, ff in self.layers:
                x = attn(x, x, x) + x
                x = ff(x) + x
                hidden_list.append(x)
            return hidden_list
        else:
            for attn, ff in self.layers:
                x = attn(x, x, x) + x
                x = ff(x) + x
            return x


class CrossTransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNormAttention(
                        dim,
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                    ),
                    PreNormForward(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                ])
            )

    def forward(self, source_x, target_x):
        for attn, ff in self.layers:
            target_x_tmp = attn(target_x, source_x, source_x)
            target_x = target_x_tmp + target_x
            target_x = ff(target_x) + target_x
        return target_x


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        num_frames,
        token_len,
        save_hidden,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()

        self.token_len = token_len
        self.save_hidden = save_hidden

        if token_len is not None:
            self.pos_embedding = nn.Parameter(
                torch.randn(1, num_frames + token_len, dim)
            )
            self.extra_token = nn.Parameter(torch.zeros(1, token_len, dim))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
            self.extra_token = None

        self.dropout = nn.Dropout(emb_dropout)

        self.encoder = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

    def forward(self, x):
        b, n, _ = x.shape

        if self.token_len is not None:
            extra_token = repeat(self.extra_token, "1 n d -> b n d", b=b)
            x = torch.cat((extra_token, x), dim=1)
            x = x + self.pos_embedding[:, : n + self.token_len]
        else:
            x = x + self.pos_embedding[:, :n]

        x = self.dropout(x)
        x = self.encoder(x, self.save_hidden)

        return x


class CrossTransformer(nn.Module):
    def __init__(
        self,
        *,
        source_num_frames,
        tgt_num_frames,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()

        self.pos_embedding_s = nn.Parameter(torch.randn(1, source_num_frames + 1, dim))
        self.pos_embedding_t = nn.Parameter(torch.randn(1, tgt_num_frames + 1, dim))
        self.extra_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.CrossTransformerEncoder = CrossTransformerEncoder(
            dim, depth, heads, dim_head, mlp_dim, dropout
        )

        self.pool = pool

    def forward(self, source_x, target_x):
        b, n_s, _ = source_x.shape
        b, n_t, _ = target_x.shape

        extra_token = repeat(self.extra_token, "1 1 d -> b 1 d", b=b)

        source_x = torch.cat((extra_token, source_x), dim=1)
        source_x = source_x + self.pos_embedding_s[:, : n_s + 1]

        target_x = torch.cat((extra_token, target_x), dim=1)
        target_x = target_x + self.pos_embedding_t[:, : n_t + 1]

        source_x = self.dropout(source_x)
        target_x = self.dropout(target_x)

        x_s2t = self.CrossTransformerEncoder(source_x, target_x)

        return x_s2t


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        """
        初始化一个Squeeze-and-Excitation风格的通道注意力模块。
        :param in_planes: 输入特征的通道数 (即特征维度)。
        :param ratio: 通道降维的比例。
        """
        super(ChannelAttention, self).__init__()
        # Squeeze-and-Excitation 模块
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // ratio, in_planes, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        前向传播。
        :param x: 输入张量，形状为 (B, L, C)，其中 B=批次大小, L=序列长度, C=通道数。
        :return: 经过通道注意力重校准后的张量，形状与输入相同。
        """
        # x shape: (B, L, C)
        # 为了进行1D池化，需要将形状变为 (B, C, L)
        x_permuted = x.permute(0, 2, 1)
        b, c, _ = x_permuted.size()

        # Squeeze (压缩): 在序列维度(L)上进行全局平均池化
        y = self.avg_pool(x_permuted).view(b, c)

        # Excitation (激励): 通过全连接层学习通道权重
        y = self.fc(y).view(b, c, 1)

        # Scale (重校准): 将权重应用到原始特征上
        scaled_x = x_permuted * y.expand_as(x_permuted)

        # 将形状恢复为 (B, L, C)
        return scaled_x.permute(0, 2, 1)


# #======================================================================
# # ALMT 主模型 (来自 models/almt.py)
# #======================================================================
# class ALMT(nn.Module):
#     def __init__(self, args):
#         super(ALMT, self).__init__()

#         args = args.model

#         self.h_hyper = nn.Parameter(torch.ones(1, args.token_len, args.token_dim))

#         self.bertmodel = BertTextEncoder(use_finetune=True, transformers='bert', pretrained=args.bert_pretrained)

#         self.proj_l = nn.Sequential(
#             nn.Linear(args.l_input_dim, args.l_proj_dst_dim),
#             Transformer(num_frames=args.l_input_length, save_hidden=False, token_len=args.token_length, dim=args.proj_input_dim, depth=args.proj_depth, heads=args.proj_heads, mlp_dim=args.proj_mlp_dim)
#         )
#         self.proj_a = nn.Sequential(
#             nn.Linear(args.a_input_dim, args.a_proj_dst_dim),
#             Transformer(num_frames=args.a_input_length, save_hidden=False, token_len=args.token_length, dim=args.proj_input_dim, depth=args.proj_depth, heads=args.proj_heads, mlp_dim=args.proj_mlp_dim)
#         )
#         self.proj_v = nn.Sequential(
#             nn.Linear(args.v_input_dim, args.v_proj_dst_dim),
#             Transformer(num_frames=args.v_input_length, save_hidden=False, token_len=args.token_length, dim=args.proj_input_dim, depth=args.proj_depth, heads=args.proj_heads, mlp_dim=args.proj_mlp_dim)
#         )

#         self.l_encoder = Transformer(num_frames=args.token_length, save_hidden=True, token_len=None, dim=args.proj_input_dim, depth=args.AHL_depth-1, heads=args.l_enc_heads, mlp_dim=args.l_enc_mlp_dim)
#         self.h_hyper_layer = HhyperLearningEncoder(dim=args.token_dim, depth=args.AHL_depth, heads=args.ahl_heads, dim_head=args.ahl_dim_head, dropout=args.ahl_droup)
#         self.fusion_layer = CrossTransformer(source_num_frames=args.token_len, tgt_num_frames=args.token_len, dim=args.proj_input_dim, depth=args.fusion_layer_depth, heads=args.fusion_heads, mlp_dim=args.fusion_mlp_dim)

#         self.regression_layer = nn.Sequential(
#             nn.Linear(args.token_dim, 1)
#         )

#     def forward(self, x_visual, x_audio, x_text):
#         b = x_visual.size(0)

#         h_hyper = repeat(self.h_hyper, '1 n d -> b n d', b=b)

#         x_text = self.bertmodel(x_text)

#         h_v = self.proj_v(x_visual)[:, :self.h_hyper.shape[1]]
#         h_a = self.proj_a(x_audio)[:, :self.h_hyper.shape[1]]
#         h_l = self.proj_l(x_text)[:, :self.h_hyper.shape[1]]

#         h_t_list = self.l_encoder(h_l)
#         h_hyper = self.h_hyper_layer(h_t_list, h_a, h_v, h_hyper)
#         feat = self.fusion_layer(h_hyper, h_t_list[-1])[:, 0]

#         output = self.regression_layer(feat)

#         return output


# ----------------------------------------------------
# 模块1: 跨模态注意力模块 (Cross-Attention Block)
# (此模块与上一版相同，用于执行核心的注意力计算)
# ----------------------------------------------------
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        self.attn = PreNormAttention(
            dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        )
        self.ff = PreNormForward(dim, FeedForward(dim, dim * 2, dropout=dropout))

    def forward(self, query, context):
        query = self.attn(query, context, context) + query
        query = self.ff(query) + query
        return query


# ----------------------------------------------------
# 模块2: 全新的门控引导层 (New Gated Guidance Layer)
# 这是本次升级的核心！它引入了残差连接和门控机制。
# ----------------------------------------------------
class GatedGuidanceLayer(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.0):
        super().__init__()
        # 为音频和视觉分别创建一个跨注意力模块，用于生成“候选更新”
        self.audio_guidance_candidate = CrossAttentionBlock(
            dim, heads, dim_head, dropout
        )
        self.visual_guidance_candidate = CrossAttentionBlock(
            dim, heads, dim_head, dropout
        )

        # 门控机制：为音频和视觉分别创建一个“门”
        # 这个门是一个线性层+Sigmoid激活函数，它的输出是一个0到1之间的值，决定了更新的程度
        # 门的输入是上一层的特征和候选更新特征的拼接，让网络根据上下文决定如何更新
        self.audio_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.visual_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

    def forward(self, h_t, h_a_prev, h_v_prev):
        """
        前向传播.
        - h_t: 当前尺度的文本特征 (上下文)
        - h_a_prev: 上一层传入的音频特征
        - h_v_prev: 上一层传入的视觉特征
        """
        # 1. 生成候选更新 (Candidate Update)
        # 这部分与之前类似，用文本去引导音/视频
        h_a_candidate = self.audio_guidance_candidate(query=h_a_prev, context=h_t)
        h_v_candidate = self.visual_guidance_candidate(query=h_v_prev, context=h_t)

        # 2. 计算门控信号 (Gate Signal)
        # 将原始特征和候选特征拼接，送入门控网络
        gate_a = self.audio_gate(torch.cat((h_a_prev, h_a_candidate), dim=-1))
        gate_v = self.visual_gate(torch.cat((h_v_prev, h_v_candidate), dim=-1))

        # 3. 执行门控残差更新 (Gated Residual Update)
        # 公式: h_new = (1 - gate) * h_old + gate * h_candidate
        # 这允许模型学习保留多少原始信息(h_old)，并融入多少新信息(h_candidate)
        h_a_new = (1 - gate_a) * h_a_prev + gate_a * h_a_candidate
        h_v_new = (1 - gate_v) * h_v_prev + gate_v * h_v_candidate

        return h_a_new, h_v_new


# ----------------------------------------------------
# 模块3: 分层引导编码器 (Hierarchical Guidance Encoder)
# (该模块现在内部使用新的 GatedGuidanceLayer)
# ----------------------------------------------------
class HierarchicalGuidanceEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout=0.0):
        super().__init__()
        # 创建一个包含多个“门控引导层”的列表
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            # *** 此处使用了新的 GatedGuidanceLayer ***
            self.layers.append(GatedGuidanceLayer(dim, heads, dim_head, dropout))

        # 最后的融合层保持不变
        self.fusion_projection = nn.Sequential(
            nn.Linear(dim * 2, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )

    def forward(self, h_t_list, h_a, h_v):
        """
        前向传播逻辑不变，只是内部调用的层变了.
        """
        for i, layer in enumerate(self.layers):
            h_a, h_v = layer(h_t_list[i], h_a, h_v)

        combined_features = torch.cat((h_a, h_v), dim=-1)
        h_hyper = self.fusion_projection(combined_features)

        return h_hyper


# ----------------------------------------------------
# 模块 B: 来自版本 (5) 的高级融合组件
# (_MultimodalCrossLayer, Multimodal_SelfAttention)
# ----------------------------------------------------
import math


class _MultimodalCrossLayer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.0):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"hidden size ({dim}) not multiple of heads ({heads})")
        self.hidden_size = dim
        self.num_head = heads
        self.attention_head_size = dim // heads
        self.all_head_size = self.num_head * self.attention_head_size
        self.q = nn.Linear(self.hidden_size, self.all_head_size)
        self.k = nn.Linear(self.hidden_size, self.all_head_size)
        self.v = nn.Linear(self.hidden_size, self.all_head_size)
        self.dropout_1 = nn.Dropout(dropout)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.LayerNorm1 = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout_2 = nn.Dropout(dropout)
        self.w_source_param = nn.Parameter(torch.tensor(0.5))
        self.w_target_param = nn.Parameter(torch.tensor(0.5))
        self.gate_source_target_linear = nn.Linear(
            self.hidden_size * 2, self.hidden_size
        )
        self.gate_target_source_linear = nn.Linear(
            self.hidden_size * 2, self.hidden_size
        )
        self.ff = FeedForward(dim, mlp_dim, dropout)
        self.LayerNorm2 = nn.LayerNorm(self.hidden_size, eps=1e-12)

    def transpose1(self, x):
        new_shape = x.size()[:-1] + (self.num_head, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def cross_attention(self, q_modal, k_modal, v_modal):
        q_proj, k_proj, v_proj = (
            self.transpose1(self.q(q_modal)),
            self.transpose1(self.k(k_modal)),
            self.transpose1(self.v(v_modal)),
        )
        attn_score = torch.matmul(q_proj, k_proj.transpose(-1, -2)) / math.sqrt(
            self.attention_head_size
        )
        attn_prob = nn.Softmax(dim=-1)(attn_score)
        attn_prob = self.dropout_1(attn_prob)
        context = torch.matmul(attn_prob, v_proj).permute(0, 2, 1, 3).contiguous()
        return context.view(context.size(0), context.size(1), -1)

    def forward(self, source_x, target_x):
        s_len, t_len = source_x.shape[1], target_x.shape[1]
        s_t_cross, t_s_cross = (
            self.cross_attention(source_x, target_x, target_x),
            self.cross_attention(target_x, source_x, source_x),
        )
        gate_s, gate_t = (
            torch.sigmoid(
                self.gate_source_target_linear(torch.cat((source_x, s_t_cross), dim=-1))
            ),
            torch.sigmoid(
                self.gate_target_source_linear(torch.cat((target_x, t_s_cross), dim=-1))
            ),
        )
        h_s_fused, h_t_fused = (
            source_x + gate_s * s_t_cross,
            target_x + gate_t * t_s_cross,
        )
        fused_hidden = torch.cat([h_s_fused, h_t_fused], dim=1)
        q_layer, k_layer, v_layer = (
            self.transpose1(self.q(fused_hidden)),
            self.transpose1(self.k(fused_hidden)),
            self.transpose1(self.v(fused_hidden)),
        )
        attention_score = torch.matmul(q_layer, k_layer.transpose(-1, -2)) / math.sqrt(
            self.attention_head_size
        )
        attention_prob_s, attention_prob_t = (
            nn.Softmax(dim=-1)(attention_score[:, :, :, :s_len]),
            nn.Softmax(dim=-1)(attention_score[:, :, :, s_len:]),
        )
        attention_prob = torch.cat((attention_prob_s, attention_prob_t), dim=-1)
        m_s = self.w_source_param * torch.ones(
            1, 1, s_len + t_len, s_len, device=attention_score.device
        )
        m_t = self.w_target_param * torch.ones(
            1, 1, s_len + t_len, t_len, device=attention_score.device
        )
        modality_mask = torch.cat((m_s, m_t), dim=3)
        attention_prob = self.dropout_1(attention_prob.mul(modality_mask))
        context_layer = (
            torch.matmul(attention_prob, v_layer).permute(0, 2, 1, 3).contiguous()
        )
        context_layer = self.dense(
            context_layer.view(context_layer.size(0), context_layer.size(1), -1)
        )
        hidden_state = self.LayerNorm1(self.dropout_2(context_layer) + fused_hidden)
        hidden_state = self.LayerNorm2(self.ff(hidden_state) + hidden_state)
        return hidden_state[:, :s_len, :], hidden_state[:, s_len:, :]


class Multimodal_SelfAttention(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            _MultimodalCrossLayer(
                dim=dim, heads=heads, mlp_dim=mlp_dim, dropout=dropout
            )
            for _ in range(depth)
        ])

    def forward(self, source_x, target_x):
        for layer in self.layers:
            source_x, target_x = layer(source_x, target_x)
        return target_x


import torch.nn.functional as F


class ProjectorAttention(nn.Module):
    """
    一个基于注意力的池化投影模块。
    它接收BERT的完整输出和对应的注意力掩码。
    1. 计算每个词元的重要性分数（注意力权重）。
    2. 基于权重计算所有词元嵌入的加权和，生成一个上下文向量。
    3. 将这个上下文向量扩展为后续层期望的序列形状。
    """

    def __init__(self, input_dim, token_dim, token_len):
        super().__init__()
        self.token_len = token_len
        self.token_dim = token_dim

        # 头部，用于将BERT输出投影到工作维度
        self.projection_head = nn.Linear(input_dim, token_dim)

        # 注意力网络，用于计算每个词元的重要性分数
        self.attention_net = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.Tanh(),
            nn.Linear(token_dim // 2, 1),
        )

        # 扩展网络，用于将最终的上下文向量映射为序列输出
        self.expansion_mlp = nn.Linear(token_dim, token_len * token_dim)

    def forward(self, embeddings, attention_mask):
        # embeddings 的形状: (B, S, D_bert), e.g., (64, 50, 768)
        # attention_mask 的形状: (B, S, 1), e.g., (64, 50, 1)

        # 1. 将高维嵌入投影到工作维度
        projected_embeddings = self.projection_head(embeddings)
        # projected_embeddings 的形状: (B, S, D_token), e.g., (64, 50, 128)

        # 2. 计算原始的注意力分数
        attention_scores = self.attention_net(projected_embeddings)
        # attention_scores 的形状: (B, S, 1)

        # 3. *** 关键步骤 ***: 应用注意力掩码
        # 将所有填充位置（mask值为0）的分数设为一个极小的负数
        # 这样在Softmax之后，它们的权重将趋近于0
        attention_scores.masked_fill_(attention_mask == 0, -1e9)

        # 4. 通过Softmax将分数转换为权重
        attention_weights = F.softmax(attention_scores, dim=1)
        # attention_weights 的形状: (B, S, 1)

        # 5. 计算加权和，得到上下文向量
        # (B, S, D_token) * (B, S, 1) -> 广播乘法
        # torch.sum(..., dim=1) -> 在序列维度上求和
        context_vector = torch.sum(projected_embeddings * attention_weights, dim=1)
        # context_vector 的形状: (B, D_token), e.g., (64, 128)

        # 6. 将上下文向量通过MLP扩展并重塑
        expanded_output = self.expansion_mlp(context_vector)
        output_seq = expanded_output.view(-1, self.token_len, self.token_dim)
        # output_seq 的形状: (B, token_len, D_token), e.g., (64, 8, 128)

        return output_seq


# ----------------------------------------------------
# 模块 C: 最终的融合主模型 CLFE_Fused
# ----------------------------------------------------
class CLFE_Fused(nn.Module):
    def __init__(self, args):
        super(CLFE_Fused, self).__init__()
        config_model = args.model
        self.token_len = config_model.token_len

        # --- 基础特征提取器 ---
        self.bertmodel = BertTextEncoder(
            use_finetune=True, pretrained=config_model.bert_pretrained
        )
        self.proj_l = ProjectorAttention(
            input_dim=config_model.l_input_dim,
            token_dim=config_model.token_dim,
            token_len=config_model.token_len,
        )
        # 新增! 使用注意力池化的音频投影器
        self.proj_a = ProjectorAttention(
            input_dim=config_model.a_input_dim,
            token_dim=config_model.token_dim,
            token_len=config_model.token_len,
        )

        # 新增! 使用注意力池化的视觉投影器
        self.proj_v = ProjectorAttention(
            input_dim=config_model.v_input_dim,
            token_dim=config_model.token_dim,
            token_len=config_model.token_len,
        )

        # === 新增：为每个模态定义一个通道注意力模块 ===
        self.chan_attn_l = ChannelAttention(config_model.token_dim, ratio=8)
        self.chan_attn_a = ChannelAttention(config_model.token_dim, ratio=8)
        self.chan_attn_v = ChannelAttention(config_model.token_dim, ratio=8)

        # --- 步骤1: 多层次文本特征和引导后的音视特征 ---
        self.l_encoder = Transformer(
            num_frames=config_model.token_length,
            save_hidden=True,
            token_len=None,
            dim=config_model.proj_input_dim,
            depth=config_model.AHL_depth,
            heads=config_model.l_enc_heads,
            mlp_dim=config_model.l_enc_mlp_dim,
        )
        self.guidance_encoder = HierarchicalGuidanceEncoder(
            dim=config_model.token_dim,
            depth=config_model.AHL_depth,
            heads=config_model.ahl_heads,
            dim_head=config_model.ahl_dim_head,
            dropout=config_model.ahl_droup,
        )

        # --- 步骤2: 最终融合 ---
        self.final_fusion_layer = Multimodal_SelfAttention(
            dim=config_model.proj_input_dim,
            depth=config_model.fusion_layer_depth,
            heads=config_model.fusion_heads,
            dim_head=config_model.ahl_dim_head,
            mlp_dim=config_model.fusion_mlp_dim,
            dropout=config_model.ahl_droup,
        )

        # --- 步骤3: 最终回归层 ---
        self.regression_layer = nn.Sequential(nn.Linear(config_model.token_dim, 1))

    def forward(self, x_visual, x_audio, x_text):
        # 1. 提取注意力掩码
        text_attention_mask = x_text[:, 1, :].unsqueeze(2)

        # 2. 获取文本嵌入
        x_text_embeddings = self.bertmodel(x_text)

        # 1. 为音频和视觉数据创建虚拟掩码
        #    形状应为 (批次大小, 序列长度, 1)
        audio_mask = torch.ones(
            x_audio.shape[0], x_audio.shape[1], 1, device=x_audio.device
        )
        vision_mask = torch.ones(
            x_visual.shape[0], x_visual.shape[1], 1, device=x_visual.device
        )
        bert_attention_mask = x_text[:, 1, :].unsqueeze(2)

        # 2. 用它们各自的掩码来调用投影器
        h_v_proj = self.proj_v(x_visual, vision_mask)
        h_l_proj = self.proj_l(x_text_embeddings, bert_attention_mask)
        h_a_proj = self.proj_a(x_audio, audio_mask)

        # === 修改：在投影后应用通道注意力 ===
        h_l_initial = self.chan_attn_l(h_l_proj)
        h_a_initial = self.chan_attn_a(h_a_proj)
        h_v_initial = self.chan_attn_v(h_v_proj)

        # --- 4. 分层引导 ---
        h_t_list = self.l_encoder(h_l_initial)
        h_hyper_guided = self.guidance_encoder(h_t_list, h_a_initial, h_v_initial)

        # --- 5. 最终融合 ---
        fused_output = self.final_fusion_layer(
            source_x=h_hyper_guided, target_x=h_t_list[-1]
        )

        # --- 6. 回归输出 ---
        feat = fused_output[:, 0]
        output = self.regression_layer(feat)

        return output


# 最后，别忘了修改build_model函数来使用这个新模型
def build_model(args):
    model = CLFE_Fused(args)  # <-- 使用我们全新的融合模型
    return model


print("所有模型和工具函数已定义。")
