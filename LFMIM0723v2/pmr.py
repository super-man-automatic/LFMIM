# PMR_final.py

import os
import sys
import time
import math
import collections
from collections import OrderedDict
import argparse
import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
import torch.utils.data.distributed
from sklearn.metrics import confusion_matrix, classification_report

# Assuming these are in your project directory
from data_prepare import MMSAATBaselineDataset
from modules.position_embedding import SinusoidalPositionalEmbedding

# --- Logger Class for simultaneous console and file logging ---
class Logger(object):
    def __init__(self, filename="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

# --- Configuration ---
max_len = 50
labels_eng =  ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
output_dir = 'output_pmr'
saved_models_dir = 'saved_models_pmr'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(saved_models_dir, exist_ok=True)

def pad_collate(batch):
    (x_t, x_a, x_v, y_t, y_a, y_v, y_m) = zip(*batch)
    x_t = torch.stack(x_t, dim=0)
    x_v = torch.stack(x_v, dim=0)
    y_t = torch.tensor(y_t)
    y_a = torch.tensor(y_a)
    y_v = torch.tensor(y_v)
    y_m = torch.tensor(y_m)
    x_a_pad = pad_sequence(x_a, batch_first=True, padding_value=0)
    len_trunc = min(x_a_pad.shape[1], max_len)
    x_a_pad = x_a_pad[:, 0:len_trunc, :]
    len_com = max_len - len_trunc
    zeros = torch.zeros([x_a_pad.shape[0], len_com, x_a_pad.shape[2]], device='cpu')
    x_a_pad = torch.cat([x_a_pad, zeros], dim=1)
    return x_t, x_a_pad, x_v, y_t, y_a, y_v, y_m

#######################################################
# MODEL DEFINITIONS (LayerNorm, QuickGELU, ResidualAttentionBlock, Transformer)
# ... (These class definitions are identical to the original file, so they are omitted here for brevity)
# Please copy the class definitions for LayerNorm, QuickGELU, ResidualAttentionBlock, and Transformer from your original file here.
class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=0.2, batch_first=True)
        self.ln_1 = LayerNorm(d_model)
        self.ln_12 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),("relu", nn.ReLU()),
            ('dropout', nn.Dropout(p=0.1)),("c_proj", nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.ln_22 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.dropout = nn.Dropout(p=0.1)
    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    def forward(self, x: torch.Tensor):
        x = x + self.ln_12(self.attention(self.ln_1(x)))
        x = x + self.ln_22(self.mlp(self.ln_2(x)))
        return x
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
#######################################################

class MultiUniTransformer(nn.Module):
    # __init__ is identical to the original file.
    # Please copy the __init__ and initialize_parameters methods from your original file here.
    def __init__(self, width: list, layers: int, heads: int, embed_dim: list, args, attn_mask: torch.Tensor = None):
        super().__init__()
        self.layers = layers
        self.width_t, self.width_a, self.width_v, self.width_m = width[0], width[1], width[2], width[3]
        self.embed_dim_t, self.embed_dim_a, self.embed_dim_v, self.embed_dim_m = embed_dim[0], embed_dim[1], embed_dim[2], embed_dim[3]
        self.width_t_ori, self.width_a_ori, self.width_v_ori = 1024, 1024, 2048
        self.dropout_ft, self.dropout_fa, self.dropout_fv, self.dropout_fm = 0.25, 0.25, 0.25, 0.25
        self.fea_len_t, self.fea_len_a, self.fea_len_v, self.fea_len_m = args.fea_len_t, args.fea_len_a, args.fea_len_v, args.fea_len_m
        num_classes = args.num_classes
        self.transformer_t = nn.Sequential(*[ResidualAttentionBlock(self.width_t, heads[0], attn_mask) for _ in range(layers)])
        self.transformer_a = nn.Sequential(*[ResidualAttentionBlock(self.width_a, heads[1], attn_mask) for _ in range(layers)])
        self.transformer_v = nn.Sequential(*[ResidualAttentionBlock(self.width_v, heads[2], attn_mask) for _ in range(2 * layers)])
        self.transformer_m = nn.Sequential(*[ResidualAttentionBlock(self.width_m, heads[3], attn_mask) for _ in range(layers)])
        self.t2m, self.v2m = nn.Sequential(*[nn.Linear(self.fea_len_t, 1) for _ in range(layers)]), nn.Sequential(*[nn.Linear(self.fea_len_v, 1) for _ in range(layers)])
        self.fc_vtom, self.fc_tdimtr, self.fc_adimtr, self.fc_vdimtr = nn.Linear(self.width_v, self.width_m), nn.Linear(self.width_t_ori, self.width_t), nn.Linear(self.width_a_ori, self.width_a), nn.Linear(self.width_v_ori, self.width_m)
        self.ln_pre_t, self.ln_pre_a, self.ln_pre_v, self.ln_pre_m = LayerNorm(self.width_t), LayerNorm(self.width_a), LayerNorm(self.width_v), LayerNorm(self.width_m)
        self.ln_post_t, self.ln_post_a, self.ln_post_v, self.ln_post_m = LayerNorm(self.width_t), LayerNorm(self.width_a), LayerNorm(self.width_v), LayerNorm(self.width_m)
        self.ln_prepe_t, self.ln_prepe_a, self.ln_prepe_v, self.ln_prepe_m = LayerNorm(self.width_t), LayerNorm(self.width_a), LayerNorm(self.width_v), LayerNorm(self.width_m)
        self.proj_a, self.proj_v = nn.Parameter(torch.empty(self.width_a, self.embed_dim_a)), nn.Parameter(torch.empty(self.width_v, self.embed_dim_v))
        self.mix_t, self.mix_a, self.mix_v = nn.Parameter((self.fea_len_t ** -0.5) * torch.rand(self.fea_len_t)), nn.Parameter((self.fea_len_a ** -0.5) * torch.rand(self.fea_len_a)), nn.Parameter((self.fea_len_v ** -0.5) * torch.rand(self.fea_len_v))
        self.mix_m = nn.Parameter(((self.fea_len_m + self.fea_len_t + self.fea_len_a + self.fea_len_v) ** -0.5) * torch.rand(self.fea_len_m + self.fea_len_t + self.fea_len_a + self.fea_len_v))
        self.mlp_t = nn.Sequential(nn.Linear(self.embed_dim_t, int(self.embed_dim_t/2)), nn.ReLU(), nn.Linear(int(self.embed_dim_t/2), 32), nn.ReLU(), nn.Linear(32, num_classes))
        self.mlp_a = nn.Sequential(nn.Linear(self.embed_dim_a, int(self.embed_dim_a/2)), nn.ReLU(), nn.Linear(int(self.embed_dim_a/2), 32), nn.ReLU(), nn.Linear(32, num_classes))
        self.mlp_v = nn.Sequential(nn.Linear(self.embed_dim_v, int(self.embed_dim_v/2)), nn.ReLU(), nn.Linear(int(self.embed_dim_v/2), 32), nn.ReLU(), nn.Linear(32, num_classes))
        self.mlp_m = nn.Sequential(nn.Linear(self.embed_dim_m + self.embed_dim_t +self.embed_dim_a + self.embed_dim_v, int((self.embed_dim_t + self.embed_dim_a + self.embed_dim_v + self.embed_dim_m)/2)), nn.ReLU(), nn.Linear(int((self.embed_dim_t + self.embed_dim_a + self.embed_dim_v + self.embed_dim_m)/2), 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes))
        self.plus_p, self.pe_t, self.pe_t2m, self.pe_a, self.pe_v, self.pe_v2m, self.pe_m = nn.Parameter(torch.randn(6)), nn.Parameter(self.width_t**-0.5*torch.randn(self.fea_len_t, self.width_t)), nn.Parameter(self.width_t**-0.5*torch.randn(self.fea_len_t, self.width_t)), nn.Parameter(self.width_a**-0.5*torch.randn(self.fea_len_a, self.width_a)), nn.Parameter(self.width_v**-0.5*torch.randn(self.fea_len_v, self.width_v)), nn.Parameter(self.width_v**-0.5*torch.randn(self.fea_len_v, self.width_m)), nn.Parameter(self.width_m**-0.5*torch.randn(self.fea_len_m, self.width_m))
        self.me = nn.Parameter(self.width_t**-0.5 * torch.randn(4))
        self.embed_scale, self.embed_positions = math.sqrt(self.width_m), SinusoidalPositionalEmbedding(self.width_m)
        self.embedding_t, self.embedding_a, self.embedding_v, self.embedding_m = nn.Parameter(self.width_t**-0.5*torch.randn(self.width_t)), nn.Parameter(self.width_a**-0.5*torch.randn(self.width_a)), nn.Parameter(self.width_v**-0.5*torch.randn(self.width_v)), nn.Parameter(self.width_m**-0.5*torch.randn(self.fea_len_m, self.width_m))
    def initialize_parameters(self):
        nn.init.normal_(self.proj_a, std=self.width_a ** -0.5)
        nn.init.normal_(self.proj_v, std=self.width_v ** -0.5)
        proj_std = (self.width_t ** -0.5) * ((2 * self.layers) ** -0.5)
        attn_std, fc_std = self.width_t ** -0.5, (2 * self.width_t) ** -0.5
        for branch in [self.transformer_t, self.transformer_a, self.transformer_m]:
            for block in branch:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std); nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std); nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        proj_std = (self.width_v ** -0.5) * ((2 * 2 * self.layers) ** -0.5)
        attn_std, fc_std = self.width_v ** -0.5, (2 * self.width_v) ** -0.5
        for branch in [self.transformer_v]:
            for block in branch:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std); nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std); nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def forward(self, x_t, x_a, x_v):
        # --- START: This initial part is the same as the original ---
        x_t = x_t[:, 0:80, :]
        x_v, x_t, x_a = x_v.to(torch.float32), x_t.to(torch.float32), x_a.to(torch.float32)
        x_t, x_a, x_v = self.fc_tdimtr(x_t), self.fc_adimtr(x_a), self.fc_vdimtr(x_v)
        x_m = self.embedding_m.to(x_t.dtype) + torch.zeros(x_t.shape[0], self.fea_len_m, x_t.shape[-1], dtype=x_t.dtype, device=x_t.device)
        x_m = x_m.to(x_t.device)
        x_t = torch.cat([self.embedding_t.to(x_t.dtype) + torch.zeros(x_t.shape[0], 1, x_t.shape[-1], dtype=x_t.dtype, device=x_t.device), x_t], dim=1)
        x_a = torch.cat([self.embedding_a.to(x_a.dtype) + torch.zeros(x_a.shape[0], 1, x_a.shape[-1], dtype=x_a.dtype, device=x_a.device), x_a], dim=1)
        x_v = torch.cat([self.embedding_v.to(x_v.dtype) + torch.zeros(x_v.shape[0], 1, x_v.shape[-1], dtype=x_v.dtype, device=x_v.device), x_v], dim=1)
        x_t, x_a, x_v, x_m = self.embed_scale*x_t, self.embed_scale*x_a, self.embed_scale*x_v, self.embed_scale*x_m
        x_t, x_a, x_v, x_m = self.ln_prepe_t(x_t), self.ln_prepe_a(x_a), self.ln_prepe_v(x_v), self.ln_prepe_m(x_m)
        x_t, x_a, x_v, x_m = x_t+self.embed_positions(x_t[:,:,0]), x_a+self.embed_positions(x_a[:,:,0]), x_v+self.embed_positions(x_v[:,:,0]), x_m+self.embed_positions(x_m[:,:,0])
        x_t, x_a, x_v, x_m = F.dropout(x_t, p=self.dropout_ft, training=self.training), F.dropout(x_a, p=self.dropout_fa, training=self.training), F.dropout(x_v, p=self.dropout_fv, training=self.training), F.dropout(x_m, p=self.dropout_fm, training=self.training)
        x_t, x_a, x_v, x_m = self.ln_pre_t(x_t), self.ln_pre_a(x_a), self.ln_pre_v(x_v), self.ln_pre_m(x_m)
        # --- END: This initial part is the same as the original ---

        for i in range(0, self.layers):
            # --- START OF MODIFICATION for PMR ---
            # 1. Prepare multi-modal input (logic from original code)
            if i == 0: 
                x_m_input = torch.cat([x_m[:, 0:self.fea_len_m, :], x_t, x_v, x_a], dim=1)
            else:
                x_tacc = x_m[:, self.fea_len_m: self.fea_len_m + self.fea_len_t, :] * self.plus_p[0] + x_t * self.plus_p[1]
                x_vacc = x_m[:, self.fea_len_m + self.fea_len_t : self.fea_len_m + self.fea_len_t + self.fea_len_v, :] * self.plus_p[2] + x_v * self.plus_p[3]
                x_aacc = x_m[:, self.fea_len_m + self.fea_len_t + self.fea_len_v :, :] * self.plus_p[4] + x_a * self.plus_p[5]
                x_m_input = torch.cat([x_m[:, 0:self.fea_len_m, :], x_tacc, x_vacc, x_aacc], dim=1)

            # 2. First, compute the multi-modal fusion to get the updated x_m
            x_m = self.transformer_m[i](x_m_input)
            
            # 3. Extract feedback and reinforce uni-modal inputs
            feedback_t = x_m[:, self.fea_len_m : self.fea_len_m + self.fea_len_t, :]
            feedback_v = x_m[:, self.fea_len_m + self.fea_len_t : self.fea_len_m + self.fea_len_t + self.fea_len_v, :]
            feedback_a = x_m[:, self.fea_len_m + self.fea_len_t + self.fea_len_v :, :]
            
            reinforced_t = x_t + feedback_t
            reinforced_a = x_a + feedback_a
            reinforced_v = x_v + feedback_v

            # 4. Update uni-modal modules using the reinforced inputs
            x_t = self.transformer_t[i](reinforced_t)
            x_a = self.transformer_a[i](reinforced_a)
            x_v = self.transformer_v[2 * i](reinforced_v)
            x_v = self.transformer_v[2 * i + 1](x_v)
            # --- END OF MODIFICATION for PMR ---

        # --- The rest of the forward pass is the same as the original ---
        x_t, x_a, x_v, x_m = self.ln_post_t(x_t), self.ln_post_a(x_a), self.ln_post_v(x_v), self.ln_post_m(x_m)
        x_t, x_a, x_v, x_m = torch.matmul(self.mix_t, x_t), torch.matmul(self.mix_a, x_a), torch.matmul(self.mix_v, x_v), torch.matmul(self.mix_m, x_m)
        x_all = torch.cat([x_t, x_a, x_v, x_m], dim=1)
        x_t, x_a, x_v, x_m = self.mlp_t(x_t), self.mlp_a(x_a), self.mlp_v(x_v), self.mlp_m(x_all)
        return [x_t, x_a, x_v, x_m]

class Trainer():
    # The Trainer class is identical to the original file.
    # Please copy the __init__, train, and test methods from your original file here.
    # Note: I've updated it to use the new directory names.
    def __init__(self, args):
        self.args, self.epoch, self.batch_size, self.log_interval = args, args.epoch, args.batch_size, args.log_interval
        self.local_rank, self.num_classes, self.beta = args.local_rank, args.num_classes, args.beta
        self.model = MultiUniTransformer([1024]*4, 4, [16]*4, [1024]*4, args)
        self.model = self.model.to(args.device)
        self.model.initialize_parameters()
        if args.is_ddp:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.005)
        self.scheduler_1r = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
        train_data = MMSAATBaselineDataset('train')
        traindata_sampler = torch.utils.data.distributed.DistributedSampler(train_data) if args.is_ddp else None
        self.train_dataloader = DataLoader(train_data, sampler=traindata_sampler, batch_size=self.batch_size, num_workers=6, collate_fn=pad_collate, shuffle=(traindata_sampler is None))
        test_data = MMSAATBaselineDataset('test')
        self.test_dataloader = DataLoader(test_data, batch_size=self.batch_size, num_workers=6, collate_fn=pad_collate)
    
    def train(self):
        self.model.train()
        for epoch in range(0, self.epoch):
            if self.args.is_ddp: self.train_dataloader.sampler.set_epoch(epoch)
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                text, audio, video, label_t, label_a, label_v, label_m = [d.to(self.args.device) for d in batch]
                output = self.model(text, audio, video)
                loss_t, loss_a, loss_v, loss_m = F.cross_entropy(output[0], label_t), F.cross_entropy(output[1], label_a), F.cross_entropy(output[2], label_v), F.cross_entropy(output[3], label_m)
                loss = loss_t + loss_a + loss_v + loss_m
                loss.backward()
                self.optimizer.step()
                if self.local_rank == 0 and batch_idx % self.log_interval == 0:
                    print(f'Train Epoch: {epoch} [{batch_idx * self.batch_size * self.args.world_size}/{len(self.train_dataloader.dataset)} ({100. * batch_idx / len(self.train_dataloader):.0f}%)]')
                    print(f'Train set: loss_t: {loss_t.item():.4f}, loss_a: {loss_a.item():.4f}, loss_v: {loss_v.item():.4f}, loss_m: {loss_m.item():.4f}, loss: {loss.item():.4f}\n')
            self.scheduler_1r.step()
            if self.local_rank == 0:
                print(f"Epoch {epoch} learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
                self.test(self.test_dataloader)
                model_to_save = self.model.module if self.args.is_ddp else self.model
                torch.save(model_to_save.state_dict(), f'{saved_models_dir}/PMR_{str(epoch)}.pth')
                self.model.train()

    def test(self, dataloader):
        self.model.eval()
        all_preds_m, all_labels_m = [], []
        with torch.no_grad():
            for batch in dataloader:
                text, audio, video, _, _, _, label_m = [d.to(self.args.device) for d in batch]
                output = self.model(text, audio, video)
                all_preds_m.append(output[3].cpu())
                all_labels_m.append(label_m.cpu())
        if self.local_rank == 0:
            preds_m, labels_m = torch.cat(all_preds_m).argmax(dim=1), torch.cat(all_labels_m)
            print("\n--- Test Results ---")
            print("Classification Report (Multimodal):")
            print(classification_report(labels_m.numpy(), preds_m.numpy(), target_names=labels_eng, digits=4))
            print("Confusion Matrix (Multimodal):")
            print(confusion_matrix(labels_m.numpy(), preds_m.numpy()))
            print("--- End Test ---\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PMR Model Training')
    # ... (Argument parsing is identical to the original file, so omitted for brevity)
    # Please copy the argparse setup from your original file here.
    parser.add_argument('--init_method', default='tcp://127.0.0.1:23456')
    parser.add_argument('--world_size', type=int, default=1, metavar='N')
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--max_len', default=50, type=int)
    parser.add_argument('--num_classes', default=7, type=int)
    parser.add_argument('--fea_len_t', default=81, type=int)
    parser.add_argument('--fea_len_a', default=51, type=int)
    parser.add_argument('--fea_len_v', default=17, type=int)
    parser.add_argument('--fea_len_m', default=4, type=int)
    parser.add_argument('--beta', default=0, type=float)
    parser.add_argument('--epoch', default=20, type=int)
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--log_interval', default=50, type=int)
    args = parser.parse_args()

    # --- DDP and Logging Setup ---
    if 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
    args.is_ddp = args.world_size > 1
    
    if args.is_ddp:
        dist.init_process_group(backend='nccl', init_method=args.init_method)
        args.rank = dist.get_rank()
        torch.cuda.set_device(args.rank)
    
    local_rank = args.rank
    args.local_rank = local_rank
    args.device = torch.device('cuda', local_rank)

    if local_rank == 0:
        log_file_path = os.path.join(output_dir, 'log_pmr.txt')
        sys.stdout = Logger(log_file_path, sys.stdout)
        print(f"Logging output to {log_file_path}")

    tic = time.time()
    if local_rank == 0:
        print("Starting training for PMR model...")
        print("Arguments:", args)

    trainer = Trainer(args)
    trainer.train()

    if local_rank == 0:
        print('Training finished.')
        toc = time.time()
        print(f'Total running time: {toc - tic:.2f} seconds')
        sys.stdout.close()