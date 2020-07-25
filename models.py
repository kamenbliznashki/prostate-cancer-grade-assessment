import json
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

from efficientnet_pytorch import EfficientNet

# --------------------
# Model on tiles
# --------------------

def make_model(name, args):
    if name == 'debug':
        model = nn.Sequential(nn.Conv2d(3, 1, kernel_size=3, stride=128), nn.Flatten(), nn.Linear(4, args.output_dim))
    elif name == 'debug_bags':
        class SelectLastGRUOutput(nn.Module):
            def __init__(self): super().__init__()
            def forward(self, x): return x[0][-1]
        model = nn.Sequential(nn.GRU(4, 3), SelectLastGRUOutput(), nn.Linear(3, args.output_dim))

    # tiles models
    elif name == 'resnet34':
        model = resnet34(pretrained=args.pretrained)
        model.fc = nn.Linear(model.fc.in_features, args.output_dim)
    elif 'efficientnet' in name:
        model = EfficientNet.from_pretrained(args.model) if args.pretrained else EfficientNet.from_name(args.model)
        model._fc = nn.Linear(model._fc.in_features, args.output_dim)

    # bags models
    elif name == 'mil-mean':
        model = MilMean(args.embedding_dim, args.output_dim)
    elif name == 'rnn':
        model = RNN(args.embedding_dim, args.embedding_dim, args.output_dim)
    elif name == 'tanh-attn-small':
        model = TanhAttnSmall(args.embedding_dim, args.embedding_dim // 2, args.output_dim)  # TODO -- change hidden dim to arg attn_dv
    elif name == 'tanh-attn-big':
        model = TanhAttnBig(args.embedding_dim, args.embedding_dim // 2, args.output_dim)  # TODO -- change hidden dim to arg attn_dv
    elif name == 'mha-small':
        model = MhaSmall(args.embedding_dim, args.output_dim)
    else:
        raise RuntimeError('Model not recognized: ', name)

    # hook feature rep before classification layer -- model now outputs (embedding, logits)
    fc = list(model.modules())[-1]
    fc.register_forward_hook(lambda module, in_tensor, out_tensor: (in_tensor[0], out_tensor))  # in_tensor is tuple so take first element

    if args.bias_model:
        fc = list(model.modules())[-1]
        fc.bias.data[-1] += 3.  # push model to predict highest class -- rationale: MIL training based on ceiling of highest prob for highest class, so start training by predicting all highest class and push it down during training

    return model

# --------------------
# Encoder on tiles
# --------------------

class Encoder(nn.Module):
    def __init__(self, model, freeze=True):
        super().__init__()
        self.model = model
        # freeze weights and put in eval mode
        self.model.eval()
        if freeze:
            for p in self.model.parameters(): p.requires_grad_(False)
        # embedding dim
        fc = list(self.model.modules())[-1]
        self.embedding_dim = fc.in_features

    def forward(self, x):
        if x.ndim == 5:
            # seq dim is 1
            embeddings, logits = [], []
            for i in range(x.shape[1]):
                e, l = self.model(x[:,i])
                embeddings.append(e)
                logits.append(l)
            embeddings, logits = torch.stack(embeddings), torch.stack(logits)
        else:
            embeddings, logits = self.model(x)

        return embeddings, logits

    @classmethod
    def from_config(cls, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        args = namedtuple('args', config.keys())
        args = args(**config)
        m = make_model(args.model, args)
        return cls(m, freeze=True)

# --------------------
# Models on bags
# --------------------

class Gate1d(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2,1)
        return torch.tanh(a) * torch.sigmoid(b)

class MilMean(nn.Module):
    def __init__(self, embedding_dim, n_classes, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim // 2, n_classes)
                )

    def forward(self, x):
        return self.net(x)

class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_classes):
        super().__init__()

        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[-1])

class TanhAttnSmall(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_classes):
        super().__init__()

        # attn module outputs score q @ k.T where in general q is (B,S,E) and k is (B,L,E)
        #   so score is (B,S,L) where S is target seq len and L is source seq len
        #   since we are classifying here and not decoding a seq, L = 1 => attn outputs (B,S,1)
        #   then to compute weights, softmax is over the S dim
        self.attn = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1))

        self.classifier = nn.Linear(embedding_dim, n_classes)


    def forward(self, x):
        # x is (S,B,E) = (seq len, batch size, embedding dim)
        x = x.permute(1,0,2)  # (B,S,E)

        attn_logits = self.attn(x)               # (B,S,1)
        attn_logits = attn_logits.transpose(1,2) # (B,1,S)
        self.weights = F.softmax(attn_logits, 2)

        attn_out = torch.bmm(self.weights, x)    # (B,1,S) @ (B,S,E) -> (B,1,E)
        attn_out = attn_out.flatten(1)           # (B,1*E)

        logits = self.classifier(attn_out)       # (B,n_classes)

        return logits

class TanhAttnBig(TanhAttnSmall):
    def __init__(self, embedding_dim, hidden_dim, n_classes, dropout=0.1):
        super().__init__(embedding_dim, hidden_dim, n_classes)

        self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim // 2, n_classes)
                )

class MhaSmall(nn.Module):
    def __init__(self, embedding_dim, n_classes, seq_len=12, attn_embed_dim=64, nh=2, dropout=0.1):
        super().__init__()

        self.in_proj_q = nn.Sequential(
                nn.Conv1d(seq_len, 1, 1),                               # (B,S,e) -> (B,1,e) for target sequence length 1
                nn.Linear(embedding_dim, attn_embed_dim))               # (B,1,e) -> (B,1,E) for embedding_dim -> attn_embed_dim
        self.in_proj_kv = nn.Linear(embedding_dim, 2*attn_embed_dim)    # proj kv to (B,L,2*dkv) before splitting
        self.mha = nn.MultiheadAttention(attn_embed_dim, nh, dropout)

        self.classifier = nn.Linear(attn_embed_dim, n_classes)


    def forward(self, x):
        q = self.in_proj_q(x)                       # (B,S,E) -> (B,1,E)
        k, v = self.in_proj_kv(x).chunk(2, dim=2)   # (B,L,dkv) each

        # transpose to MHA module view (S,B,E)
        q = q.transpose(0,1)
        k = k.transpose(0,1)
        v = v.transpose(0,1)

        attn_out, self.weights = self.mha(q, k, v)  # (1,B,E) and (B,1,L)

        logits = self.classifier(attn_out.squeeze(0))

        return logits
