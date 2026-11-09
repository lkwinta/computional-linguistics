import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Rotary Positional Embedding (RoPE) function
# Applies a rotation to each pair of dimensions in the input tensor based on position and frequency
# 
# It provides a way to encode positional information into the input tensor without adding explicit positional embeddings.
def apply_rope(x, start_pos = 0, base = 10000.0):
    N, H, T, D = x.shape

    # we divide the vector x into pairs 
    # q = [(q0, q1), (q2, q3), (q4, q5), ..., (q62, q63)]
    #
    positions = torch.arange(
        start_pos, start_pos + T,
        device=x.device,
        dtype=torch.float32
    )

    #
    # We want to "rotate" each pair with the corresponding angle
    # 
    # 
    freqs = base ** (
        -torch.arange(
            0, D, 2,
            device=x.device,
            dtype=torch.float32
        ) / D
    )

    angles = positions[:, None] * freqs[None, :]

    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]

    x_even = x[..., ::2]
    x_odd = x[..., 1::2]

    # we apply the rotation formula for vectors in each pair
    # 
    # x' = x * cos(theta) - y * sin(theta)
    # y' = x * sin(theta) + y * cos(theta)
    # 
    rotated = torch.stack(
        (
            x_even * cos - x_odd * sin,
            x_even * sin + x_odd * cos
        ),
        dim=-1
    )

    return rotated.flatten(-2).to(x.dtype)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, use_rope=False):
        super().__init__()

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        if use_rope:
            assert self.head_dim % 2 == 0, (
                "head_dim must be even when using RoPE"
            )

        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)

        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.use_rope = use_rope

    def forward(self, x):
        B, T, C = x.shape

        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        q = q.reshape(B, T, self.num_heads, self.head_dim)
        k = k.reshape(B, T, self.num_heads, self.head_dim)
        v = v.reshape(B, T, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.use_rope:
            q = apply_rope(q)
            k = apply_rope(k)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=scores.device),
            diagonal=1
        )

        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        attention = weights @ v

        attention = attention.transpose(1, 2).contiguous()
        attention = attention.view(B, T, C)

        return self.out_proj(attention)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()

        rms_inv = torch.rsqrt(
            xf.pow(2).mean(dim=-1,  keepdim=True) + self.eps
        )

        normalized = xf * rms_inv

        return (normalized * self.weight).to(x.dtype)    

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()

        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        x = F.silu(gate) * up

        x = self.down_proj(x)

        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_hidden_dim, feed_forward="mlp", use_rope=False):
        super().__init__()

        self.attention_norm = RMSNorm(dim)
        self.attention = MultiHeadAttention(dim, num_heads, use_rope=use_rope)

        self.feed_forward_norm = RMSNorm(dim)
        if feed_forward == "mlp":
            self.feed_forward = MLP(dim, ff_hidden_dim)
        elif feed_forward == "swiglu":
            self.feed_forward = SwiGLU(dim, ff_hidden_dim)
        else:
            raise ValueError(f"Unknown feed_forward type: {feed_forward}")
    
    def forward(self, x):
        normalized = self.attention_norm(x)

        attn_out = self.attention(normalized)

        x = x + attn_out

        normalized = self.feed_forward_norm(x)
        ff_out = self.feed_forward(normalized)

        x = x + ff_out

        return x


class LanguageTransformer(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 dim,
                 num_heads, 
                 ff_hidden_dim, 
                 num_layers, 
                 block_size, 
                 feed_forward="mlp", 
                 use_rope=False,
                 weight_tying=True):
        super().__init__()

        self.use_rope = use_rope
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, dim)

        if not use_rope:
            self.position_embedding = nn.Embedding(block_size, dim)

        self.blocks = nn.ModuleList()

        for _ in range(num_layers):
            block = TransformerBlock(dim, num_heads, ff_hidden_dim, feed_forward=feed_forward, use_rope=use_rope)
            self.blocks.append(block)

        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        if weight_tying:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def forward(self, token_ids):
        x = self.token_embedding(token_ids)

        if not self.use_rope:
            positions = torch.arange(token_ids.size(1), device=token_ids.device)
            x = x + self.position_embedding(positions)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = self.lm_head(x)
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

if __name__ == "__main__":

    config = dict(
        vocab_size=50257,
        dim=384,
        num_heads=6,
        num_layers=6,
        block_size=256,
    )

    # Wariant GPT-2-style positional embeddings
    gpt_learned = LanguageTransformer(
        **config,
        use_rope=False,
        ff_hidden_dim=4*384,
        feed_forward="mlp"
    )

    # Wariant z Rotary Position Embeddings
    gpt_rope = LanguageTransformer(
        **config,
        use_rope=True,
        ff_hidden_dim=4*384,
        feed_forward="mlp"
    )

    gpt_rope_swiglu = LanguageTransformer(
        **config,
        use_rope=True,
        feed_forward="swiglu",
        ff_hidden_dim=512,
    )

    token_ids = torch.randint(0, 50257, (2, 32))

    print("Learned:", gpt_learned(token_ids).shape)
    print("RoPE:", gpt_rope(token_ids).shape)
    print("RoPE Swiglu:", gpt_rope_swiglu(token_ids).shape)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())

    print("Learned:", count_parameters(gpt_learned))
    print("RoPE:", count_parameters(gpt_rope))
    print("RoPE Swiglu:", count_parameters(gpt_rope_swiglu))

    from nano_gpt import GPT, GPTConfig

    VOCAB_SIZE = 50257
    BLOCK_SIZE = 256
    BATCH_SIZE = 16
    STEPS = 20000
    EVAL_INTERVAL = 100
    EVAL_BATCHES = 20
    LEARNING_RATE = 3e-4

    N_LAYER = 6
    N_HEAD = 6
    N_EMBD = 384
    DROPOUT = 0.1

    config = GPTConfig(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=DROPOUT,
        bias=False,
    )
    model = GPT(config)

    print("Nano GPT:", count_parameters(model))
 