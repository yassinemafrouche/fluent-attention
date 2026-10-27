
```markdown
# 🌊 Fluent Attention

> **Sliding-window attention with learnable crystal pooling.**
> **760x faster than Flash Attention at 1M tokens on a T4.**

## 📖 The Story (Why this exists)

I am **not** a professional AI researcher, nor a seasoned CUDA programmer. I am just a curious person who saw a video of water flowing through a maze and wondered: *"Can a language model work like this?"*

What started as a philosophical idea (Water = local context, Ice = global memory) turned into a 20+ experiment journey. I failed 5 times (v2, v3, fp16, Loom, Wittgenstein Attention) before finally getting a working, blazing-fast CUDA kernel (v4). 

This repository is the result of that journey. It is a **Proof of Concept**, not a peer-reviewed paper. But the numbers are real, measured on a standard T4 GPU.

## 📊 Results

### Speed (attention only, T4)

| N (tokens) | Flash Attention | **Fluent (CUDA)** | Speedup |
|---|---|---|---|
| 16,384 | 27.9 ms | **3.1 ms** | 9x |
| 65,536 | 238 ms | **5.6 ms** | 43x |
| 262,144 | 3,958 ms | **20.9 ms** | 190x |
| **1,048,576** | **61,743 ms** | **81.3 ms** | **760x** |

### Quality (Shakespeare, 1.8M params, 3000 steps)

| Model | Val Loss | Perplexity |
|---|---|---|
| Full Attention | 5.3271 | 205.8 |
| Sliding Window (W=64) | 5.3086 | 202.1 |
| **Fluent (W=64 + Crystals)** | **5.1354** | **169.9** |

→ **16% lower perplexity** than sliding window, with only 4% more parameters.

## 🧠 How it works

Fluent combines two ideas:
1. **Local Water**: each token attends to its 64 nearest neighbors (sliding window, O(N·W)).
2. **Global Crystals**: every 64 tokens are pooled into a "crystal", and crystals attend to each other (O(N/W · N/W)).

Total complexity: **O(N·W + (N/W)²)** ≈ **O(N)**.

## 🛠️ Install & Use

```bash
pip install torch ninja
```

```python
import torch
from fluent import fluent_attention

q = torch.randn(1_000_000, 64, device="cuda")
k = torch.randn(1_000_000, 64, device="cuda")
v = torch.randn(1_000_000, 64, device="cuda")

output = fluent_attention(q, k, v, window=64, bq=32)
```

🔬 Reproduce

```bash
python benchmark.py
```

⚠️ Limitations (Honest Notes)

· Only tested on T4 (Turing). Performance on A100/H100 is unknown.
· Quality experiments use small models (1.8M params) and Shakespeare.
· Not benchmarked against Mistral, Longformer, or BigBird yet.
· PyTorch's own sliding window OOMs at 65K tokens (it materializes an N×N mask), which is why a custom CUDA kernel was necessary.

📜 License

MIT
