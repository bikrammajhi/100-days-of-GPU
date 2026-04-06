#!/usr/bin/env python3
"""Linear Attention Benchmark: PyTorch vs Triton"""
import torch, triton, triton.language as tl, time, sys

# ============================================================================
# Official PyTorch Reference Implementation 
# ============================================================================
def get_mask(n, slope=1):
    mask = torch.triu(torch.zeros(n, n).float().fill_(float("-inf")), 1)
    for i in range(n):
        x = torch.arange(i + 1)
        y = slope * x
        mask[i, : i + 1] = -torch.flip(y, [0])
    return torch.exp(mask)

def get_full_mask(n, slopes):
    if slopes is None:
        mask = torch.tril(torch.ones((n, n)))
    else:
        arr = []
        for slope in slopes:
            arr.append(get_mask(n, slope.item()))
        mask = torch.stack(arr, dim=0)
    return mask

def linear_attn_pytorch(q, k, v, s=None):
    """Official PyTorch reference: dense causal linear attention.
    Keeps computation in float16 end-to-end so numerical precision matches
    Triton's float16 I/O path (block accumulation in fp32, store in fp16).
    """
    b, h, n, d = q.shape
    mask = get_full_mask(n, s).to(q.device).to(q.dtype)
    qk = torch.matmul(q, k.transpose(2, 3)) * mask
    return torch.matmul(qk, v)

# ============================================================================
# Official Triton Implementation 
# ============================================================================
@triton.jit
def _fwd_kernel(Q, K, V, Out, b: tl.constexpr, h: tl.constexpr, n: tl.constexpr, d: tl.constexpr, e: tl.constexpr, BLOCK: tl.constexpr, NUM_BLOCK: tl.constexpr, BLOCK_MODEL: tl.constexpr):
    ##### get offset
    off_bh = tl.program_id(0)
    off_bh % h
    off_e = tl.program_id(1)
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    # channel offset
    e_offset = off_e * BLOCK_MODEL
    ##### get block ptr
    Q_block_ptr = Q + qk_offset + tl.arange(0, d)[None, :]
    K_trans_block_ptr = K + qk_offset + tl.arange(0, d)[:, None]
    V_block_ptr = V + v_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]
    O_block_ptr = Out + o_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]
    ##### init diag decay(Lambda); q, k decay; kv
    # q, k decay
    off_block = tl.arange(0, BLOCK)
    # diag decay
    index = off_block[:, None] - off_block[None, :]
    kv = tl.zeros([d, BLOCK_MODEL], dtype=tl.float32)
    ##### compute
    for i in range(NUM_BLOCK):
        # load
        q = tl.load(Q_block_ptr + off_block[:, None] * d, mask=off_block[:, None] < n, other=0.0).to(tl.float32)
        k_trans = tl.load(K_trans_block_ptr + off_block[None, :] * d, mask=off_block[None, :] < n, other=0.0).to(tl.float32)
        v = tl.load(V_block_ptr + off_block[:, None] * e, mask=off_block[:, None] < n, other=0.0).to(tl.float32)
        # compute
        qk = tl.dot(q, k_trans)
        qk = tl.where(index >= 0, qk, 0)
        o_intra = tl.dot(qk, v)
        o_inter = tl.dot(q, kv)
        o = o_intra + o_inter
        # save and update
        tl.store(O_block_ptr + off_block[:, None] * e, o.to(O_block_ptr.dtype.element_ty), mask=off_block[:, None] < n)
        kv += tl.dot(k_trans, v)
        off_block += BLOCK

def linear_attn_triton(q, k, v):
    """Official Triton linear attention (forward only, no decay)."""
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    b, h, n, d = q.shape; e = v.shape[-1]
    o = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)
    BLOCK, NUM_BLOCK, BLOCK_MODEL = 64, triton.cdiv(n, 64), min(triton.next_power_of_2(e), 32)
    _fwd_kernel[(b * h, triton.cdiv(e, BLOCK_MODEL))](q, k, v, o, b, h, n, d, e, BLOCK=BLOCK, NUM_BLOCK=NUM_BLOCK, BLOCK_MODEL=BLOCK_MODEL)
    return o

# ============================================================================
# Benchmark & I/O
# ============================================================================
def bench(fn, args, w=10, i=100, dev="cuda"):
    for _ in range(w): fn(*args)
    if dev == "cuda": torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(i): fn(*args)
    if dev == "cuda": torch.cuda.synchronize()
    return (time.perf_counter() - t) / i * 1000

def check(p, t):
    ok = torch.allclose(p.float(), t.float(), atol=0.5, rtol=0.05)
    if not ok:
        d = (p.float() - t.float()).abs()
        print(f"Max diff: {d.max().item():.4f}, Mean: {d.mean().item():.4f}")
    return ok

def run_bench(b, h, n, d, e, dev):
    q, k, v = [torch.randn(b, h, n, x, device=dev, dtype=torch.float16) for x in (d, d, e)]
    pt = bench(linear_attn_pytorch, (q, k, v), dev=dev)
    tt = bench(linear_attn_triton, (q, k, v), dev=dev)
    p_out, t_out = linear_attn_pytorch(q, k, v), linear_attn_triton(q, k, v)
    return {'n': n, 'pt': pt, 'tt': tt, 'tf': 2*b*h*n*n*(d+e)/tt/1e9, 'spd': pt/tt, 'ok': check(p_out, t_out)}

def table(hdr, rows):
    w = [max(len(h), max(len(str(r[i])) for r in rows)) for i, h in enumerate(hdr)]
    fmt = " | ".join(f"{{:<{w[i]}}}" for i in range(len(w)))
    sep = "-+-".join("-"*x for x in w)
    print(f"\n{fmt.format(*hdr)}\n{sep}")
    for r in rows: print(fmt.format(*[str(x) for x in r]))
    print(sep)

def run(scale=True, **params):
    dev = params.get('device', 'cuda') if torch.cuda.is_available() else 'cpu'
    B, G, C, Y, R, W = "\033[1m", "\033[92m", "\033[96m", "\033[93m", "\033[91m", "\033[0m"
    print(f"\n{'█'*16} GPU BENCHMARK | {dev.upper()} | SCALE = {scale} {'█'*16}{W}")
    cfgs = [(params.get('batch',4), params.get('heads',8), n, params.get('dim',64), params.get('out_dim',64)) 
            for n in ([128,256,512,1024,2048] if scale else [params.get('seq_len',512)])]
    res = []
    for c in cfgs:
        print(f"Running B={c[0]},H={c[1]},N={c[2]}...")
        res.append(run_bench(*c, dev))
    headers = ["Sequence Len", "PyTorch (ms)", "Triton (ms)", "Speedup", "Triton TFLOPS", "Status"]
    table(headers, 
          [[r['n'], f"{r['pt']:.4f}", f"{r['tt']:.4f}", f"{r['spd']:.2f}x", f"{r['tf']:.2f}", "✅" if r['ok'] else "❌"] for r in res])
    print(f"Avg Speedup: {sum(r['spd'] for r in res)/len(res):.2f}x")
    return {'status': 'completed', 'device': dev, 'config': params}
