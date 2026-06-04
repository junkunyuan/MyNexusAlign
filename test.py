import torch, time

x = torch.randn(8192, 8192, device="cuda")
y = torch.randn(8192, 8192, device="cuda")

for flag in [False, True]:
    torch.backends.cuda.matmul.allow_tf32 = flag
    torch.cuda.synchronize()
    t0 = time.time()

    for _ in range(10):
        z = x @ y

    torch.cuda.synchronize()
    print(flag, time.time() - t0)