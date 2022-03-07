import torch
from log_solve import fix

import timeit

def rand(*shape):
    return -torch.empty(shape).uniform_(0, 10)

for n in [1000]:
    for r in [1, 10, 100]:
        for block_size in [100, 1000]:
            if block_size > n: continue
            def f_ours():
                a = rand(n, n)
                b = rand(n, r)
                x = fix(a, b, block_size)

            def f_torch():
                a = rand(n, n)
                b = rand(n, r)
                x = torch.linalg.solve(a.exp(), b.exp()).log()

            torch.manual_seed(0)
            t_ours = timeit.timeit(f_ours, number=10)
            torch.manual_seed(0)
            t_torch = timeit.timeit(f_torch, number=10)
            print(n, r, block_size, t_ours, t_torch, t_ours/t_torch)
        
