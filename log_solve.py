__all__ = ['fix']

import torch

def star(x):
    # log(1/(1-exp(x)))
    return -torch.log1p(-torch.exp(x)) if x < 0 else torch.inf

def add_(x, y):
    torch.logaddexp(x, y, out=x)

def mul_(x, y):
    x.add_(y)


def matmul(a, b, block_size):
    """
    Space: O(rnp)
    """
    m, n = a.shape
    _, p = b.shape
    r = block_size
    dtype = a.dtype
    c = torch.empty(m, p, dtype=dtype)
    t = torch.empty(r, n, p, dtype=dtype)
    for i in range(0, m, r):
        if m-i < r: t = t[:m-i]
        # c[i:i+r] = a[i:i+r] @ b
        torch.add(a[i:i+r].unsqueeze(-1), b, out=t)
        torch.logsumexp(t, dim=1, out=c[i:i+r])
    return c
    
def outer(x, y):
    return x.unsqueeze(-1) + y


def fix_stril_(l, b, block_size):
    """Solve x = l @ x + b by forward substitution, where l is strictly
    lower triangular. If l is in fact not strictly lower triangular, the
    elements on and above the diagonal are treated as 0.

    Result:
    The solution is written to b.

    cf. Golub and van Loan, 3rd ed., Algorithm 3.1.1 (p. 89)

    Time: O(n^2 r) where r is the number of columns of b.
    """
    n = l.shape[0]
    r = block_size
    if b.ndim == 1:
        b = b.unsqueeze(-1)
    for i0 in range(0, n, r):
        i1 = min(n, i0+r)
        for i in range(i0, i1):
            add_(b[i+1:i1], outer(l[i+1:i1,i], b[i]))
        add_(b[i1:], matmul(l[i1:,i0:i1], b[i0:i1], r))
        

def fix_triu_(u, b, block_size):
    """Solve x = u @ x + b by backward substitution, where u is upper triangular.
    If u is in fact not upper triangular, the elements below the
    diagonal are treated as 0.

    Result:
    The solution is written to b.

    cf. Golub and van Loan, 3rd ed., Algorithm 3.1.2 (p. 89)

    Time: O(n^2 r) where r is the number of columns of b.
    """
    n = u.shape[0]
    r = block_size
    if b.ndim == 1:
        b = b.unsqueeze(-1)
    for i0 in reversed(range(0, n, r)):
        i1 = min(n, i0+r)
        for i in reversed(range(i0, i1)):
            mul_(b[i], star(u[i,i]))
            add_(b[i0:i], outer(u[i0:i,i], b[i]))
        add_(b[:i0], matmul(u[:i0,i0:i1], b[i0:i1], r))

        
def lu_(a):
    """If A is an m x n matrix, where m >= n, find L and U such that:
    
    - L is an m x n lower strictly triangular matrix
    - U is an n x n upper triangular matrix
    - I-A = LU

    Result:
    - I-L is written to the lower trapezoid of A
    - I-U is written to the upper triangle of A

    cf. Golub and van Loan, 3rd ed., Algorithm 3.4.1 (p. 112)

    We can't do pivoting because we are storing I-A and can't permute
    the rows of the implicit I.

    Time: O(mn^2)

    """
    m, n = a.shape
    for k in range(n):
        if a[k,k] != 0.:
            mul_(a[k+1:,k], star(a[k,k]))
            add_(a[k+1:,k+1:], outer(a[k+1:,k], a[k,k+1:]))

            
def fix_block_lu(a, b, block_size):
    """Solve x = a @ x + b by block Gaussian elimination.

    cf. Golub and van Loan, 3rd ed., Section 3.4.7 (p. 116)

    Time: O(n^3)
    """

    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f'a must be a square matrix (not {a.shape})')
    if not 1 <= b.ndim <= 2:
        raise ValueError(f'b must have 1 or 2 dimensions (not {b.ndim})')
    if a.shape[0] != b.shape[0]:
        raise ValueError(f'b must have the same number of rows as a (a has {a.shape[0]}, b has {b.shape[0]})')
    a = a.clone()
    b = b.clone()
    n = a.shape[0]
    r = block_size
    for k in range(0, n, r):
        lu_(a[k:,k:k+r])
        u12 = a[k:k+r,k+r:]
        fix_stril_(a[k:k+r,k:k+r], u12, r)
        add_(a[k+r:,k+r:], matmul(a[k+r:,k:k+r], u12, r))
    fix_stril_(a, b, r)
    fix_triu_(a, b, r)
    return b


def fix_floyd_warshall(a, b, block_size):
    """Not as fast, but included here for its simplicity."""
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f'a must be a square matrix (not {a.shape})')
    bshape = b.shape
    if b.ndim == 1:
        b = b.unsqueeze(-1)
    elif b.ndim > 2:
        raise ValueError(f'b must have 1 or 2 dimensions (not {b.ndim})')
    if a.shape[0] != b.shape[0]:
        raise ValueError(f'b must have the same number of rows as a (a has {a.shape[0]}, b has {b.shape[0]})')
    a = a.clone()
    b = b.clone()
    n = a.shape[0]
    for k in range(n):
        mul_(a[:,k],    star(a[k,k]))
        add_(b,         outer(a[:,k], b[k,:]))
        add_(a[:,k+1:], outer(a[:,k], a[k,k+1:]))
    return b.reshape(*bshape)

fix = fix_block_lu

