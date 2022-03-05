__all__ = ['fix']

import torch
import torch_semiring_einsum

def star(x):
    # log(1/(1-exp(x)))
    return -torch.log1p(-torch.exp(x)) if x < 0 else torch.inf

def add_(x, y):
    x.copy_(x.logaddexp(y))

def mul_(x, y):
    x.add_(y)

vv_eq = torch_semiring_einsum.compile_equation('i,i->')
vm_eq = torch_semiring_einsum.compile_equation('i,ij->j')
mm_eq = torch_semiring_einsum.compile_equation('ij,jk->ik')

def dot(x, y, block_size):
    if y.ndim == 1:
        return torch_semiring_einsum.log_einsum_forward(vv_eq, x, y, block_size=block_size)
    elif y.ndim == 2:
        return torch_semiring_einsum.log_einsum_forward(vm_eq, x, y, block_size=block_size)

def matmul(x, y, block_size):
    return torch_semiring_einsum.log_einsum_forward(mm_eq, x, y, block_size=block_size)
    
outer_eq = torch_semiring_einsum.compile_equation('i,j->ij')
def outer(x, y, block_size):
    return torch_semiring_einsum.log_einsum_forward(outer_eq, x, y, block_size=block_size)


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
    for i in range(1, n):
        add_(b[i], dot(l[i,:i], b[:i], block_size))

        
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
    if n > 0:
        mul_(b[n-1], star(u[n-1, n-1]))
        for i in range(n-2, -1, -1):
            add_(b[i], dot(u[i,i+1:], b[i+1:], block_size))
            mul_(b[i], star(u[i, i]))

            
def lu_(a, block_size):
    """If A is an m x n matrix, where m >= n, find L and U such that:
    
    - L is an m x n lower strictly triangular matrix
    - U is an n x n upper triangular matrix
    - I-A = LU

    Result:
    - I-L is written to the lower trapezoid of A
    - I-U is written to the upper triangle of A

    cf. Golub and van Loan, 3rd ed., Algorithm 3.4.1 (p. 112)

    We can't do pivoting, but pivoting is not necessary because A
    is diagonally dominant.

    Time: O(mn^2)

    """
    m, n = a.shape
    for k in range(n):
        if a[k,k] != 0.:
            mul_(a[k+1:,k], star(a[k,k]))
            add_(a[k+1:,k+1:], outer(a[k+1:,k], a[k,k+1:], block_size=block_size))


def fix(a, b, block_size):
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
        lu_(a[k:,k:k+r], r)
        l11 = a[k:k+r,k:k+r]
        l21 = a[k+r:,k:k+r]
        u12 = a[k:k+r,k+r:]
        fix_stril_(l11, u12, r)
        add_(a[k+r:,k+r:], matmul(l21, u12, r))
    fix_stril_(a, b, r)
    fix_triu_(a, b, r)
    return b
