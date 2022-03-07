import torch
from log_solve import add_, fix_stril_, fix_triu_, lu_, fix

import unittest

def rand(*shape):
    return -torch.empty(shape).uniform_(0, 10)

def f(a, x, b):
    return torch.log(torch.exp(a) @ torch.exp(x) + torch.exp(b))

def close(x, y):
    # Change inf-inf = nan to 0
    return torch.norm((x-y).nan_to_num()) < 1e-6

class TestSolve(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.double)
        torch.manual_seed(0)

    def _test_fix_stril(self, lshape, bshape):
        for i in range(100):
            l = rand(*lshape).tril(-1)
            b = rand(*bshape)
            x = b.clone()
            fix_stril_(torch.log(l), x, 3)
            self.assertTrue(close(x, f(l, x, b)))

    def test_fix_stril_single(self):
        self._test_fix_stril((10, 10), (10,))
    def test_fix_stril_multiple(self):
        self._test_fix_stril((10, 10), (10, 5))

    def _test_fix_triu(self, ushape, bshape):
        for i in range(100):
            u = rand(*ushape).triu()
            b = rand(*bshape)
            x = b.clone()
            fix_triu_(torch.log(u), x, 5)
            self.assertTrue(close(x, f(u, x, b)))

    def test_fix_triu_single(self):
        self._test_fix_triu((10, 10), (10,))
    def test_fix_triu_multiple(self):
        self._test_fix_triu((10, 10), (10, 5))

    def test_lu(self):
        for i in range(100):
            m, n = 10, 5
            a = rand(m, n)
            l_plus_u = a.clone()
            lu_(l_plus_u)
            a = torch.exp(a)
            l_plus_u = torch.exp(l_plus_u)
            l = l_plus_u.tril(-1)
            u = l_plus_u[:n].triu()
            # I-A = (I-L) (I-U) = I - L - U + LU => A + L@U = L + U
            self.assertTrue(close(a + l@u, l_plus_u))

    def _test_fix(self, ashape, bshape):
        for i in range(100):
            a = rand(*ashape)
            b = rand(*bshape)
            x = fix(a, b, 3)
            self.assertTrue(close(x, f(a, x, b)))

    def test_fix_single(self):
        self._test_fix((10, 10), (10,))
    def test_fix_multiple(self):
        self._test_fix((10, 10), (10, 5))

unittest.main()
