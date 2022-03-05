# Solving linear equations in the log semiring

This module exposes a single function `fix(a, b, block_size)` that
solves the linear system of equations

    (exp x) = (exp a) (exp x) + (exp b)                             (1)

where

- a is an n x n matrix
- b is a vector of size n or n x r
- x is a vector with the same size as b
- exp is taken elementwise

The reason the equations don't have the form

    (exp a) (exp x) + (exp b) = 0                                   (2)

like torch.linalg.solve is that in the log semiring, we don't have
negative numbers. The equation (1) uses only nonnegative coefficients,
has a solution with nonnegative entries, and uses the same amount of
memory as (2).

The equations are solved block by block, and block_size controls how
big the blocks are. A smaller block_size means that a greater fraction
of the computation will be done by matrix multiplications, but a
larger block_size means that the matrix multiplications will be larger
and therefore more parallelizable.
