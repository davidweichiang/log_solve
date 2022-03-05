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

like torch.linalg.solve is that solving (2) requires subtraction, that
is, being able to find z such that (exp z) = (exp x) - (exp y), but if
y > x then z does not exist.

But if we define the _star_ operation as

    (exp x)* = 1 + (exp x) + (exp x)^2 + ... = 1/(1-(exp x))

then we can solve (1) using only addition, multiplication, and star
[Lehmann, 1977]. Intuitively, the solution is

    (exp x) = (exp b) + (exp a) (exp b) + (exp a)^2 (exp b) + ...
            = (I - (exp a))^{-1} (exp b).

The equations are solved block by block, and block_size controls how
big the blocks are. A smaller block_size means that a greater fraction
of the computation will be done by matrix multiplications, but a
larger block_size means that the matrix multiplications will be larger
and therefore more parallelizable.

# References

[Lehmann, 1977]: Daniel J. Lehmann. Algebraic structures for
transitive closure. Theoretical Computer Science 4:59-76,
1977. https://doi.org/10.1016/0304-3975(77)90056-1.
