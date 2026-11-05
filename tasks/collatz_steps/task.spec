name: collatz_steps
ret: int
kind: loop
sig: fn collatz_steps(n: int) -> [] int
desc: How many steps the Collatz process takes to reach 1 from n: if n is even halve it, otherwise replace it with 3n + 1, counting each step. collatz_steps(1) is 0. Domain: 1 <= n <= 1000000.
example: collatz_steps(1) == 0
example: collatz_steps(9) == 19
example: collatz_steps(77031) == 350
case: 1
case: 2
case: 3
case: 6
case: 7
case: 9
case: 27
case: 97
case: 871
case: 6171
case: 77031
case: 837799
case: 999999
case: 1000000
case: 8
case: 64
