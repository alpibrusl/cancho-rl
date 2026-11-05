name: sum_squares_to
ret: int
kind: loop
sig: fn sum_squares_to(n: int) -> [] int
desc: 1*1 + 2*2 + ... + n*n, and 0 when n is 0. Domain: 0 <= n <= 1000000 (a loop is fine).
example: sum_squares_to(0) == 0
example: sum_squares_to(3) == 14
example: sum_squares_to(1000) == 333833500
case: 0
case: 1
case: 2
case: 3
case: 10
case: 100
case: 1000
case: 54321
case: 1000000
