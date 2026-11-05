name: int_sqrt
ret: int
sig: fn int_sqrt(n: int) -> [] int
desc: The largest whole number r with r * r <= n (the square root rounded down). Domain: 0 <= n <= 1000000000.
example: int_sqrt(16) == 4
example: int_sqrt(17) == 4
example: int_sqrt(0) == 0
case: 0
case: 1
case: 2
case: 3
case: 4
case: 8
case: 9
case: 10
case: 15
case: 16
case: 17
case: 24
case: 25
case: 26
case: 99
case: 100
case: 101
case: 9999
case: 10000
case: 10001
case: 999999
case: 1000000
case: 999999999
case: 1000000000
