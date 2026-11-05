name: sum_to
ret: int
kind: branch
sig: fn sum_to(n: int) -> [] int
desc: 1 + 2 + ... + n, and 0 when n is 0. Domain: 0 <= n <= 4000000000, so the answer fits an int but a loop of that length is far too slow: use the formula.
example: sum_to(0) == 0
example: sum_to(10) == 55
example: sum_to(65536) == 2147516416
case: 0
case: 1
case: 2
case: 3
case: 10
case: 100
case: 12345
case: 1000000
case: 65536
case: 1000000000
case: 3000000000
case: 4000000000
