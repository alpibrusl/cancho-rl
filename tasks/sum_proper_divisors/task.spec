name: sum_proper_divisors
ret: int
sig: fn sum_proper_divisors(n: int) -> [] int
desc: The sum of the positive divisors of n that are smaller than n (1 included when n > 1). sum_proper_divisors(1) is 0. Domain: 1 <= n <= 100000.
example: sum_proper_divisors(1) == 0
example: sum_proper_divisors(28) == 28
example: sum_proper_divisors(12496) == 14288
case: 1
case: 2
case: 3
case: 4
case: 6
case: 12
case: 28
case: 36
case: 97
case: 100
case: 496
case: 8128
case: 12496
case: 99991
case: 100000
case: 945
case: 220
case: 284
