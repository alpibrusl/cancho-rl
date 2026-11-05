name: count_divisors
ret: int
kind: loop
sig: fn count_divisors(n: int) -> [] int
desc: How many positive integers divide n exactly, 1 and n included. Domain: 1 <= n <= 100000.
example: count_divisors(1) == 1
example: count_divisors(16) == 5
example: count_divisors(100) == 9
case: 1
case: 2
case: 3
case: 4
case: 6
case: 12
case: 16
case: 25
case: 36
case: 49
case: 60
case: 97
case: 100
case: 360
case: 720
case: 5040
case: 65536
case: 99991
case: 83160
case: 100000
