name: is_perfect_number
ret: bool
sig: fn is_perfect_number(n: int) -> [] bool
desc: True when n equals the sum of its proper divisors (the divisors smaller than n, 1 included). 6 = 1 + 2 + 3 is perfect; 1 is not. Domain: 1 <= n <= 10000.
example: is_perfect_number(6) == true
example: is_perfect_number(12) == false
example: is_perfect_number(28) == true
case: 1
case: 2
case: 3
case: 4
case: 5
case: 6
case: 7
case: 8
case: 9
case: 10
case: 12
case: 15
case: 20
case: 27
case: 28
case: 29
case: 30
case: 100
case: 200
case: 495
case: 496
case: 497
case: 500
case: 1000
case: 8128
case: 8129
case: 9999
case: 10000
