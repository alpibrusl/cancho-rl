name: count_even_digits
ret: int
kind: loop
sig: fn count_even_digits(n: int) -> [] int
desc: How many of the decimal digits of n are even (0 is even). The number 0 has one digit, 0. Domain: 0 <= n <= 1000000.
example: count_even_digits(2468) == 4
example: count_even_digits(135) == 0
example: count_even_digits(0) == 1
case: 0
case: 1
case: 2
case: 8
case: 9
case: 10
case: 11
case: 20
case: 22
case: 101
case: 246
case: 1357
case: 2468
case: 10000
case: 13579
case: 24680
case: 100000
case: 123456
case: 987654
case: 1000000
