name: count_odd_digits
ret: int
kind: loop
sig: fn count_odd_digits(n: int) -> [] int
desc: How many of the decimal digits of n are odd (1, 3, 5, 7 or 9). Domain: 0 <= n <= 1000000000.
example: count_odd_digits(135) == 3
example: count_odd_digits(2468) == 0
example: count_odd_digits(0) == 0
case: 0
case: 1
case: 2
case: 3
case: 4
case: 5
case: 7
case: 9
case: 10
case: 11
case: 12
case: 16
case: 20
case: 25
case: 36
case: 50
case: 64
case: 99
case: 100
case: 101
case: 255
case: 1000
case: 4096
case: 12345
case: 99999
case: 100000
case: 1000000
case: 13579
case: 24680
