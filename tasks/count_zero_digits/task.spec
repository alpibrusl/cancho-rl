name: count_zero_digits
ret: int
kind: loop
sig: fn count_zero_digits(n: int) -> [] int
desc: How many of the decimal digits of n are 0; the number 0 itself has one digit, a 0. Domain: 0 <= n <= 1000000000.
example: count_zero_digits(1001) == 2
example: count_zero_digits(0) == 1
example: count_zero_digits(123) == 0
case: 0
case: 1
case: 10
case: 100
case: 101
case: 1000
case: 1001
case: 1010
case: 2020
case: 12034
case: 100000
case: 900009
case: 1000000000
