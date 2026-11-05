name: sum_odd_digits
ret: int
kind: loop
sig: fn sum_odd_digits(n: int) -> [] int
desc: The sum of the odd decimal digits of n (even digits count as nothing). Domain: 0 <= n <= 1000000000.
example: sum_odd_digits(1234) == 4
example: sum_odd_digits(2468) == 0
example: sum_odd_digits(9) == 9
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
