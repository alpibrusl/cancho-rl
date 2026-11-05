name: first_digit
ret: int
kind: loop
sig: fn first_digit(n: int) -> [] int
desc: The leftmost decimal digit of n. Domain: 0 <= n <= 1000000000.
example: first_digit(1234) == 1
example: first_digit(9) == 9
example: first_digit(70) == 7
case: 0
case: 1
case: 9
case: 10
case: 19
case: 20
case: 99
case: 100
case: 305
case: 999
case: 1000
case: 54321
case: 100000
case: 987654321
