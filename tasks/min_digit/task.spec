name: min_digit
ret: int
kind: loop
sig: fn min_digit(n: int) -> [] int
desc: The smallest decimal digit of n. min_digit(0) is 0. Domain: 0 <= n <= 1000000000.
example: min_digit(4721) == 1
example: min_digit(905) == 0
example: min_digit(7) == 7
case: 0
case: 1
case: 5
case: 9
case: 10
case: 19
case: 90
case: 99
case: 101
case: 123
case: 321
case: 555
case: 4721
case: 9000
case: 12345
case: 98765
case: 111111
case: 987654321
case: 1000000000
