name: is_perfect_square
ret: bool
kind: loop
sig: fn is_perfect_square(n: int) -> [] bool
desc: True when n is the square of an integer. Negative numbers are not. Domain: -100 <= n <= 1000000000000.
example: is_perfect_square(-100) == false
example: is_perfect_square(8) == false
example: is_perfect_square(101) == false
case: -100
case: -1
case: 0
case: 1
case: 2
case: 3
case: 4
case: 8
case: 9
case: 15
case: 16
case: 17
case: 99
case: 100
case: 101
case: 999999999999
case: 1000000000000
case: 999999000001
case: 999998000001
case: 999998000000
case: 450359962737
