name: abs_val
ret: int
sig: fn abs_val(x: int) -> [] int
desc: The absolute value of x: x when it is zero or positive, otherwise -x. Domain: -1000000 <= x <= 1000000.
example: abs_val(-5) == 5
example: abs_val(7) == 7
example: abs_val(0) == 0
case: -1000000
case: -1000
case: -99
case: -10
case: -2
case: -1
case: 0
case: 1
case: 2
case: 10
case: 99
case: 1000
case: 1000000
