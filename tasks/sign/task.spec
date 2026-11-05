name: sign
ret: int
kind: branch
sig: fn sign(x: int) -> [] int
desc: The sign of x: -1 if x is negative, 0 if x is zero, 1 if x is positive. Any int, including the extremes.
example: sign(-1000000000) == -1
example: sign(3) == 1
example: sign(100) == 1
case: -1000000000
case: -17
case: -3
case: -2
case: -1
case: 0
case: 1
case: 2
case: 3
case: 4
case: 5
case: 7
case: 10
case: 16
case: 17
case: 99
case: 100
case: 101
case: 255
case: 1000
case: 65535
case: 999999
case: 1000000000
case: -9223372036854775807
case: 9223372036854775807
