name: is_even
ret: bool
sig: fn is_even(x: int) -> [] bool
desc: True when x is even. Any int, negative ones included: -4 is even and -3 is not. (% on a negative int can be negative; compare it with == 0 only.)
example: is_even(-1000000000) == true
example: is_even(3) == false
example: is_even(100) == true
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
