name: is_power_of_two
ret: bool
sig: fn is_power_of_two(n: int) -> [] bool
desc: True when n is 1, 2, 4, 8, ... Zero and negative numbers are not powers of two. Any int.
example: is_power_of_two(-1000000000) == false
example: is_power_of_two(7) == false
example: is_power_of_two(1000000000) == false
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
case: 6
case: 8
case: 9
case: 12
case: 64
case: 63
case: 65
case: 1099511627776
case: 1099511627777
case: 4611686018427387904
case: 9223372036854775807
case: -4611686018427387904
