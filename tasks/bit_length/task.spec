name: bit_length
ret: int
kind: loop
sig: fn bit_length(n: int) -> [] int
desc: How many binary digits n needs: 1 for 1, 2 for 2 and 3, 3 for 4 to 7, and 0 for 0. Domain: 0 <= n <= 9223372036854775807.
example: bit_length(0) == 0
example: bit_length(10) == 4
example: bit_length(1000) == 10
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
case: 256
case: 1099511627776
case: 4611686018427387903
case: 4611686018427387904
case: 9223372036854775807
