name: trailing_zero_bits
ret: int
sig: fn trailing_zero_bits(n: int) -> [] int
desc: How many times 2 divides n, i.e. the number of zero bits at the end of n in binary. Domain: 1 <= n <= 1000000000000.
example: trailing_zero_bits(8) == 3
example: trailing_zero_bits(6) == 1
example: trailing_zero_bits(7) == 0
case: 1
case: 2
case: 3
case: 4
case: 5
case: 6
case: 7
case: 8
case: 12
case: 16
case: 24
case: 40
case: 64
case: 96
case: 100
case: 128
case: 1000
case: 1024
case: 4096
case: 65536
case: 1000000
case: 1048576
case: 536870912
case: 1073741824
case: 549755813888
