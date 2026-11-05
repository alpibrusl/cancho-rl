name: lowest_set_bit
ret: int
sig: fn lowest_set_bit(n: int) -> [] int
desc: The value of the lowest bit that is set in n: the largest power of two that divides n. lowest_set_bit(12) is 4. Domain: 1 <= n <= 1000000000.
example: lowest_set_bit(12) == 4
example: lowest_set_bit(7) == 1
example: lowest_set_bit(64) == 64
case: 1
case: 2
case: 3
case: 4
case: 5
case: 6
case: 7
case: 8
case: 10
case: 12
case: 24
case: 40
case: 48
case: 96
case: 100
case: 127
case: 128
case: 1000
case: 1024
case: 3072
case: 999999
case: 1000000
case: 536870912
case: 1000000000
