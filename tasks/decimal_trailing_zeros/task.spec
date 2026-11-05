name: decimal_trailing_zeros
ret: int
kind: loop
sig: fn decimal_trailing_zeros(n: int) -> [] int
desc: How many zeros n ends with in decimal: decimal_trailing_zeros(1200) is 2. Domain: 1 <= n <= 1000000000.
example: decimal_trailing_zeros(1200) == 2
example: decimal_trailing_zeros(7) == 0
example: decimal_trailing_zeros(1000000) == 6
case: 1
case: 5
case: 10
case: 11
case: 20
case: 100
case: 101
case: 110
case: 1000
case: 1200
case: 5000
case: 10010
case: 123000
case: 1000000
case: 1000000000
