name: count_halvings
ret: int
kind: loop
sig: fn count_halvings(n: int) -> [] int
desc: How many times n can be divided by 2 (whole-number division, dropping any remainder) before it reaches 1. count_halvings(9) is 3: 9 to 4 to 2 to 1. Domain: 1 <= n <= 1000000000.
example: count_halvings(8) == 3
example: count_halvings(1) == 0
example: count_halvings(9) == 3
case: 1
case: 2
case: 3
case: 4
case: 5
case: 7
case: 8
case: 9
case: 15
case: 16
case: 17
case: 100
case: 255
case: 256
case: 1000
case: 1023
case: 1024
case: 65536
case: 1000000
case: 1000000000
