name: popcount
ret: int
kind: loop
sig: fn popcount(n: int) -> [] int
desc: How many 1 bits n has in binary. Domain: 0 <= n <= 9223372036854775807.
example: popcount(0) == 0
example: popcount(16) == 1
example: popcount(999999) == 12
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
case: 1023
case: 1099511627776
case: 4611686018427387903
case: 9223372036854775807
case: 6148914691236517205
