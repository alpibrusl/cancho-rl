name: steps_to_zero
ret: int
kind: loop
sig: fn steps_to_zero(n: int) -> [] int
desc: How many steps it takes to bring n down to 0 when each step halves n if it is even and subtracts 1 if it is odd. steps_to_zero(14) is 6: 14, 7, 6, 3, 2, 1, 0. Domain: 0 <= n <= 1000000000.
example: steps_to_zero(14) == 6
example: steps_to_zero(8) == 4
example: steps_to_zero(0) == 0
case: 0
case: 1
case: 2
case: 3
case: 4
case: 5
case: 6
case: 7
case: 8
case: 14
case: 15
case: 16
case: 100
case: 123
case: 1000
case: 1023
case: 1024
case: 65535
case: 1000000
case: 999999999
