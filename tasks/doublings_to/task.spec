name: doublings_to
ret: int
kind: loop
sig: fn doublings_to(n: int) -> [] int
desc: Starting from 1 and doubling each time, how many doublings until the value is at least n. doublings_to(9) is 4 (1, 2, 4, 8, 16). Domain: 1 <= n <= 1000000000.
example: doublings_to(8) == 3
example: doublings_to(9) == 4
example: doublings_to(1) == 0
case: 1
case: 2
case: 3
case: 4
case: 5
case: 8
case: 9
case: 16
case: 17
case: 100
case: 1000
case: 1024
case: 1025
case: 65536
case: 1000000
case: 1000000000
