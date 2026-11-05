name: spread
ret: int
kind: branch
sig: fn spread(a: int, b: int, c: int) -> [] int
desc: The largest of the three numbers minus the smallest. Domain: -100 <= a, b, c <= 100.
example: spread(1, 5, 3) == 4
example: spread(2, 2, 2) == 0
example: spread(-4, 4, 0) == 8
case: -7 -7 -3
case: -7 -7 0
case: -7 -7 8
case: -7 2 -3
case: -7 2 0
case: -7 2 8
case: -7 9 -3
case: -7 9 0
case: -7 9 8
case: 0 -7 -3
case: 0 -7 0
case: 0 -7 8
case: 0 2 -3
case: 0 2 0
case: 0 2 8
case: 0 9 -3
case: 0 9 0
case: 0 9 8
case: 5 -7 -3
case: 5 -7 0
case: 5 -7 8
case: 5 2 -3
case: 5 2 0
case: 5 2 8
case: 5 9 -3
case: 5 9 0
case: 5 9 8
