name: cmp3
ret: int
kind: branch
sig: fn cmp3(a: int, b: int) -> [] int
desc: -1 when a is less than b, 0 when they are equal, 1 when a is greater than b. Domain: -100 <= a, b <= 100.
example: cmp3(1, 2) == -1
example: cmp3(2, 2) == 0
example: cmp3(5, 3) == 1
case: -100 -100
case: -100 -2
case: -100 -1
case: -100 0
case: -100 1
case: -100 2
case: -100 100
case: -2 -100
case: -2 -2
case: -2 -1
case: -2 0
case: -2 1
case: -2 2
case: -2 100
case: -1 -100
case: -1 -2
case: -1 -1
case: -1 0
case: -1 1
case: -1 2
case: -1 100
case: 0 -100
case: 0 -2
case: 0 -1
case: 0 0
case: 0 1
case: 0 2
case: 0 100
case: 1 -100
case: 1 -2
case: 1 -1
case: 1 0
case: 1 1
case: 1 2
case: 1 100
case: 2 -100
case: 2 -2
case: 2 -1
case: 2 0
case: 2 1
case: 2 2
case: 2 100
case: 100 -100
case: 100 -2
case: 100 -1
case: 100 0
case: 100 1
case: 100 2
case: 100 100
