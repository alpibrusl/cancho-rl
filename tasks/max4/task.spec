name: max4
ret: int
kind: branch
sig: fn max4(a: int, b: int, c: int, d: int) -> [] int
desc: The largest of the four numbers. Domain: -100 <= a, b, c, d <= 100.
example: max4(1, 2, 3, 4) == 4
example: max4(9, 2, 3, 4) == 9
example: max4(-1, -2, -3, -4) == -1
case: -5 -5 -9 -3
case: -5 -5 -9 8
case: -5 -5 0 -3
case: -5 -5 0 8
case: -5 -5 7 -3
case: -5 -5 7 8
case: -5 3 -9 -3
case: -5 3 -9 8
case: -5 3 0 -3
case: -5 3 0 8
case: -5 3 7 -3
case: -5 3 7 8
case: -5 7 -9 -3
case: -5 7 -9 8
case: -5 7 0 -3
case: -5 7 0 8
case: -5 7 7 -3
case: -5 7 7 8
case: 0 -5 -9 -3
case: 0 -5 -9 8
case: 0 -5 0 -3
case: 0 -5 0 8
case: 0 -5 7 -3
case: 0 -5 7 8
case: 0 3 -9 -3
case: 0 3 -9 8
case: 0 3 0 -3
case: 0 3 0 8
case: 0 3 7 -3
case: 0 3 7 8
case: 0 7 -9 -3
case: 0 7 -9 8
case: 0 7 0 -3
case: 0 7 0 8
case: 0 7 7 -3
case: 0 7 7 8
case: 7 -5 -9 -3
case: 7 -5 -9 8
case: 7 -5 0 -3
case: 7 -5 0 8
case: 7 -5 7 -3
case: 7 -5 7 8
case: 7 3 -9 -3
case: 7 3 -9 8
case: 7 3 0 -3
case: 7 3 0 8
case: 7 3 7 -3
case: 7 3 7 8
case: 7 7 -9 -3
case: 7 7 -9 8
case: 7 7 0 -3
case: 7 7 0 8
case: 7 7 7 -3
case: 7 7 7 8
