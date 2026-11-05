name: sum_range
ret: int
kind: loop
sig: fn sum_range(a: int, b: int) -> [] int
desc: The sum of every whole number from a to b, both included. Domain: -1000 <= a <= b <= 1000.
example: sum_range(1, 4) == 10
example: sum_range(-2, 2) == 0
example: sum_range(5, 5) == 5
case: -1000 -1
case: -1000 0
case: -1000 1
case: -1000 2
case: -1000 8
case: -1000 50
case: -1000 1000
case: -50 -1
case: -50 0
case: -50 1
case: -50 2
case: -50 8
case: -50 50
case: -50 1000
case: -3 -1
case: -3 0
case: -3 1
case: -3 2
case: -3 8
case: -3 50
case: -3 1000
case: -1 -1
case: -1 0
case: -1 1
case: -1 2
case: -1 8
case: -1 50
case: -1 1000
case: 0 0
case: 0 1
case: 0 2
case: 0 8
case: 0 50
case: 0 1000
case: 1 1
case: 1 2
case: 1 8
case: 1 50
case: 1 1000
case: 2 2
case: 2 8
case: 2 50
case: 2 1000
case: 7 8
case: 7 50
case: 7 1000
case: 100 1000
case: 999 1000
