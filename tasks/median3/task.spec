name: median3
ret: int
kind: branch
sig: fn median3(a: int, b: int, c: int) -> [] int
desc: The median of the three arguments: the value that is left in the middle when they are sorted. Ties are fine: median3(2, 2, 9) is 2. Any int.
example: median3(-5, -5, -5) == -5
example: median3(0, 0, 0) == 0
example: median3(3, 3, 3) == 3
case: -5 -5 -5
case: -5 -5 0
case: -5 -5 3
case: -5 -5 9
case: -5 -5 1099511627776
case: -5 0 -5
case: -5 0 0
case: -5 0 3
case: -5 0 9
case: -5 0 1099511627776
case: -5 3 -5
case: -5 3 0
case: -5 3 3
case: -5 3 9
case: -5 3 1099511627776
case: -5 9 -5
case: -5 9 0
case: -5 9 3
case: -5 9 9
case: -5 9 1099511627776
case: 0 -5 -5
case: 0 -5 0
case: 0 -5 3
case: 0 -5 9
case: 0 -5 1099511627776
case: 0 0 -5
case: 0 0 0
case: 0 0 3
case: 0 0 9
case: 0 0 1099511627776
case: 0 3 -5
case: 0 3 0
case: 0 3 3
case: 0 3 9
case: 0 3 1099511627776
case: 0 9 -5
case: 0 9 0
case: 0 9 3
case: 0 9 9
case: 0 9 1099511627776
case: 3 -5 -5
case: 3 -5 0
case: 3 -5 3
case: 3 -5 9
case: 3 -5 1099511627776
case: 3 0 -5
case: 3 0 0
case: 3 0 3
case: 3 0 9
case: 3 0 1099511627776
case: 3 3 -5
case: 3 3 0
case: 3 3 3
case: 3 3 9
case: 3 3 1099511627776
case: 3 9 -5
case: 3 9 0
case: 3 9 3
case: 3 9 9
case: 3 9 1099511627776
case: 9 -5 -5
case: 9 -5 0
case: 9 -5 3
case: 9 -5 9
case: 9 -5 1099511627776
case: 9 0 -5
case: 9 0 0
case: 9 0 3
case: 9 0 9
case: 9 0 1099511627776
case: 9 3 -5
case: 9 3 0
case: 9 3 3
case: 9 3 9
case: 9 3 1099511627776
case: 9 9 -5
case: 9 9 0
case: 9 9 3
case: 9 9 9
case: 9 9 1099511627776
