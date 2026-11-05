name: quadrant
ret: int
kind: branch
sig: fn quadrant(x: int, y: int) -> [] int
desc: The quadrant of the point (x, y): 1 when x and y are both positive, 2 when x is negative and y positive, 3 when both are negative, 4 when x is positive and y negative, and 0 when the point is on an axis (x or y is 0). Domain: -100 <= x, y <= 100.
example: quadrant(3, 4) == 1
example: quadrant(-3, 4) == 2
example: quadrant(0, 5) == 0
case: -9 -9
case: -9 -1
case: -9 0
case: -9 1
case: -9 9
case: -1 -9
case: -1 -1
case: -1 0
case: -1 1
case: -1 9
case: 0 -9
case: 0 -1
case: 0 0
case: 0 1
case: 0 9
case: 1 -9
case: 1 -1
case: 1 0
case: 1 1
case: 1 9
case: 9 -9
case: 9 -1
case: 9 0
case: 9 1
case: 9 9
