name: clamp
ret: int
sig: fn clamp(x: int, lo: int, hi: int) -> [] int
desc: x limited to the range lo..hi: lo when x is below it, hi when x is above it, otherwise x. Domain: -100 <= x <= 100, -100 <= lo <= hi <= 100.
example: clamp(5, 0, 10) == 5
example: clamp(-3, 0, 10) == 0
example: clamp(42, 0, 10) == 10
case: -100 0 10
case: -100 -5 5
case: -100 3 3
case: -100 -100 100
case: -100 10 20
case: -100 -20 -10
case: -11 0 10
case: -11 -5 5
case: -11 3 3
case: -11 -100 100
case: -11 10 20
case: -11 -20 -10
case: -10 0 10
case: -10 -5 5
case: -10 3 3
case: -10 -100 100
case: -10 10 20
case: -10 -20 -10
case: -1 0 10
case: -1 -5 5
case: -1 3 3
case: -1 -100 100
case: -1 10 20
case: -1 -20 -10
case: 0 0 10
case: 0 -5 5
case: 0 3 3
case: 0 -100 100
case: 0 10 20
case: 0 -20 -10
case: 1 0 10
case: 1 -5 5
case: 1 3 3
case: 1 -100 100
case: 1 10 20
case: 1 -20 -10
case: 9 0 10
case: 9 -5 5
case: 9 3 3
case: 9 -100 100
case: 9 10 20
case: 9 -20 -10
case: 10 0 10
case: 10 -5 5
case: 10 3 3
case: 10 -100 100
case: 10 10 20
case: 10 -20 -10
case: 11 0 10
case: 11 -5 5
case: 11 3 3
case: 11 -100 100
case: 11 10 20
case: 11 -20 -10
case: 50 0 10
case: 50 -5 5
case: 50 3 3
case: 50 -100 100
case: 50 10 20
case: 50 -20 -10
case: 100 0 10
case: 100 -5 5
case: 100 3 3
case: 100 -100 100
case: 100 10 20
case: 100 -20 -10
