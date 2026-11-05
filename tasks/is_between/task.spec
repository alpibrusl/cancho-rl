name: is_between
ret: bool
kind: branch
sig: fn is_between(x: int, lo: int, hi: int) -> [] bool
desc: True when x is at least lo and at most hi (both ends count). Domain: -100 <= lo <= hi <= 100, -100 <= x <= 100.
example: is_between(5, 1, 10) == true
example: is_between(1, 1, 10) == true
example: is_between(11, 1, 10) == false
case: -11 -10 10
case: -11 0 0
case: -11 1 9
case: -11 -5 -1
case: -11 3 4
case: -10 -10 10
case: -10 0 0
case: -10 1 9
case: -10 -5 -1
case: -10 3 4
case: -9 -10 10
case: -9 0 0
case: -9 1 9
case: -9 -5 -1
case: -9 3 4
case: -1 -10 10
case: -1 0 0
case: -1 1 9
case: -1 -5 -1
case: -1 3 4
case: 0 -10 10
case: 0 0 0
case: 0 1 9
case: 0 -5 -1
case: 0 3 4
case: 1 -10 10
case: 1 0 0
case: 1 1 9
case: 1 -5 -1
case: 1 3 4
case: 5 -10 10
case: 5 0 0
case: 5 1 9
case: 5 -5 -1
case: 5 3 4
case: 9 -10 10
case: 9 0 0
case: 9 1 9
case: 9 -5 -1
case: 9 3 4
case: 10 -10 10
case: 10 0 0
case: 10 1 9
case: 10 -5 -1
case: 10 3 4
case: 11 -10 10
case: 11 0 0
case: 11 1 9
case: 11 -5 -1
case: 11 3 4
