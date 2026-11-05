name: hours_between
ret: int
kind: branch
sig: fn hours_between(a: int, b: int) -> [] int
desc: How many hours pass going forward on a 24-hour clock from hour a to hour b (wrapping past midnight); 0 when they are the same hour. Domain: 0 <= a, b <= 23.
example: hours_between(22, 3) == 5
example: hours_between(3, 22) == 19
example: hours_between(7, 7) == 0
case: 0 0
case: 0 1
case: 0 3
case: 0 7
case: 0 12
case: 0 18
case: 0 23
case: 1 0
case: 1 1
case: 1 3
case: 1 7
case: 1 12
case: 1 18
case: 1 23
case: 5 0
case: 5 1
case: 5 3
case: 5 7
case: 5 12
case: 5 18
case: 5 23
case: 11 0
case: 11 1
case: 11 3
case: 11 7
case: 11 12
case: 11 18
case: 11 23
case: 12 0
case: 12 1
case: 12 3
case: 12 7
case: 12 12
case: 12 18
case: 12 23
case: 13 0
case: 13 1
case: 13 3
case: 13 7
case: 13 12
case: 13 18
case: 13 23
case: 22 0
case: 22 1
case: 22 3
case: 22 7
case: 22 12
case: 22 18
case: 22 23
case: 23 0
case: 23 1
case: 23 3
case: 23 7
case: 23 12
case: 23 18
case: 23 23
