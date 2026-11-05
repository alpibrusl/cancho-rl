name: round_to_ten
ret: int
kind: branch
sig: fn round_to_ten(n: int) -> [] int
desc: n rounded to the nearest multiple of 10, halves rounding up (15 becomes 20). Domain: 0 <= n <= 100000.
example: round_to_ten(14) == 10
example: round_to_ten(15) == 20
example: round_to_ten(0) == 0
case: 0
case: 1
case: 4
case: 5
case: 6
case: 9
case: 10
case: 11
case: 14
case: 15
case: 16
case: 24
case: 25
case: 26
case: 99
case: 100
case: 995
case: 1000
case: 1004
case: 1005
