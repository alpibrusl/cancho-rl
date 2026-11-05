name: round_up_to
ret: int
kind: branch
sig: fn round_up_to(x: int, m: int) -> [] int
desc: The smallest multiple of m that is at least x. Domain: 0 <= x <= 1000000000000 and 1 <= m <= 1000000. round_up_to(0, 8) is 0 and round_up_to(16, 8) is 16.
example: round_up_to(0, 1) == 0
example: round_up_to(8, 4096) == 4096
example: round_up_to(17, 8) == 24
case: 0 1
case: 0 2
case: 0 8
case: 0 10
case: 0 4096
case: 0 1000000
case: 1 1
case: 1 2
case: 1 8
case: 1 10
case: 1 4096
case: 1 1000000
case: 7 1
case: 7 2
case: 7 8
case: 7 10
case: 7 4096
case: 7 1000000
case: 8 1
case: 8 2
case: 8 8
case: 8 10
case: 8 4096
case: 8 1000000
case: 9 1
case: 9 2
case: 9 8
case: 9 10
case: 9 4096
case: 9 1000000
case: 15 1
case: 15 2
case: 15 8
case: 15 10
case: 15 4096
case: 15 1000000
case: 16 1
case: 16 2
case: 16 8
case: 16 10
case: 16 4096
case: 16 1000000
case: 17 1
case: 17 2
case: 17 8
case: 17 10
case: 17 4096
case: 17 1000000
case: 1000 1
case: 1000 2
case: 1000 8
case: 1000 10
case: 1000 4096
case: 1000 1000000
case: 999999999999 1
case: 999999999999 2
case: 999999999999 8
case: 999999999999 10
case: 999999999999 4096
case: 999999999999 1000000
case: 1000000000000 1
case: 1000000000000 2
case: 1000000000000 8
case: 1000000000000 10
case: 1000000000000 4096
case: 1000000000000 1000000
