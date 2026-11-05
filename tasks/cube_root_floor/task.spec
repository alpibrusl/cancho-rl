name: cube_root_floor
ret: int
sig: fn cube_root_floor(n: int) -> [] int
desc: The largest r with r * r * r <= n. Domain: 0 <= n <= 1000000000000000000.
example: cube_root_floor(0) == 0
example: cube_root_floor(9) == 2
example: cube_root_floor(1000) == 10
case: 0
case: 1
case: 2
case: 7
case: 8
case: 9
case: 26
case: 27
case: 28
case: 999
case: 1000
case: 1001
case: 123456789
case: 999999999999999999
case: 1000000000000000000
case: 999999999999999
