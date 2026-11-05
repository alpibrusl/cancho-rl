name: clamp_percent
ret: int
kind: branch
sig: fn clamp_percent(n: int) -> [] int
desc: n limited to the range 0 to 100: 0 for anything below, 100 for anything above. Domain: -1000 <= n <= 1000.
example: clamp_percent(50) == 50
example: clamp_percent(-5) == 0
example: clamp_percent(130) == 100
case: -1000
case: -1
case: 0
case: 1
case: 50
case: 99
case: 100
case: 101
case: 250
case: 1000
