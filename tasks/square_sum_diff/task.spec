name: square_sum_diff
ret: int
kind: loop
sig: fn square_sum_diff(n: int) -> [] int
desc: The square of the sum of 1..n minus the sum of the squares of 1..n. For n = 2: (1 + 2)^2 - (1 + 4) = 4. Domain: 0 <= n <= 100.
example: square_sum_diff(2) == 4
example: square_sum_diff(10) == 2640
example: square_sum_diff(0) == 0
case: 0
case: 1
case: 2
case: 3
case: 4
case: 5
case: 6
case: 7
case: 10
case: 12
case: 20
case: 33
case: 50
case: 64
case: 99
case: 100
