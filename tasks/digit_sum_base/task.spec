name: digit_sum_base
ret: int
sig: fn digit_sum_base(n: int, b: int) -> [] int
desc: The sum of the digits of n written in base b. Domain: 0 <= n <= 1000000000000 and 2 <= b <= 10. digit_sum_base(255, 2) is 8.
example: digit_sum_base(0, 2) == 0
example: digit_sum_base(8, 2) == 1
example: digit_sum_base(1000, 2) == 6
case: 0 2
case: 0 3
case: 0 7
case: 0 8
case: 0 9
case: 0 10
case: 1 2
case: 1 3
case: 1 7
case: 1 8
case: 1 9
case: 1 10
case: 7 2
case: 7 3
case: 7 7
case: 7 8
case: 7 9
case: 7 10
case: 8 2
case: 8 3
case: 8 7
case: 8 8
case: 8 9
case: 8 10
case: 255 2
case: 255 3
case: 255 7
case: 255 8
case: 255 9
case: 255 10
case: 256 2
case: 256 3
case: 256 7
case: 256 8
case: 256 9
case: 256 10
case: 1000 2
case: 1000 3
case: 1000 7
case: 1000 8
case: 1000 9
case: 1000 10
case: 123456789 2
case: 123456789 3
case: 123456789 7
case: 123456789 8
case: 123456789 9
case: 123456789 10
case: 1000000000000 2
case: 1000000000000 3
case: 1000000000000 7
case: 1000000000000 8
case: 1000000000000 9
case: 1000000000000 10
