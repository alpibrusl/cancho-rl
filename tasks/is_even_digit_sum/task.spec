name: is_even_digit_sum
ret: bool
kind: loop
sig: fn is_even_digit_sum(n: int) -> [] bool
desc: True when the sum of the decimal digits of n is even. Domain: 0 <= n <= 1000000.
example: is_even_digit_sum(11) == true
example: is_even_digit_sum(12) == false
example: is_even_digit_sum(0) == true
case: 0
case: 1
case: 2
case: 9
case: 10
case: 11
case: 12
case: 19
case: 20
case: 99
case: 100
case: 101
case: 123
case: 124
case: 999
case: 1000
case: 54321
