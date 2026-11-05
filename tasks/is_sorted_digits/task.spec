name: is_sorted_digits
ret: bool
kind: loop
sig: fn is_sorted_digits(n: int) -> [] bool
desc: True when the decimal digits of n never decrease from left to right (equal neighbours are fine): 1139 is, 1329 is not. Single-digit numbers are. Domain: 0 <= n <= 1000000000.
example: is_sorted_digits(1139) == true
example: is_sorted_digits(1329) == false
example: is_sorted_digits(7) == true
case: 0
case: 5
case: 9
case: 10
case: 11
case: 12
case: 21
case: 99
case: 100
case: 111
case: 112
case: 121
case: 123
case: 321
case: 1139
case: 1329
case: 2999
case: 3000
case: 12345
case: 12354
case: 99999
case: 123456789
case: 987654321
