name: reverse_number
ret: int
kind: loop
sig: fn reverse_number(n: int) -> [] int
desc: The digits of n in reverse order, as a number; leading zeros of the result vanish. reverse_number(120) is 21. Domain: 0 <= n <= 1000000.
example: reverse_number(123) == 321
example: reverse_number(100) == 1
example: reverse_number(7) == 7
case: 0
case: 1
case: 9
case: 10
case: 12
case: 100
case: 120
case: 121
case: 123
case: 1000
case: 1234
case: 90909
case: 120300
case: 99999
case: 1000000
