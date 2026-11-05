name: is_palindrome_number
ret: bool
kind: loop
sig: fn is_palindrome_number(n: int) -> [] bool
desc: True when the decimal digits of n read the same forwards and backwards. Domain: 0 <= n <= 1000000.
example: is_palindrome_number(121) == true
example: is_palindrome_number(123) == false
example: is_palindrome_number(5) == true
case: 0
case: 1
case: 9
case: 10
case: 11
case: 12
case: 22
case: 100
case: 101
case: 121
case: 123
case: 1001
case: 1221
case: 1231
case: 12321
case: 12345
case: 100001
case: 123321
