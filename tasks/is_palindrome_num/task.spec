name: is_palindrome_num
ret: bool
kind: loop
sig: fn is_palindrome_num(n: int) -> [] bool
desc: True when the decimal digits of n read the same forwards and backwards. Domain: 0 <= n <= 1000000000. 0 is a palindrome; 10 is not.
example: is_palindrome_num(0) == true
example: is_palindrome_num(16) == false
example: is_palindrome_num(999999) == true
case: 0
case: 1
case: 2
case: 3
case: 4
case: 5
case: 7
case: 10
case: 16
case: 17
case: 99
case: 100
case: 101
case: 255
case: 1000
case: 65535
case: 999999
case: 1000000000
case: 11
case: 121
case: 1221
case: 12321
case: 12345
case: 999999999
