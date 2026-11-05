name: digital_root
ret: int
kind: loop
sig: fn digital_root(n: int) -> [] int
desc: Add the decimal digits of n, repeat on the result until a single digit is left, and return it. digital_root(0) is 0 and digital_root(493193) is 2 (4+9+3+1+9+3 = 29, 2+9 = 11, 1+1 = 2). Domain: 0 <= n <= 9223372036854775807.
example: digital_root(0) == 0
example: digital_root(10) == 1
example: digital_root(1000) == 1
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
case: 493193
case: 9223372036854775807
case: 999999999999
case: 18
case: 19
