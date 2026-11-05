name: is_harshad
ret: bool
kind: loop
sig: fn is_harshad(n: int) -> [] bool
desc: True when n is divisible by the sum of its own decimal digits. is_harshad(18) is true (1 + 8 = 9 divides 18). Domain: 1 <= n <= 100000.
example: is_harshad(18) == true
example: is_harshad(19) == false
example: is_harshad(1) == true
case: 1
case: 2
case: 9
case: 10
case: 11
case: 12
case: 18
case: 19
case: 20
case: 21
case: 24
case: 25
case: 100
case: 101
case: 111
case: 112
case: 1000
case: 1729
