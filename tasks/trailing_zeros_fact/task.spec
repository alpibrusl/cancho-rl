name: trailing_zeros_fact
ret: int
kind: loop
sig: fn trailing_zeros_fact(n: int) -> [] int
desc: How many zeros the decimal form of n! (n factorial) ends with. Do not compute n!. Domain: 0 <= n <= 1000000000.
example: trailing_zeros_fact(5) == 1
example: trailing_zeros_fact(10) == 2
example: trailing_zeros_fact(4) == 0
case: 0
case: 1
case: 4
case: 5
case: 9
case: 10
case: 24
case: 25
case: 26
case: 49
case: 50
case: 99
case: 100
case: 124
case: 125
case: 126
case: 625
case: 1000
case: 3125
case: 100000
case: 1000000
case: 1000000000
