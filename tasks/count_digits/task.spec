name: count_digits
ret: int
kind: loop
sig: fn count_digits(n: int) -> [] int
desc: How many decimal digits n has. Domain: 0 <= n <= 9223372036854775807. count_digits(0) is 1.
example: count_digits(0) == 1
example: count_digits(10) == 2
example: count_digits(1000) == 4
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
case: 9
case: 99999
case: 100000
case: 9223372036854775807
case: 1000000000000
