name: alt_digit_sum
ret: int
kind: loop
sig: fn alt_digit_sum(n: int) -> [] int
desc: Start from the last digit of n and move left, adding the first digit, subtracting the second, adding the third, and so on. alt_digit_sum(1234) is 4 - 3 + 2 - 1 = 2. Domain: 0 <= n <= 1000000000.
example: alt_digit_sum(1234) == 2
example: alt_digit_sum(5) == 5
example: alt_digit_sum(90) == 9
case: 0
case: 1
case: 5
case: 9
case: 10
case: 12
case: 21
case: 90
case: 99
case: 123
case: 1234
case: 4321
case: 10101
case: 99999
case: 123456
