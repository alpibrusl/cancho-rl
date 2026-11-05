name: sum_evens_to
ret: int
kind: loop
sig: fn sum_evens_to(n: int) -> [] int
desc: The sum of the even numbers from 2 up to n (n included when it is even). sum_evens_to(1) is 0. Domain: 0 <= n <= 10000.
example: sum_evens_to(10) == 30
example: sum_evens_to(1) == 0
example: sum_evens_to(0) == 0
case: 0
case: 1
case: 2
case: 3
case: 4
case: 5
case: 6
case: 9
case: 10
case: 11
case: 50
case: 99
case: 100
case: 101
case: 1000
case: 9999
case: 10000
