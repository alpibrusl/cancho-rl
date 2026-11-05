name: fib_even_sum
ret: int
kind: loop
sig: fn fib_even_sum(limit: int) -> [] int
desc: The sum of the even numbers in the sequence 1, 2, 3, 5, 8, 13, ... (each term the sum of the two before) that are at most limit. Domain: 0 <= limit <= 4000000.
example: fib_even_sum(10) == 10
example: fib_even_sum(1) == 0
example: fib_even_sum(100) == 44
case: 0
case: 1
case: 2
case: 3
case: 7
case: 8
case: 9
case: 33
case: 34
case: 35
case: 89
case: 100
case: 144
case: 233
case: 1000
case: 4000000
