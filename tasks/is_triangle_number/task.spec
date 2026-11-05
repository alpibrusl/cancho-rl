name: is_triangle_number
ret: bool
kind: loop
sig: fn is_triangle_number(n: int) -> [] bool
desc: True when n is a triangle number, that is 1 + 2 + ... + k for some k >= 0 (0 counts, with k = 0). Domain: 0 <= n <= 1000000.
example: is_triangle_number(10) == true
example: is_triangle_number(11) == false
example: is_triangle_number(0) == true
case: 0
case: 1
case: 2
case: 3
case: 4
case: 5
case: 6
case: 7
case: 10
case: 15
case: 20
case: 21
case: 22
case: 28
case: 36
case: 45
case: 55
case: 66
case: 100
case: 105
case: 120
case: 5050
case: 5051
case: 500500
case: 500501
case: 1000000
case: 1000001
case: 449999
