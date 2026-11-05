name: sign_product
ret: int
kind: branch
sig: fn sign_product(a: int, b: int) -> [] int
desc: The sign of a * b: 1 when it is positive, -1 when it is negative, 0 when it is zero. Do not rely on the product being small. Domain: -1000 <= a, b <= 1000.
example: sign_product(-3, 4) == -1
example: sign_product(-3, -4) == 1
example: sign_product(0, 9) == 0
case: -1000 -1000
case: -1000 -3
case: -1000 -1
case: -1000 0
case: -1000 1
case: -1000 3
case: -1000 1000
case: -7 -1000
case: -7 -3
case: -7 -1
case: -7 0
case: -7 1
case: -7 3
case: -7 1000
case: -1 -1000
case: -1 -3
case: -1 -1
case: -1 0
case: -1 1
case: -1 3
case: -1 1000
case: 0 -1000
case: 0 -3
case: 0 -1
case: 0 0
case: 0 1
case: 0 3
case: 0 1000
case: 1 -1000
case: 1 -3
case: 1 -1
case: 1 0
case: 1 1
case: 1 3
case: 1 1000
case: 7 -1000
case: 7 -3
case: 7 -1
case: 7 0
case: 7 1
case: 7 3
case: 7 1000
case: 1000 -1000
case: 1000 -3
case: 1000 -1
case: 1000 0
case: 1000 1
case: 1000 3
case: 1000 1000
