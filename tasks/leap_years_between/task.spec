name: leap_years_between
ret: int
sig: fn leap_years_between(a: int, b: int) -> [] int
desc: How many years from a to b, both included, are leap years in the Gregorian calendar (divisible by 4, except years divisible by 100 that are not divisible by 400). Domain: 1 <= a <= b <= 3000.
example: leap_years_between(2000, 2004) == 2
example: leap_years_between(1900, 1900) == 0
example: leap_years_between(1, 100) == 24
case: 1 1
case: 1 4
case: 1 100
case: 1 400
case: 1 1900
case: 1 2000
case: 1 2024
case: 1 2100
case: 1 3000
case: 4 4
case: 4 100
case: 4 400
case: 4 1900
case: 4 2000
case: 4 2024
case: 4 2100
case: 4 3000
case: 100 100
case: 100 400
case: 100 1900
case: 100 2000
case: 100 2024
case: 100 2100
case: 100 3000
case: 399 400
case: 399 1900
case: 399 2000
case: 399 2024
case: 399 2100
case: 399 3000
case: 400 400
case: 400 1900
case: 400 2000
case: 400 2024
case: 400 2100
case: 400 3000
case: 1600 1900
case: 1600 2000
case: 1600 2024
case: 1600 2100
case: 1600 3000
case: 1896 1900
case: 1896 2000
case: 1896 2024
case: 1896 2100
case: 1896 3000
case: 1900 1900
case: 1900 2000
case: 1900 2024
case: 1900 2100
case: 1900 3000
case: 1999 2000
case: 1999 2024
case: 1999 2100
case: 1999 3000
case: 2000 2000
case: 2000 2024
case: 2000 2100
case: 2000 3000
case: 2001 2024
case: 2001 2100
case: 2001 3000
