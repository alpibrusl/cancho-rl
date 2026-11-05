name: day_of_year
ret: int
kind: loop
sig: fn day_of_year(m: int, d: int) -> [] int
desc: The position of the date (month m, day d) in a year that is not a leap year: January 1 is day 1, December 31 is day 365. Domain: 1 <= m <= 12 and d is a valid day of month m (February has 28 days).
example: day_of_year(1, 1) == 1
example: day_of_year(3, 1) == 60
example: day_of_year(12, 31) == 365
case: 1 1
case: 1 15
case: 1 2
case: 1 28
case: 1 31
case: 10 1
case: 10 15
case: 10 2
case: 10 28
case: 10 31
case: 11 1
case: 11 15
case: 11 2
case: 11 28
case: 11 30
case: 12 1
case: 12 15
case: 12 2
case: 12 28
case: 12 31
case: 2 1
case: 2 15
case: 2 2
case: 2 28
case: 3 1
case: 3 15
case: 3 2
case: 3 28
case: 3 31
case: 4 1
case: 4 15
case: 4 2
case: 4 28
case: 4 30
case: 5 1
case: 5 15
case: 5 2
case: 5 28
case: 5 31
case: 6 1
case: 6 15
case: 6 2
case: 6 28
case: 6 30
case: 7 1
case: 7 15
case: 7 2
case: 7 28
case: 7 31
case: 8 1
case: 8 15
case: 8 2
case: 8 28
case: 8 31
case: 9 1
case: 9 15
case: 9 2
case: 9 28
case: 9 30
