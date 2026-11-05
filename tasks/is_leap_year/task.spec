name: is_leap_year
ret: bool
sig: fn is_leap_year(y: int) -> [] bool
desc: True when y is a leap year in the Gregorian calendar: divisible by 4, except years divisible by 100 that are not divisible by 400. Domain: 1 <= y <= 3000.
example: is_leap_year(2024) == true
example: is_leap_year(1900) == false
example: is_leap_year(2000) == true
case: 1
case: 4
case: 100
case: 400
case: 1700
case: 1800
case: 1900
case: 1996
case: 1999
case: 2000
case: 2001
case: 2004
case: 2023
case: 2024
case: 2100
case: 2400
case: 2999
case: 3000
case: 1990
case: 1991
case: 1992
case: 1993
case: 1994
case: 1995
case: 1996
case: 1997
case: 1998
case: 1999
case: 2000
case: 2001
case: 2002
case: 2003
case: 2004
case: 2005
case: 2006
case: 2007
case: 2008
case: 2009
case: 2010
