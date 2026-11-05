name: grade_code
ret: int
kind: branch
sig: fn grade_code(score: int) -> [] int
desc: A grade from a score: 4 for 90 or more, 3 for 80 to 89, 2 for 70 to 79, 1 for 60 to 69, 0 below 60. Domain: 0 <= score <= 100.
example: grade_code(95) == 4
example: grade_code(80) == 3
example: grade_code(12) == 0
case: 0
case: 1
case: 30
case: 58
case: 59
case: 60
case: 61
case: 69
case: 70
case: 71
case: 79
case: 80
case: 81
case: 89
case: 90
case: 91
case: 99
case: 100
