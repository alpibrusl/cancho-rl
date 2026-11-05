name: bmi_class
ret: int
kind: branch
sig: fn bmi_class(kg: int, cm: int) -> [] int
desc: Compute bmi = kg * 10000 / (cm * cm) with whole-number division (dropping any remainder). Return 0 when bmi < 19, 1 when 19 <= bmi < 25, else 2. Domain: 30 <= kg <= 200, 120 <= cm <= 220.
example: bmi_class(50, 170) == 0
example: bmi_class(70, 175) == 1
example: bmi_class(90, 170) == 2
case: 40 150
case: 40 160
case: 40 170
case: 40 180
case: 40 190
case: 50 150
case: 50 160
case: 50 170
case: 50 180
case: 50 190
case: 55 150
case: 55 160
case: 55 170
case: 55 180
case: 55 190
case: 60 150
case: 60 160
case: 60 170
case: 60 180
case: 60 190
case: 65 150
case: 65 160
case: 65 170
case: 65 180
case: 65 190
case: 70 150
case: 70 160
case: 70 170
case: 70 180
case: 70 190
case: 80 150
case: 80 160
case: 80 170
case: 80 180
case: 80 190
case: 90 150
case: 90 160
case: 90 170
case: 90 180
case: 90 190
case: 100 150
case: 100 160
case: 100 170
case: 100 180
case: 100 190
case: 120 150
case: 120 160
case: 120 170
case: 120 180
case: 120 190
