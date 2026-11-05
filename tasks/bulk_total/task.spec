name: bulk_total
ret: int
kind: branch
sig: fn bulk_total(price: int, qty: int) -> [] int
desc: The total cost of qty items at price each. An order of 10 or more items gets a discount of 20% of the total, rounded down to a whole number (a total of 77 gets a discount of 15 and costs 62). Domain: 1 <= price <= 1000, 0 <= qty <= 100.
example: bulk_total(5, 3) == 15
example: bulk_total(5, 10) == 40
example: bulk_total(7, 11) == 62
case: 1 0
case: 1 1
case: 1 2
case: 1 9
case: 1 10
case: 1 11
case: 1 25
case: 5 0
case: 5 1
case: 5 2
case: 5 9
case: 5 10
case: 5 11
case: 5 25
case: 7 0
case: 7 1
case: 7 2
case: 7 9
case: 7 10
case: 7 11
case: 7 25
case: 100 0
case: 100 1
case: 100 2
case: 100 9
case: 100 10
case: 100 11
case: 100 25
case: 999 0
case: 999 1
case: 999 2
case: 999 9
case: 999 10
case: 999 11
case: 999 25
