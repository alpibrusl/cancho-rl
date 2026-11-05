name: shipping_cost
ret: int
kind: branch
sig: fn shipping_cost(g: int) -> [] int
desc: Shipping cost for a parcel of g grams: 5 up to and including 500 g, 8 up to and including 2000 g, 12 up to and including 5000 g, otherwise 20. Domain: 1 <= g <= 10000.
example: shipping_cost(500) == 5
example: shipping_cost(2000) == 8
example: shipping_cost(6000) == 20
case: 1
case: 100
case: 499
case: 500
case: 501
case: 1000
case: 1999
case: 2000
case: 2001
case: 4999
case: 5000
case: 5001
case: 8000
case: 10000
