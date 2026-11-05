name: ticket_price
ret: int
kind: branch
sig: fn ticket_price(age: int) -> [] int
desc: The price of a ticket by age: 0 under 5, 10 from 5 to 17, 20 from 18 to 64, 15 for 65 and over. Domain: 0 <= age <= 120.
example: ticket_price(3) == 0
example: ticket_price(30) == 20
example: ticket_price(70) == 15
case: 0
case: 1
case: 4
case: 5
case: 6
case: 12
case: 17
case: 18
case: 19
case: 40
case: 64
case: 65
case: 66
case: 90
case: 120
