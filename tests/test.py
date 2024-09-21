
def tranform(order_id):
    ship_idx = order_id // 20 + 1
    customer_id = order_id % 20 + 1
    print(f"ship_idx: {ship_idx}, customer_id: {customer_id}")

[72, 13, 59, 81, 1, 26]
for order_id in [72, 13, 59, 81, 1, 26]:
    tranform(order_id)

# ship_idx: 4, customer_id: 13 demand:25
# ship_idx: 1, customer_id: 14 demand:12
# ship_idx: 3, customer_id: 20 demand:20
# ship_idx: 5, customer_id: 2 demand:18
# ship_idx: 1, customer_id: 2 demand:20
# ship_idx: 2, customer_id: 7 demand:6

# orders1 = [72, 13, 59, 81, 1, 26]
# orders2 = [40, 21, 16]
# orders3 = [2, 62, 88, 28, 10, 54]
# orders4 = [0, 99, 76, 51]
# orders5 = [57, 89, 9, 3, 87, 14]
# orders6 = [78, 32, 70]
# orders7 = [38, 44, 93, 60]
# orders8 = [53, 43, 41, 20, 75]
# orders9 = [4, 65, 19]

[81, 9, 87, 93, 72]
5,2
1,10
5,8
5,14
4,13
demand:86
[3, 4, 20, 60, 38]
[41, 40, 1, 53, 75, 14, 76]