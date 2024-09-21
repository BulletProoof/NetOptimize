import params

def get_orders_statistics(orders, instance):
    cost = 0
    upload_cost = 0
    unload_cost = 0
    income = 0
    net_income = 0
    demand = 0
    distance = 0
    max_ready_time = 0
    transport_cost = 0
    last_station_idx = 0
    for order in orders:
        ship_idx = order // instance['customers_num'] + 1
        customer_id = order % instance['customers_num'] + 1
        demand += instance[f'order_{ship_idx}_{customer_id}']['demand']
        destination_idx = instance[f'order_{ship_idx}_{customer_id}']['destination_idx']
        transport_cost += instance['cost_matrix'][last_station_idx][destination_idx]
        unload_cost += instance[f'order_{ship_idx}_{customer_id}']['unload_cost']
        upload_cost = unload_cost
        income += instance[f'order_{ship_idx}_{customer_id}']['income']
        max_ready_time = max(max_ready_time, instance[f'order_{ship_idx}_{customer_id}']['ready_time'])
    net_income = income - upload_cost - unload_cost
    # cost, unload_cost, upload_cost,income,net_income, demand, distance, max_ready_time, transport_cost
    return {
        'cost': cost,
        'unload_cost': unload_cost,
        'upload_cost': upload_cost,
        'income': income,
        'net_income': net_income,
        'demand': demand,
        'distance': distance,
        'max_ready_time': max_ready_time,
        'transport_cost': transport_cost
    }

def val_orders(orders, instance):
    elasped_time = 0
    total_demand = 0
    merge_orders = []
    temp_orders = []
    last_station_idx = 0
    for order in orders:
        ship_idx = order // instance['customers_num'] + 1
        customer_id = order % instance['customers_num'] + 1
        ready_time = instance[f'order_{ship_idx}_{customer_id}']['ready_time']
        elasped_time = max(elasped_time, ready_time)
        total_demand += instance[f'order_{ship_idx}_{customer_id}']['demand']
        destination_idx = instance[f'order_{ship_idx}_{customer_id}']['destination_idx']
        if destination_idx != last_station_idx:
            merge_orders.append(temp_orders)
            temp_orders = []
        else:
            temp_orders.append(order)
    if len(temp_orders) > 0:
        merge_orders.append(temp_orders)

    last_station_idx = 0 # 港口/场站
    updated_load = 0
    stations = ['大连']
    i = 0
    for i in range(len(orders)):
        order = orders[i]
        ship_idx = order // instance['customers_num'] + 1
        customer_id = order % instance['customers_num'] + 1
        info = get_order_info(order, instance)
        ready_time = info['ready_time']
        demand = info['demand']
        destination_idx = info['destination_idx']
        destination = info['destination']

        if destination != stations[-1]:
            stations.append(destination)
        
        # 如果station 在stations中出现了2次，返回false。避免出现折返现象
        if stations.count(destination) > 1:
            return False
        
        updated_load += demand
        if updated_load > instance['max_load']:
            return False

    last_station_idx = 0
    for i in range(len(merge_orders)):  
        temp_orders = merge_orders[i]
        if temp_orders == []:
            elasped_time += (params.stop_time + params.container_unload_time * total_demand)
            continue
        is_same_station = False
        travel_time = 0
        for order in temp_orders:
            info = get_order_info(order, instance)
            destination_idx = info['destination_idx']
            travel_time = instance['distance_matrix'][last_station_idx][destination_idx] / info['avg_speed']
            last_station_idx = info['destination_idx']
            elasped_time += travel_time
            if elasped_time > info['due_time']: # 到达时间大于截止时间
                return False
            elasped_time += info['service_time']
            if is_same_station == False:
                elasped_time += info['stop_time']
            is_same_station = True

    return True

def get_order_info(order, instance):
    res = {}
    ship_idx = order // instance['customers_num'] + 1
    customer_id = order % instance['customers_num'] + 1
    res['ready_time'] = instance[f'order_{ship_idx}_{customer_id}']['ready_time']
    res['demand'] = instance[f'order_{ship_idx}_{customer_id}']['demand']
    res['destination_idx'] = instance[f'order_{ship_idx}_{customer_id}']['destination_idx']
    res['due_time'] = instance[f'order_{ship_idx}_{customer_id}']['due_time']
    res['service_time'] = instance[f'order_{ship_idx}_{customer_id}']['service_time']
    res['travel_time'] = instance['distance_matrix'][0][res['destination_idx']] / instance['avg_speed']
    res['stop_time'] = instance['stop_time']
    res['destination'] = instance[f'order_{ship_idx}_{customer_id}']['destination']
    res['origin'] = instance[f'order_{ship_idx}_{customer_id}']['origin']
    res['income'] = instance[f'order_{ship_idx}_{customer_id}']['income']
    res['unload_cost'] = instance[f'order_{ship_idx}_{customer_id}']['unload_cost']
    res['max_load'] = instance['max_load']
    res['avg_speed'] = instance['avg_speed']
    res['train_cost'] = instance['train_cost']
    res['ships_num'] = instance['ships_num']
    res['customers_num'] = instance['customers_num']
    res['ship_idx'] = ship_idx
    res['customer_idx'] = customer_id
    return res

orders1 = [72, 13, 59, 81, 1, 26]
orders2 = [40, 21, 16]
orders3 = [2, 62, 88, 28, 10, 54]
orders4 = [0, 99, 76, 51]
orders5 = [57, 89, 9, 3, 87, 14]
orders6 = [78, 32, 70]
orders7 = [38, 44, 93, 60]
orders8 = [53, 43, 41, 20, 75]
orders9 = [4, 65, 19]

json_file = "./data/json_customize/" + f'{instance_name}.json'
instance = load_instance(json_file=json_file)
print(val_orders(orders1))

