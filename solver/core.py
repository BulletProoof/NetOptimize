# -*- coding: utf-8 -*-


import os
import io
import random
import re
from csv import DictWriter
from deap import base, creator, tools
from . import BASE_DIR
from .utils import make_dirs_for_file, exist, load_instance, merge_rules
from decorator import calculate_time

class Train:
    def __init__(self, station_list, customer_list, ship_list, unload_demand_list, remaining_demand_list, arrival_time_list):
        self.station_list = station_list
        self.customer_list = customer_list
        self.ship_list = ship_list
        self.unload_demand_list = unload_demand_list
        self.remaining_demand_list = remaining_demand_list
        self.arrival_time_list = arrival_time_list

def val_orders(orders, instance):
    elasped_time = 0
    for order in orders:
        ship_idx = order // instance['customers_num'] + 1
        customer_id = order % instance['customers_num'] + 1
        ready_time = instance[f'order_{ship_idx}_{customer_id}']['ready_time']
        elasped_time = max(elasped_time, ready_time)

    last_station_idx = 0 # 港口/场站
    updated_load = 0
    stations = ['大连']
    for order in orders:
        ship_idx = order // instance['customers_num'] + 1
        customer_id = order % instance['customers_num'] + 1
        ready_time = instance[f'order_{ship_idx}_{customer_id}']['ready_time']
        demand = instance[f'order_{ship_idx}_{customer_id}']['demand']
        destination_idx = instance[f'order_{ship_idx}_{customer_id}']['destination_idx']
        travel_time = instance['distance_matrix'][last_station_idx][destination_idx] / instance['avg_speed']
        stop_time = instance['stop_time']
        service_time = instance[f'order_{ship_idx}_{customer_id}']['service_time']
        destination = instance[f'order_{ship_idx}_{customer_id}']['destination']
        if destination != stations[-1]:
            stations.append(destination)
        
        # 如果station 在stations中出现了2次，返回false
        if stations.count(destination) > 1:
            return False
        
        updated_load += demand
        if updated_load > instance['max_load']:
            return False
        elasped_time += travel_time
        if elasped_time > instance[f'order_{ship_idx}_{customer_id}']['due_time']: # 到达时间大于截止时间
            return False
        elasped_time = elasped_time + stop_time + service_time

        last_station_idx = destination_idx

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

    res['ship_idx'] = ship_idx
    res['customer_idx'] = customer_id
    return res

    

def ind2route(individual, instance):
    route = []
    max_load = instance['max_load']
    # Initialize a sub-route
    sub_route = []
    # individual is a list of order ids:[0,1,2,3,4,9,10,13,14,16,19,20,21,26,28,32,38,40,41,43,44,51,53,54,57,59,60,62,65,70,72,75,76,78,81,87,88,89,93,99]
    i = 0
    j = 0
    while i < len(individual):
        j = i
        while j < len(individual):
            if val_orders(individual[i:j+1], instance):
                sub_route.append(individual[j])
                j += 1
            else:
                if sub_route != []:
                    route.append(sub_route)
                    sub_route = []
                i = j
                break
        if j == len(individual):
            break
    if sub_route != []:
        route.append(sub_route)
    return route


def print_route(route, instance ,merge=False):
    '''gavrptw.core.print_route(route, merge=False)'''
    sub_route_count = 0
    acc_cost = 0
    acc_income = 0
    for sub_route in route:
        sub_route_count += 1
        print(f'[班列 {sub_route_count}]:\n')
        print('   orders:',sub_route, end='\n')

        
        orders = [] # 记录每个班列含哪些订单
        stations = ['大连'] # 记录每个班列包含哪些站点
        station_ids = [0] # 记录每个班列包含哪些站点id
        sub_orders = []   # 记录每个站点包含哪些订单
        customers = [] # 记录每个站点包含哪些顾客
        sub_customers = [] #记录一个站点包含的顾客
        ships = [] # 记录每个站点包含哪些船舶
        sub_ships = [] #记录一个站点包含的船舶

        net_income = -20000

        total_demand = 0
        last_destination = 0
        max_ready_time = 0
        unload_cost = 0
        income = 0
        for order_id in sub_route:
            info = get_order_info(order_id, instance)
            net_income = net_income + info['income'] - info['unload_cost']
            total_demand += info['demand']
            max_ready_time = max(max_ready_time, info['ready_time'])
            unload_cost += info['unload_cost']
            income += info['income']
            # station路径
            if info['destination'] != stations[-1]:
                stations.append(info['destination'])
                station_ids.append(info['destination_idx'])
            if info['destination'] != last_destination:
                orders.append(sub_orders)
                sub_orders = []
            sub_orders.append(order_id)
        if sub_orders != []:
            orders.append(sub_orders)

        for sub_orders in orders:
            if sub_orders == []:
                customers.append([])
                ships.append([])
            else:
                for order_id in sub_orders:
                    info = get_order_info(order_id, instance)
                    sub_customers.append(info['customer_idx'])
                    sub_ships.append(info['ship_idx'])
                customers.append(sub_customers)
                ships.append(sub_ships)
                sub_customers = []
                sub_ships = []

        # 计算剩余列表、装卸列表、到达时间列表
        unload_list = []
        remain_list = []
        arrival_time_list = []
        remain_demand = total_demand
        for sub_orders in orders:
            sub_total_demand = 0
            elasped_time = max_ready_time
            for order_id in sub_orders:
                info = get_order_info(order_id, instance)
                sub_total_demand += info['demand']
            unload_list.append(sub_total_demand)
            remain_list.append(remain_demand - sub_total_demand)
            remain_demand -= sub_total_demand
            arrival_time_list.append(elasped_time)

        # 计算班列运输费用
        transport_cost = 0
        for i,j in zip(station_ids[0:],station_ids[1:]):
            cost = instance['cost_matrix'][i][j]
            transport_cost += cost
            net_income = net_income - cost
        acc_cost += transport_cost
        acc_cost += unload_cost
        acc_cost += 20000
        acc_income += income
        print('   stations:',stations, end='\n')
        print('   station_ids:',station_ids, end='\n')
        print('   customers:',customers, end='\n')
        print('   ships:',ships, end='\n')
        print('   unload:',unload_list, end='\n')
        print('   remain:',remain_list, end='\n')
        print('   income:',income, end='\n')
        print('   unload_cost:',unload_cost, end='\n')
        print('   transport_cost:',transport_cost, end='\n')
        print('   net_income:',net_income, end='\n')
    print('acc_cost:',acc_cost, end='\n')
    print('acc_income:',acc_income, end='\n')
    print('total_income:',acc_income - acc_cost, end='\n')
        
def eval_vrptw(individual, instance, unit_cost=0.0, init_cost=0, wait_cost=0, delay_cost=0):
    '''gavrptw.core.eval_vrptw(individual, instance, unit_cost=1.0, init_cost=0, wait_cost=0,
        delay_cost=0)'''
    # print('----------eval_vrptw----------')

    total_cost = 0
    route = ind2route(individual, instance)

    for sub_route in route:
        sub_route_distance = 0
        elapsed_time = 0
        last_station_idx = 0
        sub_route_cost = 0
        sub_route_load = 0

        # 计算子路线订单最大到港时间
        for order_id in sub_route:
            ship_idx = order_id // instance['customers_num'] + 1
            customer_id = order_id % instance['customers_num'] + 1
            elapsed_time =  max(instance[f'order_{ship_idx}_{customer_id}']['ready_time'], elapsed_time)

        for order_id in sub_route:
            ship_idx = order_id // instance['customers_num'] + 1
            customer_id = order_id % instance['customers_num'] + 1
            destination_idx = instance[f'order_{ship_idx}_{customer_id}']['destination_idx']

            # Calculate section distance
            distance = instance['distance_matrix'][last_station_idx][destination_idx]
            travel_time = distance / instance['avg_speed']
            cost = instance['cost_matrix'][last_station_idx][destination_idx] + instance[f'order_{ship_idx}_{customer_id}']['unload_cost']
            sub_route_load  = sub_route_load + instance[f'order_{ship_idx}_{customer_id}']['demand']

            # Update sub-route distance
            sub_route_distance += distance
            sub_route_cost += cost

            arrival_time = elapsed_time + travel_time
            elapsed_time = arrival_time + instance[f'order_{ship_idx}_{customer_id}']['service_time'] + instance['stop_time']

            # 顾客时间窗约束
            if arrival_time > instance[f'order_{ship_idx}_{customer_id}']['due_time']:
                return (0.0, )
            
            # 最大装载量限制
            if sub_route_load > instance['max_load']:
                return (0.0, )
        

            last_station_idx = destination_idx

        total_cost += sub_route_cost

    total_cost += len(route) * unit_cost
    fitness = 1.0 / total_cost
    return (fitness, )


def cx_partially_matched(ind1, ind2):
    '''gavrptw.core.cx_partially_matched(ind1, ind2)'''
    cxpoint1, cxpoint2 = sorted(random.sample(range(min(len(ind1), len(ind2))), 2))
    part1 = ind2[cxpoint1:cxpoint2+1]
    part2 = ind1[cxpoint1:cxpoint2+1]
    rule1to2 = list(zip(part1, part2))
    is_fully_merged = False
    while not is_fully_merged:
        rule1to2, is_fully_merged = merge_rules(rules=rule1to2)
    rule2to1 = {rule[1]: rule[0] for rule in rule1to2}
    rule1to2 = dict(rule1to2)
    ind1 = [gene if gene not in part2 else rule2to1[gene] for gene in ind1[:cxpoint1]] + part2 + \
        [gene if gene not in part2 else rule2to1[gene] for gene in ind1[cxpoint2+1:]]
    ind2 = [gene if gene not in part1 else rule1to2[gene] for gene in ind2[:cxpoint1]] + part1 + \
        [gene if gene not in part1 else rule1to2[gene] for gene in ind2[cxpoint2+1:]]
    return ind1, ind2


def mut_inverse_indexes(individual):
    '''gavrptw.core.mut_inverse_indexes(individual)'''
    start, stop = sorted(random.sample(range(len(individual)), 2))
    temp = individual[start:stop+1]
    temp.reverse()
    individual[start:stop+1] = temp
    return (individual, )

def get_generator(instance):
    custom_list = []
    for k in instance.keys():
        if k.startswith('order_'):
            # 从"order_1_1"提取数字
            numbers = re.findall(r'\d+', k)
            numbers = [int(num) for num in numbers]
            ship_id = numbers[0]
            customer_id = numbers[1]
            # print("numbers:", numbers)
            custom_list.append((ship_id-1)*instance['customers_num'] + customer_id-1)
    print("customer_list:", custom_list)
    return custom_list

@calculate_time
def run_gavrptw(instance_name, unit_cost, init_cost, wait_cost, delay_cost, ind_size, pop_size, \
    cx_pb, mut_pb, n_gen, export_csv=False, customize_data=False):
    '''gavrptw.core.run_gavrptw(instance_name, unit_cost, init_cost, wait_cost, delay_cost,
        ind_size, pop_size, cx_pb, mut_pb, n_gen, export_csv=False, customize_data=False)'''
    # if customize_data:
    #     json_data_dir = os.path.join(BASE_DIR, 'data', 'json_customize')
    # else:
    #     json_data_dir = os.path.join(BASE_DIR, 'data', 'json')

    # json_data_dir = os.path.join(BASE_DIR, 'data', 'json_customize')
    # json_file = os.path.join(json_data_dir, f'{instance_name}.json')
    json_file = "./data/json_customize/" + f'{instance_name}.json'
    instance = load_instance(json_file=json_file)
    if instance is None:
        return
    print(f'Instance {instance_name} loaded')
    # print("instance:", instance)

    creator.create('FitnessMax', base.Fitness, weights=(1.0, ))
    creator.create('Individual', list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()


    # Attribute generator
    # toolbox.register('indexes', random.sample, range(1, ind_size + 1), ind_size)
    # 注册 'indexes' 函数，使用自定义列表
    custom_list = get_generator(instance)
    toolbox.register('indexes', random.sample, custom_list, len(custom_list))

    # Structure initializers
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.indexes)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    # Operator registering
    toolbox.register('evaluate', eval_vrptw, instance=instance, unit_cost=unit_cost, \
        init_cost=init_cost, wait_cost=wait_cost, delay_cost=delay_cost)
    toolbox.register('select', tools.selRoulette)
    toolbox.register('mate', cx_partially_matched)
    toolbox.register('mutate', mut_inverse_indexes)
    pop = toolbox.population(n=pop_size)
    # Results holders for exporting results to CSV file
    csv_data = []
    print('Start of evolution')
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    print(f'  Evaluated {len(pop)} individuals')
    # Begin the evolution
    for gen in range(n_gen):
        print(f'-- Generation {gen} --')
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_pb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < mut_pb:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        print(f'  Evaluated {len(invalid_ind)} individuals')
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum([x**2 for x in fits])
        std = abs(sum2 / length - mean**2)**0.5
        print(f'  Min {min(fits)}')
        print(f'  Max {max(fits)}')
        print(f'  Avg {mean}')
        print(f'  Std {std}')
        # Write data to holders for exporting results to CSV file
        if export_csv:
            csv_row = {
                'generation': gen,
                'evaluated_individuals': len(invalid_ind),
                'min_fitness': min(fits),
                'max_fitness': max(fits),
                'avg_fitness': mean,
                'std_fitness': std,
            }
            csv_data.append(csv_row)
    print('-- End of (successful) evolution --')
    best_ind = tools.selBest(pop, 1)[0]
    print(f'Best individual: {best_ind}')
    print(f'Fitness: {best_ind.fitness.values[0]}')
    print_route(ind2route(best_ind, instance),instance)
    print(f'Total cost: {1 / best_ind.fitness.values[0]}')
    if export_csv:
        csv_file_name = f'{instance_name}_uC{unit_cost}_iC{init_cost}_wC{wait_cost}' \
            f'_dC{delay_cost}_iS{ind_size}_pS{pop_size}_cP{cx_pb}_mP{mut_pb}_nG{n_gen}.csv'
        # csv_file = os.path.join(BASE_DIR, 'results', csv_file_name)
        csv_file = "../results/" + csv_file_name
        print(f'Write to file: {csv_file}')
        make_dirs_for_file(path=csv_file)
        if not exist(path=csv_file, overwrite=True):
            with io.open(csv_file, 'wt', encoding='utf-8', newline='') as file_object:
                fieldnames = [
                    'generation',
                    'evaluated_individuals',
                    'min_fitness',
                    'max_fitness',
                    'avg_fitness',
                    'std_fitness',
                ]
                writer = DictWriter(file_object, fieldnames=fieldnames, dialect='excel')
                writer.writeheader()
                for csv_row in csv_data:
                    writer.writerow(csv_row)
