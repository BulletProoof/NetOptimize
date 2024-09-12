# -*- coding: utf-8 -*-

'''gavrptw/core.py'''

import os
import io
import random
from csv import DictWriter
from deap import base, creator, tools
from . import BASE_DIR
from .utils import make_dirs_for_file, exist, load_instance, merge_rules
from SegmentTree import SegmentTree


def ind2route(individual, instance):
    route = []
    max_load = instance['max_load']
    depart_due_time = instance['depart']['due_time']
    # Initialize a sub-route
    sub_route = []
    cur_load = 0
    elapsed_time = 0
    last_station_idx = 0 # 港口
    # individual is a list of order ids:[0,1,2,3,4,9,10,13,14,16,19,20,21,26,28,32,38,40,41,43,44,51,53,54,57,59,60,62,65,70,72,75,76,78,81,87,88,89,93,99]
    segment_list = []
    demand_acc_list = [0]
    demand_acc = 0
    for order_id in individual:
        # Update vehicle load
        ship_idx = order_id // instance['customers_num'] + 1
        customer_id = order_id % instance['customers_num'] + 1
        ready_time = instance[f'order_{ship_idx}_{customer_id}']['ready_time']
        segment_list.append(ready_time)
        demand_acc += instance[f'order_{ship_idx}_{customer_id}']['demand']
        demand_acc_list.append(demand_acc)
    
    # 线段树
    segment_tree = SegmentTree(segment_list)

    i = 0
    j = len(individual)-1
    while i <= j:
        if demand_acc_list[j+1] - demand_acc_list[i] <= max_load:
            max_ready_time = segment_tree.query(i, j)
        else:
            i += 1

    for i in range(len(individual)):
        for j in range(len(individual)-1, i-1, -1):
            ready_time = 0
            for order_id in individual[i:j+1]:
                ship_idx = order_id // instance['customers_num'] + 1
                customer_id = order_id % instance['customers_num'] + 1
                ready_time = max(ready_time, instance[f'order_{ship_idx}_{customer_id}']['ready_time'])
                
            for order_id in individual[i:j+1]:
                ship_idx = order_id // instance['customers_num'] + 1
                customer_id = order_id % instance['customers_num'] + 1
                destination_idx = instance[f'order_{ship_idx}_{customer_id}']['destination_idx']
                demand = instance[f'order_{ship_idx}_{customer_id}']['demand']
                updated_cur_load = cur_load + demand
                # Update elapsed time
                service_time = instance[f'order_{ship_idx}_{customer_id}']['service_time']
                stop_time = instance['stop_time']
                travel_time = instance['distance_matrix'][last_station_idx][destination_idx] / instance['avg_speed']


    for order_id in individual:
        # Update vehicle load
        ship_idx = order_id // instance['customers_num'] + 1
        customer_id = order_id % instance['customers_num'] + 1
        destination_idx = instance[f'order_{ship_idx}_{customer_id}']['destination_idx']
        demand = instance[f'order_{ship_idx}_{customer_id}']['demand']
        updated_cur_load = cur_load + demand
        # Update elapsed time
        service_time = instance[f'order_{ship_idx}_{customer_id}']['service_time']
        stop_time = instance['stop_time']
        travel_time = instance['distance_matrix'][last_station_idx][destination_idx] / instance['avg_speed']

        if (route == []) :
            elapsed_time = instance[f'order_{ship_idx}_{customer_id}']['ready_time']

        order_due_time = instance[f'order_{ship_idx}_{customer_id}']['due_time']
        updated_elapsed_time = elapsed_time + travel_time + stop_time + service_time
        # Validate vehicle load and elapsed time
        if (updated_cur_load <= max_load) and (updated_elapsed_time <= order_due_time):
            # Add to current sub-route
            sub_route.append(order_id)
            cur_load = updated_cur_load
            elapsed_time = updated_elapsed_time
        else:
            # Save current sub-route
            route.append(sub_route)
            # Initialize a new sub-route and add to it
            sub_route = [order_id]
            cur_load = demand
            elapsed_time = instance[f'order_{ship_idx}_{customer_id}']['ready_time'] + travel_time + stop_time + service_time
        # Update last customer ID
        last_station_idx = destination_idx
    if sub_route != []:
        # Save current sub-route before return if not empty
        route.append(sub_route)
    return route


def print_route(route, merge=False):
    '''gavrptw.core.print_route(route, merge=False)'''
    route_str = '0'
    sub_route_count = 0
    for sub_route in route:
        sub_route_count += 1
        sub_route_str = '0'
        for customer_id in sub_route:
            sub_route_str = f'{sub_route_str} - {customer_id}'
            route_str = f'{route_str} - {customer_id}'
        sub_route_str = f'{sub_route_str} - 0'
        if not merge:
            print(f'  Vehicle {sub_route_count}\'s route: {sub_route_str}')
        route_str = f'{route_str} - 0'
    if merge:
        print(route_str)


def eval_vrptw(individual, instance, unit_cost=20000.0, init_cost=0, wait_cost=0, delay_cost=0):
    '''gavrptw.core.eval_vrptw(individual, instance, unit_cost=1.0, init_cost=0, wait_cost=0,
        delay_cost=0)'''
    
    total_cost = 0
    route = ind2route(individual, instance)

    for sub_route in route:
        sub_route_distance = 0
        elapsed_time = 0
        last_station_idx = 0
        sub_route_cost = 0
        sub_route_load = 0

        # 计算子路线最大到港时间
        for order_id in sub_route:
            elapsed_time =  max(instance[f'order_{order_id}']['ready_time'], elapsed_time)

        for order_id in sub_route:
            ship_idx = order_id // instance['customers_num'] + 1
            customer_id = order_id % instance['customers_num'] + 1
            destination_idx = instance[f'order_{ship_idx}_{customer_id}']['destination_idx']

            # Calculate section distance
            distance = instance['distance_matrix'][last_station_idx][destination_idx]
            travel_time = distance / instance['avg_speed']
            cost = instance['cost'][last_station_idx][destination_idx]
            sub_route_load  = sub_route_load + instance[f'order_{ship_idx}_{customer_id}']['demand']

            # Update sub-route distance
            sub_route_distance = sub_route_distance + distance
            sub_route_cost = sub_route_cost + cost

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

    total_cost += init_cost
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


def run_gavrptw(instance_name, unit_cost, init_cost, wait_cost, delay_cost, ind_size, pop_size, \
    cx_pb, mut_pb, n_gen, export_csv=False, customize_data=False):
    '''gavrptw.core.run_gavrptw(instance_name, unit_cost, init_cost, wait_cost, delay_cost,
        ind_size, pop_size, cx_pb, mut_pb, n_gen, export_csv=False, customize_data=False)'''
    if customize_data:
        json_data_dir = os.path.join(BASE_DIR, 'data', 'json_customize')
    else:
        json_data_dir = os.path.join(BASE_DIR, 'data', 'json')
    json_file = os.path.join(json_data_dir, f'{instance_name}.json')
    instance = load_instance(json_file=json_file)
    if instance is None:
        return
    creator.create('FitnessMax', base.Fitness, weights=(1.0, ))
    creator.create('Individual', list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    # Attribute generator
    toolbox.register('indexes', random.sample, range(1, ind_size + 1), ind_size)
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
    print_route(ind2route(best_ind, instance))
    print(f'Total cost: {1 / best_ind.fitness.values[0]}')
    if export_csv:
        csv_file_name = f'{instance_name}_uC{unit_cost}_iC{init_cost}_wC{wait_cost}' \
            f'_dC{delay_cost}_iS{ind_size}_pS{pop_size}_cP{cx_pb}_mP{mut_pb}_nG{n_gen}.csv'
        csv_file = os.path.join(BASE_DIR, 'results', csv_file_name)
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
