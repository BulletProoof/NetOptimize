# -*- coding: utf-8 -*-

import random
from solver.core import run_gavrptw
from multiprocessing import Pool,cpu_count
from get_data import init_data_file_by_order, init_data_file_by_container

def main(pool):
    '''main()'''
    random.seed(64)

    instance_name = 'data'

    unit_cost = 20000.0
    init_cost = 0.0
    wait_cost = 1.0
    delay_cost = 1.5

    ind_size = 100
    pop_size = 1000
    cx_pb = 0.85
    mut_pb = 0.05 #0.02
    n_gen = 200

    export_csv = True
    run_gavrptw(pool=pool,instance_name=instance_name, unit_cost=unit_cost, init_cost=init_cost, \
        wait_cost=wait_cost, delay_cost=delay_cost, ind_size=ind_size, pop_size=pop_size, \
        cx_pb=cx_pb, mut_pb=mut_pb, n_gen=n_gen, export_csv=export_csv)


if __name__ == '__main__':
    init_data_file_by_order()
    with Pool(processes=cpu_count()) as pool:
        main(pool)
    # main(pool)
