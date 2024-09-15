# -*- coding: utf-8 -*-

import random
from solver.core import run_gavrptw
from multiprocessing import Pool,cpu_count
from get_data import init_data_file_by_order, init_data_file_by_container
import params

def main():

    random.seed(params.random_seed)
    instance_name = params.instance_name
    unit_cost = params.unit_cost

    ind_size = params.ind_size
    pop_size = params.pop_size
    cx_pb = params.cx_pb
    mut_pb = params.mut_pb 
    n_gen = params.n_gen

    export_csv = params.export_csv
    use_pool = params.use_pool
    run_gavrptw(use_pool=use_pool,instance_name=instance_name, unit_cost=unit_cost, init_cost=0, \
        wait_cost=0, delay_cost=0, ind_size=ind_size, pop_size=pop_size, \
        cx_pb=cx_pb, mut_pb=mut_pb, n_gen=n_gen, export_csv=export_csv)


if __name__ == '__main__':
    init_data_file_by_order()
    main()
