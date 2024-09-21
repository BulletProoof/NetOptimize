# -*- coding: utf-8 -*-

import random
from solver.core import run_gavrptw
from get_data import init_data_file_by_order
import params

def main():

    random.seed(params.random_seed)
    instance_name = params.instance_name
    train_cost = params.train_cost

    ind_size = params.ind_size
    pop_size = params.pop_size
    cx_pb = params.cx_pb
    mut_pb = params.mut_pb 
    n_gen = params.n_gen

    export_csv = params.export_csv
    use_pool = params.use_pool
    run_gavrptw(use_pool, instance_name, train_cost, ind_size, pop_size,cx_pb, mut_pb, n_gen, export_csv)


if __name__ == '__main__':
    init_data_file_by_order()
    main()
