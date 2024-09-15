# model parameters
train_cost = 20000
container_unload_cost = 50
max_load = 120
avg_speed = 80
transfer_time = 1
stop_time = 0.7
container_unload_time = 0.05

# GA parameters
ind_size = 100
pop_size = 400
cx_pb = 0.85
mut_pb = 0.02 #0.02
n_gen = 200

# 读取数据文件 "./data/json_customize/data.json"
instance_name = 'data'
unit_cost = train_cost
init_cost = 0.0
wait_cost = 1.0
delay_cost = 1.5
random_seed = 64
export_csv = True
use_pool = False
