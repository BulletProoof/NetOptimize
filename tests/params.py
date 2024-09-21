# model parameters
train_cost = 20000 # 班列初始费用
container_unload_cost = 50 # 箱子卸载费用
max_load = 120 # 班列最大载重
avg_speed = 80 # 班列平均速度
transfer_time = 1 # 集装箱转场时间
stop_time = 0.7 # 站点停车时间
container_unload_time = 0.05 # 单个箱子卸载时间

# GA parameters
ind_size = 100 # 染色体长度，这里不需要改，程序自动识别
pop_size = 400 # 群体规模
cx_pb = 0.85 # 交叉概率
mut_pb = 0.05 #0.02， 变异概率
n_gen = 200 # 迭代次数

# 读取数据文件 "./data/json_customize/data.json"
instance_name = 'data'
unit_cost = train_cost
init_cost = 0.0
wait_cost = 1.0
delay_cost = 1.5
random_seed = 64
export_csv = True
use_pool = True
