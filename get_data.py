import pandas as pd
import json
import numpy as np

train_cost = 20000
container_unload_cost = 50
max_load = 120
avg_speed = 80
transfer_time = 1
stop_time = 0.7
container_unload_time = 0.05


# 读取city_code.xlsx文件，返回字典，key为城市名，value为城市代码
city_code_dict = pd.read_excel('./data/city_code.xlsx').set_index('city')['code'].to_dict()

# 读取 距离 文件
distance_matrix = pd.read_excel('./data/distance.xlsx').iloc[0:, 1:].values

# 读取 成本 文件
cost_matrix = pd.read_excel('./data/cost.xlsx').iloc[0:, 1:].values

# 读取 运输量 文件
shipment_matrix = pd.read_excel('./data/shipment.xlsx').iloc[0:, 1:].values

# 读取 时间窗 文件
time_window_matrix = pd.read_excel('./data/time_window.xlsx').iloc[0:, 1:].values

# 读取 船舶到港 文件
ships_matrix = pd.read_excel('./data/ship_arrival.xlsx').values

# 读取 船舶容量 文件
income_matrix = pd.read_excel('./data/income.xlsx').iloc[0:, 1:].values

# 读取 customer.xlsx 文件
customer_matrix = pd.read_excel('./data/customer.xlsx').values


print("distance_matrix:\n", distance_matrix)
print("cost_matrix:\n", cost_matrix)
print("shipment_matrix:\n", shipment_matrix)
print("time_window_matrix:\n", time_window_matrix)
print("ships_matrix:\n", ships_matrix)
print("income_matrix:\n", income_matrix)
print("city_code_dict:\n", city_code_dict)
print("customer_matrix:\n", customer_matrix)
print("读取数据完成")

# i,j遍历shipment_matrix
json_dic = {}
customers = []
city_codes = list(city_code_dict.keys())
for i in range(len(shipment_matrix)):
    for j in range(len(shipment_matrix[i])):
        if shipment_matrix[i][j] != 0:
            customers.append(i * 20 + j)
            demand = shipment_matrix[i][j]
            ready_time = ships_matrix[i][2] + transfer_time
            due_time = ships_matrix[i][2] + time_window_matrix[i][j]
            service_time = round(shipment_matrix[i][j] * container_unload_time, 2)
            origin = city_codes[0]
            destination = customer_matrix[j][1]
            destination_idx = city_code_dict[destination]
            unload_cost = container_unload_cost * demand
            income = income_matrix[0][destination_idx-1] * demand
            json_dic["order_{}_{}".format(i+1, j+1)] = {
                "demand": demand,
                "ready_time": ready_time,
                "due_time": due_time,
                "service_time" : service_time,
                "origin": origin,
                "origin_idx": 0,
                "destination": destination,
                "destination_idx": destination_idx,
                "unload_cost": unload_cost,
                "income": income,
                "ship_idx" : i+1,
                "customer_idx" : j+1,
            }


json_dic["distance_matrix"] = distance_matrix
json_dic["cost_matrix"] = cost_matrix
json_dic["income_matrix"] = income_matrix
json_dic["max_load"] = max_load
json_dic["avg_speed"] = avg_speed
json_dic["train_cost"] = train_cost
json_dic["stop_time"] = stop_time
json_dic["depart"] = {
    "ready_time": 0,
    "due_time": 10000,
    "service_time": 0,
    "city_code": "大连",
    "city_code": 0,
}
json_dic["ships_num"] = len(ships_matrix)
json_dic["customers_num"] = len(customer_matrix)
json_dic["customers"] = customers

def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError

json_data = json.dumps(json_dic, default=convert_numpy, indent=4, ensure_ascii=False)
# 保存json文件
with open('./data/json_customize/data.json', 'w', encoding='utf-8') as f:
    f.write(json_data)
print("保存json文件完成")
