from solver.core import eval_vrptw
from solver.utils import load_instance

individual =  [87, 2, 65, 38, 89, 28, 40, 4, 44, 3, 19, 
               21, 0, 70, 72, 53, 41, 1, 62, 26, 93, 59, 
               20, 13, 76, 16, 78, 81, 57, 10, 9, 60, 32, 
               51, 54, 75, 88, 43, 14, 99]

instance_name = "data"
json_file = "./data/json_customize/" + f'{instance_name}.json'
instance = load_instance(json_file=json_file)

eval = eval_vrptw(individual,instance,unit_cost=20000, init_cost=0,wait_cost=0,delay_cost=0)
print(eval)
