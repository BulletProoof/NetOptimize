# decorators.py

import time

def calculate_time(func):
    """计算函数执行时间的装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took [{round(end_time - start_time, 2)}] seconds to execute")
        return result
    return wrapper
