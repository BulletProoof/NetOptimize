import re

# 假设你的字符串是 "order_1_1"
s = "order_1_10"

# 使用正则表达式提取数字
numbers = re.findall(r'\d+', s)

# 将提取的数字转换为整数
numbers = [int(num) for num in numbers]

print(numbers)  # 输出: [1, 1]

lst= [1,2,3]
print(lst)
