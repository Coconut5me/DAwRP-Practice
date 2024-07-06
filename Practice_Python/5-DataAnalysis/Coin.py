import random

random_values = [random.choice([1, -1]) for _ in range(100000)]

count_1 = random_values.count(1)
count_minus_1 = random_values.count(-1)

print("Số lần xuất hiện của 1:", count_1)
print("Số lần xuất hiện của -1:", count_minus_1)