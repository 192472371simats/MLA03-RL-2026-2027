tasks = [
    "Pick Object",
    "Move",
    "Place Object"
]
reward = 0
print("Task Execution")
for task in tasks:
    print(task)
    reward += 10
print("\nTotal Reward =", reward)