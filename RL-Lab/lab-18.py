import random
tasks = [
    "Welding",
    "Painting",
    "Assembly"
]
knowledge = {}
for task in tasks:
    knowledge[task] = random.randint(80, 100)
print("Meta Learning Results")
for task in knowledge:
    print(task, ":", knowledge[task], "% Learned")