from benchmark import BenchMark

from Scheduler import douban_scheduler

model_configs = list()

"""
model_configs.append({
    "model": "MF",
    "lambda1": 0.01,
    "lambda2": 0.01,
    "scheduler": douban_scheduler
})

model_configs.append({
    "model": "SR",
    "lambda1": 0.01,
    "lambda2": 0.01,
    "beta": 5,
    "scheduler": douban_scheduler
})

model_configs.append({
    "model": "TriSR",
    "lambda1": 0.01,
    "lambda2": 0.01,
    "alpha": 2,
    "beta":  5,
    "gamma": 3,
    "scheduler": douban_scheduler
})
"""

"""
# judge all

for i in range(5):
    model_configs.append({
        "model": "MF",
        "lambda1": 0.01,
        "lambda2": 0.01
    })

    model_configs.append({
        "model": "SR",
        "lambda1": 0.01,
        "lambda2": 0.01,
        "beta": 5
    })

    model_configs.append({
        "model": "TriSR",
        "lambda1": 0.01,
        "lambda2": 0.01,
        "alpha": 2,
        "beta":  5,
        "gamma": 3
    })
"""

"""
try_arg = [
    [1.5, 5, 3],
    [1.5, 5, 3],
    [1.5, 5, 3],

    [1.5, 4, 3],
    [1.5, 4, 3],
    [1.5, 4, 3],
    
    [1.5, 4, 2],
    [1.5, 4, 2],
    [1.5, 4, 2],

    [1.5, 3, 3],
    [1.5, 3, 3],
    [1.5, 3, 3],

    [2, 3, 3],
    [2, 3, 3],
    [2, 3, 3],

    [2, 5, 3],
    [2, 5, 3],
    [2, 5, 3],
]


for arg in try_arg:
    model_configs.append({
        "model": "TriSR",
        "lambda1": 0.01,
        "lambda2": 0.01,
        "alpha": arg[0],
        "beta":  arg[1],
        "gamma": arg[2]
    })
"""

for n in range(-2, 3):
    alpha = 2 * (10**n)
    print("alpha = ", alpha)
    model_configs.append({
        "model": "TriSR",
        "lambda1": 0.01,
        "lambda2": 0.01,
        "alpha": alpha,
        "beta":  5,
        "gamma": 3,
        "scheduler": douban_scheduler
    })

benchmark = BenchMark(dataset="douban", 
                      learning_rate=0.006,
                      model_configs=model_configs)

benchmark.train(iterations = 500, display_interval = 10, test_interval = 10)
benchmark.show()
benchmark.evaluate()

_, ave = benchmark.average_dump()
_, best_para = benchmark.best_dump()

print(ave)
print(best_para)