from benchmark import BenchMark

from Scheduler import epinions_scheduler

model_configs = list()

"""
model_configs.append({
    "model": "MF",
    "lambda1": 0.00001,
    "lambda2": 0.00001,
    "scheduler": epinions_scheduler
})

model_configs.append({
    "model": "SR",
    "lambda1": 0.00001,
    "lambda2": 0.00001,
    "beta": 0.55,
    "scheduler": epinions_scheduler
})
"""

model_configs.append({
    "model": "TriSR",
    "lambda1": 0.00001,
    "lambda2": 0.00001,
    "alpha": 0.25,
    "beta":  0.5,
    "gamma": 0.25,
    "scheduler": epinions_scheduler
})

benchmark = BenchMark(dataset="epinions",
                      learning_rate=0.001,
                      batch_size = 2048,
                      feature_dim = 20,
                      model_configs=model_configs)

benchmark.train(iterations = 100, display_interval = 1, test_interval = 1)
benchmark.show()
benchmark.evaluate()

_, ave = benchmark.average_dump()
_, best_para = benchmark.best_dump()

print(ave)
print(best_para)