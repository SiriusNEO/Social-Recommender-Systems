from benchmark import BenchMark

from Scheduler import dianping_scheduler

model_configs = list()


"""
model_configs.append({
    "model": "MF",
    "lambda1": 0.01,
    "lambda2": 0.01,
    "scheduler": dianping_scheduler
})

model_configs.append({
    "model": "SR",
    "lambda1": 0.01,
    "lambda2": 0.01,
    "beta": 0.55,
    "scheduler": dianping_scheduler
})
"""

benchmark = BenchMark(dataset="dianping",
                      learning_rate=0.001,
                      batch_size = 8192,
                      feature_dim = 20,
                      model_configs=model_configs)

benchmark.train(iterations = 2000, display_interval = 1, test_interval = 10)
benchmark.show()
benchmark.evaluate()

_, ave = benchmark.average_dump()
_, best_para = benchmark.best_dump()

print(ave)
print(best_para)