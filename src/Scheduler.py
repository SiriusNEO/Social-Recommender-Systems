"""
    scheduling: adam to SGD and learning rate decay
"""

def default_scheduler(iteration: int, learning_rate = None):
    if learning_rate is None:
        return iteration >= 20

    return learning_rate


def douban_scheduler(iteration: int, learning_rate = None):
    if learning_rate is None:
        return iteration >= 20

    if learning_rate > 0.0001: # 1e-4 as lowerbound
        return 0.992*learning_rate
    return learning_rate


def dianping_scheduler(iteration: int, learning_rate = None):
    if learning_rate is None:
        return iteration >= 15

    if learning_rate > 0.0001 and (iteration+1) % 10 == 0: # 1e-4 as lowerbound
        return 0.99*learning_rate
    return learning_rate


def epinions_scheduler(iteration: int, learning_rate = None):
    if learning_rate is None:
        return iteration > 10

    if learning_rate > 0.0006: # 6e-4 as lowerbound
        return 0.995*learning_rate
    return learning_rate