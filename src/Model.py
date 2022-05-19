import numpy as np
import random as rd

try:
    from numba import jit
except Exception as e:
    print("WARNING: numba is not installed.")
    
    def jit(parallel=True):
        pass

from Adam import AdamOptimizer
from Scheduler import *

class Model():

    def __init__(self, 
                train_set: list, test_set: list,
                rating: list, 
                m: int, n: int, l: int,
                model_name = "Basic",
                scheduler = default_scheduler,
                iterations=1000, learning_rate=0.01, lambda1=0.000001, lambda2=0.000001,
                batch_size=8192):

        self.train_set = train_set
        self.test_set = test_set

        self.rating = rating

        self.m = m
        self.n = n
        self.l = l

        self.iterations = iterations
        self.learning_rate = learning_rate
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.batch_size = batch_size

        self.U = np.random.rand(m+1, l)
        self.V = np.random.rand(n+1, l)

        # scheduler
        self.scheduler = scheduler

        # optimizer
        self.U_Adam = AdamOptimizer(m+1, l)
        self.V_Adam = AdamOptimizer(n+1, l)

        # loss
        self.loss_record = []

        # result
        self.MAE = 2147483647
        self.RMSE = 2147483647
        self.MAE_train_record = []
        self.RMSE_train_record = []
        self.MAE_test_record = []
        self.RMSE_test_record = []

        self.model_name = model_name

        # print("initialize finish. ", self.U.shape, self.V.shape)
    

    @jit(parallel=True)
    def calc(self, batch_indices: list):
        print("basic func called")
        pass

    """
        SGD
    """
    def train(self, iterations = None, display_interval = 10, test_interval = 50):
        if iterations is None:
            iterations = self.iterations
        
        use_adam = True
        adam_lr = 0.01

        for iteration in range(iterations):
            if self.scheduler(iteration): # get switch flag 
                use_adam = False

            if use_adam:
                batch_indices = range(self.m+1)
            else:
                batch_indices = np.random.choice(self.m, self.batch_size, replace=False)
            
            # print(batch_indices)
            loss, grad_U, grad_V = self.calc(batch_indices)

            self.loss_record.append(loss)

            if (iteration+1) % display_interval == 0:
                print("iteration {0}, loss = {1}, learning rate = {2}".format(iteration, loss, self.learning_rate))

            if not use_adam:
               self.learning_rate = self.scheduler(iteration, self.learning_rate)

            self.test(((iteration+1) % test_interval == 0))

            if use_adam:
                self.U -= self.U_Adam.get_update(grad_U, adam_lr)
                self.V -= self.V_Adam.get_update(grad_V, adam_lr)
            else:
                self.U -= self.learning_rate * grad_U
                self.V -= self.learning_rate * grad_V
    
    def predict(self, user_id: int, item_id: int):
        return np.dot(self.U[user_id], self.V[item_id])

    def test(self, show_result: bool):
        MAE_train = 0
        RMSE_train = 0
        MAE_test = 0
        RMSE_test = 0

        T = 1.0*len(self.train_set)

        for testCase in self.train_set:
            pred = self.predict(testCase[0], testCase[1])
            # print(pred, testCase[2])
            MAE_train += np.abs(pred - testCase[2]) / T
            RMSE_train += (pred - testCase[2])**2 / T

        T = 1.0*len(self.test_set)

        for testCase in self.test_set:
            pred = self.predict(testCase[0], testCase[1])
            
            if pred < 1:
                pred = 1
            elif pred > 5:
                pred = 5
            
            # print(pred, testCase[2])
            MAE_test += np.abs(pred - testCase[2]) / T
            RMSE_test += (pred - testCase[2])**2 / T

        RMSE_train = np.sqrt(RMSE_train)
        RMSE_test = np.sqrt(RMSE_test)

        if show_result:
            print("=== [test result] ===")
            print("[train set] {2}: MAE={0} RMSE={1}".format(MAE_train, RMSE_train, self.model_name))
            print("[test set] {2}: MAE={0} RMSE={1}".format(MAE_test, RMSE_test, self.model_name))
            print("")

        self.MAE = min(self.MAE, MAE_test)
        self.RMSE = min(self.RMSE, RMSE_test)

        self.MAE_train_record.append(MAE_train)
        self.RMSE_train_record.append(RMSE_train)
        self.MAE_test_record.append(MAE_test)
        self.RMSE_test_record.append(RMSE_test)
    

    def show(self):
        print("=== [best result on test set] ===")
        print("{2}: MAE={0} RMSE={1}".format(self.MAE, self.RMSE, self.model_name))


    def load(self):
        self.U = np.load('../model/{0}_U.npy'.format(self.model_name))
        self.V = np.load('../model/{0}_V.npy'.format(self.model_name))


    def save(self):
        np.save('../model/{0}_U'.format(self.model_name), self.U)
        np.save('../model/{0}_V'.format(self.model_name), self.V)



