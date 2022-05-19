import torch
import numpy as np
import random as rd

class Model():

    def __init__(self, 
                train_set: list, test_set: list,
                rating: list, 
                m: int, n: int, l: int,
                model_name = "Basic",
                epochs=1000, learning_rate=0.01, lambda1=0.000001, lambda2=0.000001,
                batch_size=8192):

        self.__device = "cuda"

        self.train_set = train_set
        self.test_set = test_set

        self.m = m
        self.n = n
        self.l = l

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.batch_size = batch_size

        self.U = torch.randn(m+1, l)
        self.V = torch.randn(n+1, l)

        # result
        self.MAE = 2147483647
        self.RMSE = 2147483647

        self.__model = model_name

        indices = list()
        values  = list()

        for i in range(len(self.rating)):
            indices.append([i, self.rating[i][0]])
            values.append(self.rating[i][1])
        
        indices = torch.LongTensor(indices).to(self.__device)
        values  = torch.FloatTensor(values).to(self.__device)

        self.rating_matrix = torch.sparse.FloatTensor(indices, values, torch.Size(m+1, n+1)).to(self.__device)

        print("initialize finish. rating matrix: ", self.rating_matrix.size())
        print(self.rating_matrix)
    

    def calc(self, batch_indices: list):
        print("basic func called")
        pass

    """
        SGD
    """
    def train(self, epochs = None):
        if epochs is None:
            epochs = self.epochs

        for epoch in range(1, epochs+1):
            batch_indices = np.random.choice(self.m, self.batch_size, replace=True)
            loss, grad_U, grad_V = self.calc(batch_indices)

            if epoch % 50 == 0:
                print("epoch {0}, loss = {1}".format(epoch, loss))

            self.test((epoch % 50 == 0))

            self.U -= self.learning_rate * grad_U
            self.V -= self.learning_rate * grad_V
    

    def predict(self, user_id: int, item_id: int):
        return int(torch.dot(self.U[user_id], self.V[item_id]))

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
            # print(pred, testCase[2])
            MAE_test += np.abs(pred - testCase[2]) / T
            RMSE_test += (pred - testCase[2])**2 / T

        RMSE_train = np.sqrt(RMSE_train)
        RMSE_test = np.sqrt(RMSE_test)

        if show_result:
            print("=== [test result] ===")
            print("[train set] {2}: MAE={0} RMSE={1}".format(MAE_train, RMSE_train, self.__model))
            print("[test set] {2}: MAE={0} RMSE={1}".format(MAE_test, RMSE_test, self.__model))
            print("")

        self.MAE = min(self.MAE, MAE_test)
        self.RMSE = min(self.RMSE, RMSE_test)
    

    def show(self):
        print("=== [best result on test set] ===")
        print("{2}: MAE={0} RMSE={1}".format(self.MAE, self.RMSE, self.__model))



