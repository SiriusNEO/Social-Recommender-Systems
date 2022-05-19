from Model import *

class MF(Model):

    def __init__(self, 
                train_set: list, test_set: list,
                rating: list, 
                m: int, n: int, l: int,
                scheduler = default_scheduler,
                iterations=1000, learning_rate=0.01, lambda1=0.000001, lambda2=0.000001,
                batch_size=8192):

        Model.__init__(self, 
                      train_set, test_set, rating, m, n, l,
                      model_name="Simple MF",  
                      scheduler = scheduler,
                      iterations=iterations, 
                      learning_rate=learning_rate, 
                      lambda1=lambda1, 
                      lambda2=lambda2, 
                      batch_size=batch_size)
    
    @jit(parallel=True)
    def calc(self, batch_indices: list):
        loss = 0
        grad_U = np.zeros(self.U.shape)
        grad_V = np.zeros(self.V.shape)

        for i in batch_indices:
            for rating_tuple in self.rating[i]:
                # rating_tuple = [item, score]
                pred = np.dot(self.U[i], self.V[rating_tuple[0]])
                loss += (rating_tuple[1] - pred)**2
                
                grad_U[i] += (pred - rating_tuple[1]) * self.V[rating_tuple[0]]
                grad_V[rating_tuple[0]] += (pred - rating_tuple[1]) * self.U[i]
        
        grad_U = grad_U + self.lambda1 * self.U
        grad_V = grad_V + self.lambda2 * self.V

        loss = 0.5*loss + 0.5*self.lambda1*np.linalg.norm(self.U)**2 + 0.5*self.lambda2*np.linalg.norm(self.V)**2

        return loss / len(batch_indices), grad_U, grad_V
    
        """
        self.R_row = []
        self.R_col = []
        R_val = []

        for i in range(len(self.rating)):
            for rating_tuple in self.rating[i]:
                self.R_row.append(i)
                self.R_col.append(rating_tuple[0])
                R_val.append(rating_tuple[1])
                
        self.R = csr_matrix((R_val, (self.R_row, self.R_col)))
        """

    """
    def train(self, iterations = None):
        if iterations is None:
            iterations = self.iterations

        for iteration in range(1, iterations+1):
            loss, grad_U, grad_V = self.calc()

            if iteration % 10 == 0:
                print("iteration {0}, loss = {1}, learning rate = {2}".format(iteration, loss, self.learning_rate))

            if self.learning_rate > 0.0001: # 1e-4 as lowerbound
               self.learning_rate = 0.992*self.learning_rate

            self.test((iteration % 50 == 0))

            self.U -= self.learning_rate * grad_U
            self.V -= self.learning_rate * grad_V


    def calc(self):
        grad_U = np.zeros(self.U.shape)
        grad_V = np.zeros(self.V.shape)

        pos = np.array(self.R.todok().keys()).T
        R_pred_val = []
        for i in range(len(self.R_row)):
            R_pred_val.append(self.U[self.R_row[i]].dot(self.V[self.R_col[i]]))
        
        R_pred = csr_matrix((R_pred_val, (self.R_row, self.R_col)))

        loss = 0.5*((self.R -R_pred).power(2)).sum() + 0.5*self.lambda1*np.linalg.norm(self.U)**2 + 0.5*self.lambda2*np.linalg.norm(self.V)**2

        grad_U += (R_pred - self.R).dot(self.V) + self.lambda1 * self.U
        grad_V += (R_pred - self.R).T.dot(self.U) + self.lambda2 * self.V

        return loss, grad_U, grad_V
    """