from Model import *

class SR(Model):

    def __init__(self, 
                train_set: list, test_set: list,
                rating: list, rset: list, network: list,
                m: int, n: int, l: int,
                model_name = "SR",
                scheduler = default_scheduler,
                iterations=1000, learning_rate=0.01, lambda1=0.000001, lambda2=0.000001, beta=0.001,
                batch_size=8192):

        Model.__init__(self, 
                      train_set, test_set, rating, m, n, l,
                      model_name=model_name,  
                      scheduler=scheduler,
                      iterations=iterations, 
                      learning_rate=learning_rate, 
                      lambda1=lambda1, 
                      lambda2=lambda2, 
                      batch_size=batch_size)

        self.rset = rset
        self.network = network
        self.beta = beta
    
        self.PCC_cache = dict()

    """
        Pearson Correlation Coefficient
    """
    @jit(parallel=True)
    def PCC(self, i: int, f: int):
        
        str_index = str(i) + '#' + str(f)
        str_index_rev = str(f) + '#' + str(i)

        if str_index in self.PCC_cache:
            return self.PCC_cache[str_index]

        # cold start
        if len(self.rating[i]) <= 0 or len(self.rating[f]) <= 0 or self.rset[i].isdisjoint(self.rset[f]):
            return 0

        vec_i = list()
        vec_f = list()

        Ri_mean = 0
        Rf_mean = 0
        for rating_tuple in self.rating[i]:
            Ri_mean += rating_tuple[1]
            if rating_tuple[0] in self.rset[f]:
                vec_i.append(1.0*rating_tuple[1])

        Ri_mean /= len(self.rating[i])

        ind = 0

        for rating_tuple in self.rating[f]:
            Rf_mean += rating_tuple[1]
            if rating_tuple[0] in self.rset[i]:
                vec_f.append(1.0*rating_tuple[1])

        Rf_mean /= len(self.rating[f])

        vec_i = np.array(vec_i)
        vec_f = np.array(vec_f)

        vec_i -= Ri_mean
        vec_f -= Rf_mean

        norm_i = np.linalg.norm(vec_i)
        norm_f = np.linalg.norm(vec_f)

        if norm_i == 0 or norm_f == 0:
            return 0
        else:
            ret = np.dot(vec_i, vec_f) / norm_i / norm_f
        
        self.PCC_cache[str_index] = self.PCC_cache[str_index_rev] = (ret + 1) / 2
        return (ret + 1) / 2


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

            for friend in self.network[i]:
                sim = self.PCC(i, friend)
                if sim < 0:
                    sim = 0
                loss += self.beta * sim * np.linalg.norm(self.U[i] - self.U[friend])**2
                grad_U[i] += self.beta * sim * (self.U[i] - self.U[friend])
        
        grad_U += self.lambda1 * self.U
        grad_V += self.lambda2 * self.V

        loss = 0.5*loss + 0.5*self.lambda1*np.linalg.norm(self.U)**2 + 0.5*self.lambda2*np.linalg.norm(self.V)**2

        return loss / len(batch_indices), grad_U, grad_V