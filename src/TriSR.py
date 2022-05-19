from SR import *

class TriSR(SR):

    def __init__(self, 
                train_set: list, test_set: list,
                rating: list, rset: list, network: list,
                m: int, n: int, l: int,
                scheduler = default_scheduler,
                iterations=1000, learning_rate=0.01, lambda1=0.000001, lambda2=0.000001, 
                alpha=0.01, beta=0.001, gamma=0.01,
                batch_size=8192):

        SR.__init__(self, 
                    train_set, test_set, rating, rset, network, m, n, l,
                    model_name="TriSR",  
                    scheduler = scheduler,
                    iterations=iterations, 
                    learning_rate=learning_rate, 
                    lambda1=lambda1, 
                    lambda2=lambda2, 
                    beta=beta,
                    batch_size=batch_size)
        
        self.alpha = alpha
        self.gamma = gamma

        self.rank = np.ones(m) / m
    

    @jit(parallel=True)
    def user_rank(self, iter_num: int, d: int):
        
        for it in range(iter_num):
            # print("user rank iter {0}".format(it))
            old_rank = self.rank.copy()
            
            for i in range(self.m):

                for friend in self.network[i]:
                    self.rank[i] += old_rank[friend] / len(self.network[friend])
                
                self.rank[i] = d * self.rank[i] + (1-d)/self.m
        
        # normalize

    def fam(self, i: int, f: int):
        leni = len(self.network[i]) 
        lenf = len(self.network[f])

        if leni == 0 or lenf == 0:
            return 0
        
        common = len(set(self.network[i]).intersection(set(self.network[f])))

        return common / np.sqrt(leni) / np.sqrt(lenf)


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
                
                # print(self.rank[friend], sim, self.fam(i, friend))
                vec = np.array([self.alpha*self.rank[friend] + self.beta*sim + self.gamma*self.fam(i, friend)])

                loss += np.linalg.norm(vec) * sim * np.linalg.norm(self.U[i] - self.U[friend])**2
                grad_U[i] += np.linalg.norm(vec) * sim * (self.U[i] - self.U[friend])
            
        grad_U += self.lambda1 * self.U
        grad_V += self.lambda2 * self.V

        loss = 0.5*loss + 0.5*self.lambda1*np.linalg.norm(self.U)**2 + 0.5*self.lambda2*np.linalg.norm(self.V)**2

        return loss / len(batch_indices), grad_U, grad_V