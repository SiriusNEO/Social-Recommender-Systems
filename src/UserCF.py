from SR import *

class UserCF(SR):
    def __init__(self, 
                train_set: list, test_set: list,
                rating: list, rset: list, network: list,
                m: int, n: int, l: int):

        SR.__init__(self, 
                      train_set, test_set, rating, rset, network, m, n, l,
                      model_name="UserCF")
                      
        self.item_rated = [[] for i in range(n+1)]

        for i in range(len(self.rating)):
            for rating_tuple in self.rating[i]:
                self.item_rated[rating_tuple[0]].append(i)
        
        """
            predict is slow, so we overwrite it by randomly choose small test sample
        """

        self.train_set = list()
        self.test_set = list()

        indices = np.random.choice(len(train_set), 1000, replace=False)
        for index in indices:
            self.train_set.append(train_set[index])

        indices = np.random.choice(len(test_set), 1000, replace=False)
        for index in indices:
            self.test_set.append(test_set[index])
    
    def train(self, iterations = None):
        pass

    def predict(self, user_id: int, item_id: int):
        rated_users = self.item_rated[item_id].copy()

        if user_id in rated_users:
            rated_users.remove(user_id)

        if len(rated_users) == 0:
            return 2.5
        
        ret = 0
        ref_cnt = 0
        max_sim = 0
        
        for rated_user in rated_users:
            max_sim = max(max_sim, self.PCC(user_id, rated_user))

        for rated_user in rated_users:
            if self.PCC(user_id, rated_user) >= 0.8*max_sim:
                check = False
                for rating_tuple in self.rating[rated_user]:
                    if rating_tuple[0] == item_id:
                        ret += rating_tuple[1]
                        ref_cnt += 1
                        check = True
                        break
                assert check

        if ref_cnt == 0:
            return 2.5

        return 1.0*ret/ref_cnt 