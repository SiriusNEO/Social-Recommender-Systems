from SR import *

class ItemCF(SR):
    def __init__(self, 
                train_set: list, test_set: list,
                rating: list, rset: list, network: list,
                m: int, n: int, l: int):

        SR.__init__(self, 
                      train_set, test_set, rating, rset, network, m, n, l,
                      model_name="ItemCF")

        self.user_rating = [[] for i in range(m+1)]
                      
        self.item_rating = [[] for i in range(n+1)]
        self.item_rset = [set() for i in range(n+1)]

        for i in range(len(self.rating)):
            for rating_tuple in self.rating[i]:
                self.item_rating[rating_tuple[0]].append([i, rating_tuple[1]])
                self.item_rset[rating_tuple[0]].add(i)
                self.user_rating[i].append(rating_tuple[0])
        
        self.rating = self.item_rating
        self.rset = self.item_rset
        
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
        rated_items = self.user_rating[user_id].copy()

        if item_id in rated_items:
            rated_items.remove(item_id)

        if len(rated_items) == 0:
            return 2.5
        
        ret = 0
        ref_cnt = 0
        max_sim = 0
        
        for rated_item in rated_items:
            max_sim = max(max_sim, self.PCC(item_id, rated_item))

        for rated_item in rated_items:
            if self.PCC(item_id, rated_item) >= 0.9*max_sim:
                check = False
                for rating_tuple in self.rating[rated_item]:
                    if rating_tuple[0] == user_id:
                        ret += rating_tuple[1]
                        ref_cnt += 1
                        check = True
                        break
                assert check
        
        if ref_cnt == 0:
            return 2.5

        return 1.0*ret/ref_cnt 