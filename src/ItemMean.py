from Model import *

class ItemMean(Model):

    def __init__(self, 
                train_set: list, test_set: list,
                rating: list, 
                m: int, n: int, l: int):
        
        Model.__init__(self, 
                      train_set, test_set, rating, m, n, l,
                      model_name="ItemMean")

        self.item_rating = [[] for i in range(n+1)]

        for i in range(len(self.rating)):
            for rating_tuple in self.rating[i]:
                self.item_rating[rating_tuple[0]].append(rating_tuple[1])
                

    def train(self, iterations = None):
        pass

    
    def predict(self, user_id: int, item_id: int):
        # no data
        if self.item_rating[item_id] == []:
            return 2.5

        return 1.0 * np.mean(self.item_rating[item_id])