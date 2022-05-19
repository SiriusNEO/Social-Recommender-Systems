from Model import *

class UserMean(Model):

    def __init__(self, 
                train_set: list, test_set: list,
                rating: list, 
                m: int, n: int, l: int):
        
        Model.__init__(self, 
                      train_set, test_set, rating, m, n, l,
                      model_name="UserMean")

    
    def train(self, iterations = None):
        pass

    def predict(self, user_id: int, item_id: int):
        pred = 0

        if len(self.rating[user_id]) == 0:
            return 2.5

        for rating_tuple in self.rating[user_id]:
            pred += rating_tuple[1]

        return 1.0 * pred / len(self.rating[user_id])