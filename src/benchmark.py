import numpy as np

from dataloader import DataLoader
from UserMean import UserMean
from ItemMean import ItemMean
from MF import MF
from SR import SR
from TriSR import TriSR
from UserCF import UserCF
from ItemCF import ItemCF


class BenchMark():

    """
        Baseline: UserMean, ItemMean
        Model_Config
            dict{
                model_name
                arg1
                arg2
                ...
            }
    """

    def __init__(self, dataset: str,
                       model_configs: list,
                       learning_rate=0.008, 
                       max_iterations = 5000, 
                       batch_size = 256,
                       feature_dim = 10):
        self.max_iterations = max_iterations
        self.dataloader = DataLoader()

        if dataset == "dianping":
            self.dataloader.read_dianping()
        elif dataset == "douban":
            self.dataloader.read_douban()
        elif dataset == "epinions":
            self.dataloader.read_epinions()
        else:
            raise Exception("Invalid dataset")
        
        print("[benchmark] dataset load finish")

        self.baselines = list()

        self.baselines.append(UserMean( 
                    train_set = self.dataloader.trainSet,
                    test_set = self.dataloader.testSet,
                    rating = self.dataloader.ratingList, 
                    m = self.dataloader.userNum, 
                    n = self.dataloader.itemNum, 
                    l = feature_dim))
        
        self.baselines.append(ItemMean( 
                    train_set = self.dataloader.trainSet,
                    test_set = self.dataloader.testSet,
                    rating = self.dataloader.ratingList, 
                    m = self.dataloader.userNum, 
                    n = self.dataloader.itemNum, 
                    l = feature_dim))
        
        self.baselines.append(UserCF(
                            train_set = self.dataloader.trainSet,
                            test_set = self.dataloader.testSet,
                            rating = self.dataloader.ratingList,
                            rset = self.dataloader.ratingSet,
                            network = self.dataloader.adjList,
                            m = self.dataloader.userNum, 
                            n = self.dataloader.itemNum, 
                            l = feature_dim))
        
        self.baselines.append(ItemCF(
                            train_set = self.dataloader.trainSet,
                            test_set = self.dataloader.testSet,
                            rating = self.dataloader.ratingList,
                            rset = self.dataloader.ratingSet,
                            network = self.dataloader.adjList,
                            m = self.dataloader.userNum, 
                            n = self.dataloader.itemNum, 
                            l = feature_dim))

        print("[benchmark] baselines load finish")

        self.models = list()
        self.model_configs = model_configs

        for model_config in model_configs:
        
            if model_config["model"] == "MF":
                model = MF(
                            train_set = self.dataloader.trainSet,
                            test_set = self.dataloader.testSet,
                            rating = self.dataloader.ratingList, 
                            m = self.dataloader.userNum, 
                            n = self.dataloader.itemNum, 
                            l = feature_dim,
                            iterations = self.max_iterations, 
                            learning_rate=learning_rate,
                            batch_size=batch_size,
                            lambda1=model_config["lambda1"], 
                            lambda2=model_config["lambda2"])

            elif model_config["model"] == "SR":
                model = SR(
                            train_set = self.dataloader.trainSet,
                            test_set = self.dataloader.testSet,
                            rating = self.dataloader.ratingList,
                            rset = self.dataloader.ratingSet,
                            network = self.dataloader.adjList,
                            m = self.dataloader.userNum, 
                            n = self.dataloader.itemNum, 
                            l = feature_dim,
                            iterations = self.max_iterations, 
                            learning_rate=learning_rate,
                            batch_size=batch_size,
                            lambda1=model_config["lambda1"], 
                            lambda2=model_config["lambda2"],
                            beta=model_config["beta"])

            elif model_config["model"] == "TriSR":
                model = TriSR(
                            train_set = self.dataloader.trainSet,
                            test_set = self.dataloader.testSet,
                            rating = self.dataloader.ratingList,
                            rset = self.dataloader.ratingSet,
                            network = self.dataloader.adjList,
                            m = self.dataloader.userNum, 
                            n = self.dataloader.itemNum, 
                            l = feature_dim,
                            iterations = self.max_iterations, 
                            learning_rate=learning_rate,
                            batch_size=batch_size,
                            alpha=model_config["alpha"], 
                            beta=model_config["beta"], 
                            gamma=model_config["gamma"],
                            lambda1=model_config["lambda1"], 
                            lambda2=model_config["lambda2"])

                model.user_rank(10, 0.85)
                # show user_rank
                # sample = -np.sort(-self.model3.rank)
                # print(sample[:1000])
                #print("[benchmark] the most influencial user is {}".format(np.where(self.model3.rank==np.max(self.model3.rank))))

            else:
                raise Exception("Unimplemented Model")

            if "scheduler" in model_config:
                model.scheduler = model_config["scheduler"]

            self.models.append(model)

        print("[benchmark] models load finish")


    def train(self, iterations = None, display_interval = 10, test_interval = 50):

        if iterations is None:
            iterations = self.max_iterations
            
        print("running on CPU. iterations = {}.".format(iterations))
        
        for model in self.models:
            model.train(iterations, display_interval, test_interval)
            print("Model {} [{}] trained finish.".format(self.models.index(model), model.model_name))


    def evaluate(self):
        # === test ===

        print("[benchmark] evaluating...")

        for baseline in self.baselines:
            baseline.test(True)

        for model in self.models:
            print("[benchmark] the config info: ", self.model_configs[self.models.index(model)])
            model.test(True)
    

    def show(self):
        print("[benchmark] show best result.")

        for model in self.models:
            print("[benchmark] the config info: ", self.model_configs[self.models.index(model)])
            model.show()
    

    def average_dump(self):
        print("[benchmark] average dump.")

        ret_MAE = dict() 
        ret_RMSE = dict()

        dump_list = self.baselines + self.models

        for model in dump_list:
            if model in self.models:
                config = self.model_configs[self.models.index(model)]
            else:
                model.test(False)
                config = {"model":model.model_name}

            if config["model"] not in ret_MAE:
                ret_MAE[config["model"]] = list()
                ret_RMSE[config["model"]] = list()
                
            ret_MAE[config["model"]].append(model.MAE)
            ret_RMSE[config["model"]].append(model.RMSE)

        for key in ret_MAE:
            ret_MAE[key] = np.mean(ret_MAE[key])
        
        for key in ret_RMSE:
            ret_RMSE[key] = np.mean(ret_RMSE[key])

        return ret_MAE, ret_RMSE

    def best_dump(self):
        print("[benchmark] best para dump.")

        ret_MAE = dict()
        ret_RMSE = dict()

        dump_list = self.baselines + self.models

        for model in dump_list:
            if model in self.models:
                config = self.model_configs[self.models.index(model)]
            else:
                model.test(False)
                config = {"model":model.model_name}

            if config["model"] not in ret_MAE:
                ret_MAE[config["model"]] = [config, model.MAE]
                ret_RMSE[config["model"]] = [config, model.RMSE]
            else:
                if model.MAE < ret_MAE[config["model"]][1]:
                    ret_MAE[config["model"]] = [config, model.MAE]

                if model.RMSE < ret_RMSE[config["model"]][1]:
                    ret_RMSE[config["model"]] = [config, model.RMSE]

        return ret_MAE, ret_RMSE

    def load_model(self):
        for model in self.models:
            model.load()
    
    def save_model(self):
        for model in self.models:
            model.save()
        
        np.save('../model/testSet', np.array(self.dataloader.testSet))