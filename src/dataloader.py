import numpy as np
from random import sample

path = '../dataset/'

DIANPING_MAX_USER = 147918
DIANPING_MAX_ITEM = 11123
DIANPING_MAX_RATING = 2149675
DIANPING_TEST_NUM = 859870 # 2149675 * 0.4

DOUBAN_MAX_USER = 3022
DOUBAN_MAX_ITEM = 6977
DOUBAN_MAX_RATING = 195493
DOUBAN_TEST_NUM = 77798 # 195493 * 0.4

EPINIONS_MAX_USER = 49289
EPINIONS_MAX_ITEM = 139738
EPINIONS_MAX_RATING = 664824
EPINIONS_TEST_NUM = 66482 # 132965 # 265929 # 664824 * 0.4


class DataLoader():

    def __init__(self):
        self.adjList = list()
        self.ratingList = list()
        self.ratingSet = list()
        self.trainSet = list()
        self.testSet = list()
        
        self.userNum = 0
        self.itemNum = 0
        self.ratingNum = 0

        self.dataset_name = ""

    """
        show dataset profile
    """

    def show_dataset(self):
        print("[dataloader] === dataset profile ===")
        print("Dataset Name: {}".format(self.dataset_name))
        print("UserNum={0}, ItemNum={1}, RatingNum={2}".format(self.userNum, self.itemNum, self.ratingNum))
        print("TrainSet Size={}, TestSet Size={}".format(len(self.trainSet), len(self.testSet)))
        print("")

    def read_dianping(self, split=True):
        self.dataset_name = "Dianping"

        for i in range(DIANPING_MAX_USER + 1):
            self.adjList.append(list())
            self.ratingList.append(list())
            self.ratingSet.append(set())

        with open(path + 'dianping/user.txt') as f:
            lines = f.readlines()
        
            for line in lines:
                delim = line.find('|')
                if delim == -1:
                    continue
                user = int(line[:delim])
                neighbor_raw = line[delim+1:].split(' ')
                
                for neib in neighbor_raw:
                    self.adjList[user].append(int(neib))

        with open(path + 'dianping/rating.txt') as f:
            lines = f.readlines()
                
            if split:
                testLines = set(sample(lines, DIANPING_TEST_NUM))
            else:
                testLines = set()

            for line in lines:
                if line.find('|') == -1:
                    continue
                info = line.split('|')
                user = int(info[0])
                item = int(info[1])
                score = (int(info[2]) + 1) * 5 / 6 # scalar to [1, 5]

                if line in testLines:
                    self.testSet.append([user, item, score])
                else:
                    self.trainSet.append([user, item, score])
                    self.ratingList[user].append([item, score])
                    self.ratingSet[user].add(item)
                
                self.itemNum = max(self.itemNum, item)
        
        self.userNum = DIANPING_MAX_USER
        self.ratingNum = DIANPING_MAX_RATING

        print("[dataloader] dataset dianping loaded finish.")
        self.show_dataset()

    def read_douban(self, split=True):
        self.dataset_name = "Douban"

        for i in range(DOUBAN_MAX_USER + 1):
            self.adjList.append(list())
            self.ratingList.append(list())
            self.ratingSet.append(set())

        with open(path + 'douban/uu.txt') as f:
            lines = f.readlines()
        
            for line in lines:
                users_raw = line.split()
                assert len(users_raw) == 2
                user1 = int(users_raw[0])
                user2 = int(users_raw[1])

                self.adjList[user1].append(user2)

        with open(path + 'douban/um.txt') as f:
            lines = f.readlines()
            
            if split:
                testLines = set(sample(lines, DOUBAN_TEST_NUM))
            else:
                testLines = set()

            for line in lines:
                info = line.split()
                assert len(info) == 3
                
                user = int(info[0])
                movie = int(info[1])
                score = int(info[2])

                assert score >= 1 and score <= 5

                if line in testLines:
                    self.testSet.append([user, movie, score])
                else:
                    self.trainSet.append([user, movie, score])
                    self.ratingList[user].append([movie, score])
                    self.ratingSet[user].add(movie)
                
                self.itemNum = max(self.itemNum, movie)

        self.userNum = DOUBAN_MAX_USER
        self.ratingNum = DOUBAN_MAX_RATING

        print("[dataloader] dataset douban loaded finish.")
        self.show_dataset()


    def read_epinions(self, split=True):
        self.dataset_name = "Epinions"

        for i in range(EPINIONS_MAX_USER + 1):
            self.adjList.append(list())
            self.ratingList.append(list())
            self.ratingSet.append(set())

        with open(path + 'epinions/trust_data.txt') as f:
            lines = f.readlines()
        
            for line in lines:
                users_raw = line.split()
                assert len(users_raw) == 3
                user1 = int(users_raw[0])
                user2 = int(users_raw[1])

                self.adjList[user2].append(user1)
                self.adjList[user1].append(user2)

        with open(path + 'epinions/ratings_data.txt') as f:
            lines = f.readlines()

            for line in lines:
                info = line.split()

                # drop empty
                if len(info) != 3:
                    lines = list(filter((line).__ne__, lines))
                    break

            if split:
                testLines = set(sample(lines, EPINIONS_TEST_NUM))
            else:
                testLines = set()

            for line in lines:
                info = line.split()

                assert len(info) == 3

                user = int(info[0])
                movie = int(info[1])
                score = int(info[2])

                assert score >= 1 and score <= 5

                if line in testLines:
                    self.testSet.append([user, movie, score])
                else:
                    self.trainSet.append([user, movie, score])
                    self.ratingList[user].append([movie, score])
                    self.ratingSet[user].add(movie)
                
                self.itemNum = max(self.itemNum, movie)
        
        self.userNum = EPINIONS_MAX_USER
        self.ratingNum = EPINIONS_MAX_RATING

        print("[dataloader] dataset epinions loaded finish.")
        self.show_dataset()