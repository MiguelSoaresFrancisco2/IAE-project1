import random
import numpy as np

class Config:
    FAST_MODE:bool = True
    PRINT_CONFIRM:bool = True
    COMPARE_METHODS:bool = True
    RANDOM_STATE:int = 42
    TOP_K:int = 10
    LR_MF_SGD:float = 0.01
    REG_MF_SGD:float = 0.05
    
    MAX_USERS_EVAL:int = None
    MF_DIM:int = None
    SGD_EPOCHS:int = None
    ALS_ITERS:int = None
    TOP_M:int = None
    

    def __init__(self):
        if Config.FAST_MODE:
            self.MAX_USERS_EVAL = 300
            self.MF_DIM = 20
            self.SGD_EPOCHS = 10
            self.ALS_ITERS = 8
            self.TOP_M = 80
        else:
            self.MAX_USERS_EVAL = None
            self.MF_DIM = 50
            self.SGD_EPOCHS = 25
            self.ALS_ITERS = 15
            self.TOP_M = 200

    def set_seed(self):
        np.random.seed(self.RANDOM_STATE)
        random.seed(self.RANDOM_STATE)


