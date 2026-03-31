import random
import numpy as np


class Config:
    FAST_MODE: bool = True
    SAVE_IMAGES: bool = True
    SHOW_PLOTS: bool = False
    PRINT_CONFIRM: bool = True

    COMPARE_METHODS: bool = True
    METHODS: list[str] = ["mf_sgd", "mf_als"]

    RANDOM_STATE: int = 42

    MAX_USERS_EVAL: int = None
    TOP_K: int = 10

    MF_SGD_EPOCHS: int = None
    MF_SGD_DIM: int = None
    MF_SGD_LR: float = 0.01
    MF_SGD_REG: float = 0.05

    MF_ALS_ITERS: int = 8
    MF_ALS_DIM: int = 20
    MF_ALS_REG: float = 0.05

    TOP_M: int = None

    def __init__(self):
        if Config.FAST_MODE:
            self.MAX_USERS_EVAL = 300
            self.MF_SGD_DIM = 20
            self.MF_SGD_EPOCHS = 10
            self.MF_ALS_ITERS = 8
            self.TOP_M = 80
        else:
            self.MAX_USERS_EVAL = None
            self.MF_SGD_DIM = 50
            self.MF_SGD_EPOCHS = 25
            self.MF_ALS_ITERS = 15
            self.TOP_M = 200

    def set_seed(self):
        np.random.seed(self.RANDOM_STATE)
        random.seed(self.RANDOM_STATE)
