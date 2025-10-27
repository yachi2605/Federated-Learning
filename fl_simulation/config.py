import torch

class Config:
    NUM_CLIENTS = 64
    NUM_ROUNDS = 250
    FRACTION_FIT = 0.25
    EVAL_EVERY = 1
    LOCAL_EPOCHS = 2
    LOCAL_BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    PROX_MU = 0.001  
    MODEL_NAME = "resnet18"
    NUM_CLASSES = 100
    MAX_WORKERS = 4

    NON_IID_ALPHA: float = 0.4
    
    DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"