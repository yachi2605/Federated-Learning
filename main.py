import torch
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import random
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

from fl_simulation.run import run

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print("Starting Federated Learning Simulation for CIFAR-100")

  
    run()