import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from typing import Dict, List
import time
import os
import logging
import threading
import random
from .model_utils import get_model, evaluate
import datetime


class FederatedLearning:
   
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize global model
        self.global_model = get_model(config.MODEL_NAME, config.NUM_CLASSES)
        self.global_model.to(self.device)
        
        
        self.round_metrics = []
        self.batch_logs = []                 # per-batch/per-client rows
        self._batch_lock = threading.Lock()  # thread-safe appends
        self._current_round = -1   

    def _log_batch(self, *, round_idx: int, batch_num: int, client_id: int, loss_value: float, acc_value: float) -> None:
        row = {
            "time": datetime.datetime.utcnow().isoformat(),
            "round": int(round_idx),
            "batch_num": int(batch_num),         
            "client_id": int(client_id),
            "train_loss": float(loss_value),
            "train_acc": float(acc_value),
        }
        with self._batch_lock:
            self.batch_logs.append(row)

    def local_training(self, client_id: int, client_indices: List[int], 
                      train_dataset, round_idx: int) -> Dict:
       


        """Local training on client data - simplified without detailed logging"""
        local_model = get_model(self.config.MODEL_NAME, self.config.NUM_CLASSES)
        local_model.load_state_dict(self.global_model.state_dict())
        local_model.to(self.device)
        
     
        optimizer = torch.optim.Adam(
                local_model.parameters(),
                lr=self.config.LEARNING_RATE)
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        client_dataset = Subset(train_dataset, client_indices)
        client_loader = DataLoader(client_dataset, batch_size=self.config.LOCAL_BATCH_SIZE, 
                                 shuffle=True)
        
        local_model.train()
        
        # Track only final metrics for the client
        total_loss = 0
        correct = 0
        total = 0
        num_batches = 0
        
        for epoch in range(self.config.LOCAL_EPOCHS):
            for data, target in client_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = local_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                pred = output.argmax(dim=1, keepdim=True)
                batch_correct = pred.eq(target.view_as(pred)).sum().item()
                
                total_loss += float(loss.item())
                correct += batch_correct
                total += target.size(0)
                num_batches += 1

                
                batch_acc = 100.0 * (batch_correct / max(1, target.size(0)))
                self._log_batch(
                    round_idx=round_idx,
                    batch_num=num_batches,
                    client_id=client_id,
                    loss_value=float(loss.item()),
                    acc_value=batch_acc,
                )
        
        # Calculate final metrics for this client
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = 100. * correct / total if total > 0 else 0.0
        
        return {

    'client_id': client_id,
    'model_state': local_model.state_dict(),
    'data_size': len(client_indices),   # ← defined here
    'train_loss': avg_loss,
    'train_acc': accuracy

        }
    
    def federated_averaging(self, client_updates: List[Dict]):
        """Perform federated averaging"""
        if not client_updates:
            return
            
        global_dict = self.global_model.state_dict()
        averaged_dict = {}
        
        for key, param in global_dict.items():
            if torch.is_floating_point(param):
                averaged_dict[key] = torch.zeros_like(param)
            else:
                averaged_dict[key] = param.clone()

        total_size = sum(update['data_size'] for update in client_updates)

        for update in client_updates:
            client_state = update['model_state']
            client_weight = update['data_size'] / total_size

            for key in averaged_dict.keys():
                if torch.is_floating_point(averaged_dict[key]):
                    averaged_dict[key] += client_state[key] * client_weight
        
        self.global_model.load_state_dict(averaged_dict)
    
    def run_federated_learning(self, train_dataset, test_dataset, client_data_indices: Dict):
        """Main federated learning loop - simplified"""
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        print(f"Starting federated learning with {self.config.NUM_CLIENTS} clients")

        # Initial test
        initial_loss, initial_accuracy = evaluate(
            self.global_model, test_loader, self.device
        )
        self.round_metrics.append({
            'round': 0,
            'test_loss': initial_loss,
            'test_accuracy': initial_accuracy,
            'train_loss': 0,
            'train_acc': 0,
            'num_clients': 0
        })
        
        for round_idx in range(1, self.config.NUM_ROUNDS + 1):
            round_start = time.time()
            print(f"\nRound {round_idx}/{self.config.NUM_ROUNDS}")

            # Select clients
            num_selected = max(int(self.config.FRACTION_FIT * self.config.NUM_CLIENTS), 1)
            selected_clients = random.sample(range(self.config.NUM_CLIENTS), num_selected)
            
            # Local training
            client_updates = []
            client_train_losses = []
            client_train_accs = []
            
            with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
                futures = {
                    executor.submit(
                        self.local_training, 
                        client_id, 
                        client_data_indices[client_id],
                        train_dataset,
                        round_idx
                    ): client_id for client_id in selected_clients
                }
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        client_updates.append(result)
                        client_train_losses.append(result['train_loss'])
                        client_train_accs.append(result['train_acc'])
                    except Exception as e:
                        logging.exception(f"Client training failed: {e}")
            
            # Federated averaging
            if client_updates:
                self.federated_averaging(client_updates)
            
            # Test global model
            test_loss, test_accuracy = evaluate(
                self.global_model, test_loader, self.device
            )
            
            # Calculate averages
            avg_train_loss = np.mean(client_train_losses) if client_train_losses else 0
            avg_train_acc = np.mean(client_train_accs) if client_train_accs else 0
            round_duration = time.time() - round_start
            
            # Record round metrics
            self.round_metrics.append({
                'round': round_idx,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'train_loss': avg_train_loss,
                'train_acc': avg_train_acc,
                'num_clients': len(client_updates),
                'round_duration': round_duration
            })
            
            logging.info(
                "Round %d | Train Loss: %.4f, Train Acc: %.2f%% | Test Loss: %.4f, Test Acc: %.2f%% | Time: %.2fs",
                round_idx, avg_train_loss, avg_train_acc, test_loss, test_accuracy, round_duration
            )
        
        # Save results
        self.save_results()

    def save_results(self):
        """Save results to CSV files"""
        # Save round metrics only
        round_df = pd.DataFrame(self.round_metrics)
        round_df.to_csv('training_results_rounds.csv', index=False)
        
        # Save model
        torch.save(self.global_model.state_dict(), 'global_model_final.pth')
        if getattr(self, "batch_logs", None):
            pd.DataFrame(self.batch_logs).to_csv("training_results_batches.csv", index=False)
            logging.info("Wrote per-batch logs to training_results_batches.csv")
        # Create summary
        if len(self.round_metrics) > 1:
            final_round = self.round_metrics[-1]
            best_round = max(self.round_metrics[1:], key=lambda x: x['test_accuracy'])
            
            summary = {
                'total_rounds': len(self.round_metrics) - 1,
                'final_test_accuracy': final_round['test_accuracy'],
                'best_test_accuracy': best_round['test_accuracy'],
                'best_round': best_round['round'],
                'total_time': sum(r['round_duration'] for r in self.round_metrics[1:])
            }
            pd.DataFrame([summary]).to_csv('training_summary.csv', index=False)
        
        print("Saved all results and model")
