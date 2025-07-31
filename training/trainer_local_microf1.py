import torch
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import numpy as np
from tqdm.auto import tqdm
import os
import json
from datetime import datetime
import torch.nn.functional as F
from losses.losses_local import GradedMicroF1Loss

class Trainer:
    def __init__(self, train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, checkpoint_dir, tensorboard_logs_dir, metrics=[], start_epoch=0, accumulation_steps=1) -> None:
        self.last_train_loss = np.inf
        self.best_val_loss = np.inf
        self.val_map_at_k = []
        self.val_micro_f1 = []
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.n_epochs = n_epochs
        self.cuda = cuda
        self.log_interval = log_interval
        self.checkpoint_dir = checkpoint_dir
        self.tensorboard_logs_dir = tensorboard_logs_dir
        self.metrics = metrics
        self.start_epoch = start_epoch
        self.current_epoch = start_epoch
        self.accumulation_steps = accumulation_steps
        self.tensorboard_writer = SummaryWriter(self.tensorboard_logs_dir)
        self.tensorboard_writer.add_hparams
        self.use_amp = self.cuda
        self.scaler = GradScaler(enabled=self.use_amp)
        #self.tensorboard_writer.add_graph(self.model)
        for epoch in range(0, self.start_epoch):
            scheduler.step()
            print(f'### SKIPPED EPOCH {epoch+1} ###')

    def fit(self):
        """
        Loaders, model, loss function and metrics should work together for a given task,
        i.e. The model should be able to process data output of loaders,
        loss function should process target output of loaders and outputs from the model

        Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
        Siamese network: Siamese loader, siamese model, contrastive loss
        Online triplet learning: batch loader, embedding model, online triplet loss
        """
        for epoch in range(0, self.start_epoch):
            self.scheduler.step()
            print(f'### SKIPPED EPOCH {epoch+1} ###')

        for epoch in tqdm(range(self.start_epoch, self.n_epochs)):
            self.current_epoch = epoch+1
            print(f'\n### EPOCH {self.current_epoch} START ###')
            # Train stage
            train_loss, metrics = self.train_epoch()
            self.last_train_loss = train_loss
            self.scheduler.step()
            
            message = 'Epoch: {}/{}. Train set:\n\tAverage loss: {:.4f}'.format(self.current_epoch, self.n_epochs, train_loss)
            for metric in metrics:
                message += '\nTraining {}:\n{}'.format(metric.name(), metric.value())

            val_loss, metrics = self.test_epoch()

            message += '\n\nEpoch: {}/{}. Validation set:\n\tAverage loss: {:.4f}'.format(self.current_epoch, self.n_epochs, val_loss)
            for metric in metrics:
                dict_metric = None
                if isinstance(metric, dict):
                    dict_metric = json.dumps(metric.value(), indent=2)
                    message = '\nValidation {}:\n'.format(metric.name())
                else:
                    message = '\nValidation {}: {}'.format(metric.name(), metric.value())
                
                # Track metrics for checkpointing
                if "mAP@k" in metric.name():
                    k = 10
                    self.val_map_at_k.append(metric.value()['mean_average_precision'][k])
                elif metric.name() == "Micro-F1":
                    self.val_micro_f1.append(metric.value())
                    
                print(message)
                if(dict_metric):
                    print(dict_metric)

            # Save best model based on micro-F1 if available, otherwise mAP@k
            best_metric_reached = False
            if self.val_micro_f1 and self.val_micro_f1[-1] == max(self.val_micro_f1):
                best_metric_reached = True
                self.best_val_loss = val_loss
                print(f"Best validation micro-F1 reached! micro-F1={self.val_micro_f1[-1]:.4f}. Saving model checkpoint...")
                metric_str = f"micro-f1={self.val_micro_f1[-1]:.4f}"
                timestamp = datetime.now().strftime('%Y%m%d')
                epoch_str = "epoch={:03d}".format(self.current_epoch)
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.checkpoint_dir, f'microf1_{timestamp}_{epoch_str}_{metric_str}.pth'),
                )
            elif self.val_map_at_k and self.val_map_at_k[-1] == max(self.val_map_at_k):
                best_metric_reached = True
                self.best_val_loss = val_loss
                k = 10
                print(f"Best validation mAP@{k} reached! mAP@{k}={self.val_map_at_k[-1]:.4f}. Saving model checkpoint...")
                metric_str = f"mAP@{k}={self.val_map_at_k[-1]:.4f}"
                timestamp = datetime.now().strftime('%Y%m%d')
                epoch_str = "epoch={:03d}".format(self.current_epoch)
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.checkpoint_dir, f'microf1_{timestamp}_{epoch_str}_{metric_str}.pth'),
                )
            
            # Save final checkpoint at end of training
            if self.current_epoch == self.n_epochs and not best_metric_reached:
                print(f"Reached end of training! Saving final model checkpoint...")
                timestamp = datetime.now().strftime('%Y%m%d')
                epoch_str = "epoch={:03d}".format(self.current_epoch)
                if self.val_micro_f1:
                    metric_str = f"micro-f1={self.val_micro_f1[-1]:.4f}"
                elif self.val_map_at_k:
                    k = 10
                    metric_str = f"mAP@{k}={self.val_map_at_k[-1]:.4f}"
                else:
                    metric_str = "final"
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.checkpoint_dir, f'microf1_{timestamp}_{epoch_str}_{metric_str}.pth'),
                )


            #self.tensorboard_writer.add_scalars('Loss', {'Training Loss': train_loss, 'Validation Loss': val_loss})
            print(f'### EPOCH {self.current_epoch} END ###\n')
        self.tensorboard_writer.flush()
        self.tensorboard_writer.close()


    def train_epoch(self):
        for metric in self.metrics:
            metric.reset()
        
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        losses = []
        accumulation_counter = 0
        
        # Training loop - only calculate loss and update weights
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader)):
            accumulation_counter += 1
            target = target if len(target) > 0 else None
            
            if self.cuda:
                data = data.cuda()
                if target is not None:
                    target = target.cuda()

            with autocast('cuda', enabled=self.use_amp):
                # Get model outputs (logits for classification task)
                outputs = self.model(data)
                
                # For micro-F1 loss, outputs should be logits and target should be relevance vectors
                loss = self.loss_fn(outputs, target)
                
            # Scale loss by accumulation steps
            loss = loss / self.accumulation_steps
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            losses.append(loss.item())
            total_loss += loss.item()
            
            # Perform optimizer step only when accumulation_counter reaches accumulation_steps
            if accumulation_counter % self.accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
            
            if batch_idx % self.log_interval == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}: Loss = {loss.item():.6f}')
        
        in_epoch_mean_train_loss = sum(losses) / len(losses) if losses else 0.0
        
        # Now extract embeddings after weights have been updated
        print('### EXTRACTING POST-EPOCH TRAINING EMBEDDINGS ###')
        self.model.eval()
        all_embeddings = []
        all_labels = []
        all_losses = []
        
        with torch.no_grad():
            for data, target in tqdm(self.train_loader):
                target = target if len(target) > 0 else None
                
                if self.cuda:
                    data = data.cuda()
                    if target is not None:
                        target = target.cuda()
                
                # Extract embeddings from second-to-last layer for classification task
                B, _, _, H, W = data.size()
                x = data.view(B * 100, 3, H, W)
                x = self.model.features(x)
                x = x.view(B, 100, 512, 10, 10)
                x = self.model.reducingconvs(x)
                x = x.view(B, self.model.flattened_size)
                embeddings = self.model.fc(x)
                embeddings = F.normalize(embeddings, p=2, dim=1)  # Second-to-last layer
                
                all_embeddings.append(embeddings.cpu())
                all_labels.append(target.cpu())

                # Get logits for loss calculation by applying classifier to embeddings
                logits = self.model.classifier(embeddings)
                
                # Calculate loss using micro-F1 loss
                loss = self.loss_fn(logits, target)
                all_losses.append(loss.item())
        
        # Concatenate all embeddings and labels for metrics calculation
        if all_embeddings and all_labels:
            all_embeddings = torch.cat(all_embeddings, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
        
        post_epoch_mean_train_loss = np.append(all_losses)

        # Store training embeddings and labels for use in test_epoch
        self.train_embeddings = all_embeddings
        self.train_labels = all_labels

        print('in_epoch_mean_train_loss =', in_epoch_mean_train_loss)
        print('post_epoch_mean_train_loss =', post_epoch_mean_train_loss)
        
        for metric in self.metrics:
            with torch.no_grad():
                metric(
                    self.current_epoch,
                    all_embeddings,  # Training embeddings
                    all_labels,      # Training labels
                    all_embeddings,  # Use same for both query and db
                    all_labels,      # Use same for both query and db
                    post_epoch_mean_train_loss,
                    [],  # No triplets
                    self.tensorboard_writer,
                    training=True,
                )
        
        return post_epoch_mean_train_loss, self.metrics

    def test_epoch(self):
        print('### GETTING VALIDATION METRICS ###')
        with torch.no_grad():
            self.model.eval()
            for metric in self.metrics:
                metric.reset()
            
            # Single loop over validation set to calculate loss and extract embeddings
            losses = []
            val_embeddings = []
            val_labels = []
            
            for data, target in tqdm(self.val_loader):
                target = target if len(target) > 0 else None
                
                if self.cuda:
                    data = data.cuda()
                    if target is not None:
                        target = target.cuda()
                
                with autocast('cuda', enabled=self.use_amp):
                    # Extract embeddings from second-to-last layer for classification task
                    B, _, _, H, W = data.size()
                    x = data.view(B * 100, 3, H, W)
                    x = self.model.features(x)
                    x = x.view(B, 100, 512, 10, 10)
                    x = self.model.reducingconvs(x)
                    x = x.view(B, self.model.flattened_size)
                    embeddings = self.model.fc(x)
                    embeddings = F.normalize(embeddings, p=2, dim=1)  # Second-to-last layer
                    
                    # Get logits for loss calculation by applying classifier to embeddings
                    logits = self.model.classifier(embeddings)
                    
                    # Calculate loss using micro-F1 loss
                    loss = self.loss_fn(logits, target)
                    
                losses.append(loss.item())
                val_embeddings.append(embeddings.cpu())
                val_labels.append(target.cpu())
            
            mean_val_loss = sum(losses) / len(losses) if losses else 0.0
            
            # Concatenate validation embeddings and labels
            if val_embeddings and val_labels:
                val_embeddings = torch.cat(val_embeddings, dim=0)
                val_labels = torch.cat(val_labels, dim=0)
            
            # Use training embeddings from train_epoch as database
            for metric in self.metrics:
                with torch.no_grad():
                    metric(
                        self.current_epoch,
                        self.train_embeddings,  # Database embeddings (training set)
                        self.train_labels,      # Database labels (training set)
                        val_embeddings,         # Query embeddings (validation set)
                        val_labels,             # Query labels (validation set)
                        mean_val_loss,
                        [],  # No triplets
                        self.tensorboard_writer,
                        training=False,
                    )
            
            return mean_val_loss, self.metrics
