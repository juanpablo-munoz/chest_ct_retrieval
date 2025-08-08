import torch
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import numpy as np
from tqdm.auto import tqdm
import os
import json
from datetime import datetime
from utils.embedding import extract_embeddings
from utils.selectors import HardestNegativeTripletSelector, SemihardNegativeTripletSelector
from datasets.base import LabelVectorHelper
from losses.losses_local import OnlineTripletLoss
import torch.nn.functional as F
import kornia.augmentation as K
from utils.transforms import RandomGaussianNoise3D
from utils.logging_utils import TripletLogger
import logging

class Trainer:
    def __init__(self, train_loader, val_loader, train_eval_loader, val_eval_loader, train_full_loader, val_full_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, checkpoint_dir, tensorboard_logs_dir, train_full_loader_switch, metrics=[], start_epoch=0, accumulation_steps=1) -> None:
        self.last_train_loss = np.inf
        self.best_val_avg_nonzero_triplets = np.inf
        self.last_val_avg_nonzero_triplets = np.inf
        self.best_val_loss = np.inf
        self.val_map_at_k = []
        self.avg_nonzero_triplets = []
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_eval_loader = train_eval_loader
        self.val_eval_loader = val_eval_loader
        self.train_full_loader = train_full_loader
        self.val_full_loader = val_full_loader
        self.train_full_loader_switch = train_full_loader_switch
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

        self.label_vector_helper = LabelVectorHelper()

        self.train_embeddings = None
        self.train_labels = None
        
        # Add GPU augmentation pipeline similar to micro-F1 training
        self.apply_gpu_aug = True
        self.gpu_aug = K.AugmentationSequential(
            K.RandomAffine3D(degrees=(5, 5, 5), scale=(0.95, 1.05), p=0.5),
            RandomGaussianNoise3D(mean=0.0, std=0.01, p=0.5),
            data_keys=["input"]
        ).to("cuda")
        
        # Initialize structured training loggers
        self.trainer_logger = TripletLogger("TripletTrainer", logging.INFO)
        self.batch_logger = TripletLogger("TripletBatch", logging.DEBUG)
        
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
                if metric.name() == "Average nonzero triplets":
                    self.avg_nonzero_triplets.append(metric.value())

            val_loss, metrics = self.test_epoch()
            #val_loss /= len(val_loader.dataset)

            message += '\n\nEpoch: {}/{}. Validation set:\n\tAverage loss: {:.4f}'.format(self.current_epoch, self.n_epochs,
                                                                                    val_loss)
            for metric in metrics:
                dict_metric = None
                if isinstance(metric, dict):
                    dict_metric = json.dumps(metric.value(), indent=2)
                    message = '\nValidation {}:\n'.format(metric.name())
                else:
                    message = '\nValidation {}: {}'.format(metric.name(), metric.value())
                if metric.name() == "Average nonzero triplets":
                    self.last_val_avg_nonzero_triplets = metric.value()
                elif "mAP@k" in metric.name():
                    k = 10
                    self.val_map_at_k.append(metric.value()['mean_average_precision'][k])
                print(message)
                if(dict_metric):
                    print(dict_metric)


            #if val_loss < best_val_loss or epoch+1 >= self.n_epochs:
            #if self.last_val_avg_nonzero_triplets < self.best_val_avg_nonzero_triplets:
            if self.val_map_at_k[-1] == max(self.val_map_at_k):
                self.best_val_avg_nonzero_triplets = self.last_val_avg_nonzero_triplets
                self.best_val_loss = val_loss
                #print(f"Best validation loss and/or last epoch reached! val_loss:{round(best_val_loss, 4)} with {round(np.array(self.avg_nonzero_triplets).mean(), 1)} average non-zero triplets. Saving model checkpoint...")
                #print(f"Best amount of non-zero triplets reached! val_loss:{round(self.best_val_loss, 4)} with {round(np.array(self.avg_nonzero_triplets).mean())} average non-zero triplets. Saving model checkpoint...")
                print(f"Best validation mAP@{k} reached! mAP@{k}={self.val_map_at_k[-1]} with {self.last_val_avg_nonzero_triplets} average non-zero validation triplets. Saving model checkpoint...")
                metric_str = f"mAP@{k}="+"{:.4f}".format(self.val_map_at_k[-1])
                timestamp = datetime.now().strftime('%Y%m%d')
                epoch_str = "epoch={:03d}".format(self.current_epoch)
                avg_nonzero_triplets_str = "avg-nonzero-val-triplets={:.1f}".format(round(self.last_val_avg_nonzero_triplets, 1))
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.checkpoint_dir, f'triplets_{timestamp}_{epoch_str}_{metric_str}_{avg_nonzero_triplets_str}.pth'),
                )
            elif self.current_epoch == self.n_epochs:
                print(f"Reached end of training! mAP@{k}={self.val_map_at_k[-1]} with {self.last_val_avg_nonzero_triplets} average non-zero validation triplets. Saving model checkpoint...")
                metric_str = f"mAP@{k}="+"{:.4f}".format(self.val_map_at_k[-1])
                timestamp = datetime.now().strftime('%Y%m%d')
                epoch_str = "epoch={:03d}".format(self.current_epoch)
                avg_nonzero_triplets_str = "avg-nonzero-triplets={:.1f}".format(round(self.last_val_avg_nonzero_triplets, 1))
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.checkpoint_dir, f'triplets_{timestamp}_{epoch_str}_{metric_str}_{avg_nonzero_triplets_str}.pth'),
                )


            #self.tensorboard_writer.add_scalars('Loss', {'Training Loss': train_loss, 'Validation Loss': val_loss})
            print(f'### EPOCH {self.current_epoch} END ###\n')
        self.tensorboard_writer.flush()
        self.tensorboard_writer.close()


    def train_epoch(self):
        self.dataset_embeddings = None
        self.dataset_labels = None
        for metric in self.metrics:
            metric.reset()
        self.optimizer.zero_grad()
        losses = []
        total_loss = 0
        in_epoch_train_embeddings = []
        post_epoch_train_embeddings = []
        train_labels = []
        epoch_n_triplets = []
        epoch_n_nonzero_triplets = []
        total_batches = 0
        activation_conditions = len(self.avg_nonzero_triplets) >= 50 and np.array(self.avg_nonzero_triplets[-5:]).mean() < 3.0
        if self.train_full_loader_switch or activation_conditions:
            print(f'\n### NOW FINDING TRIPLETS ACROSS COMPLETE TRAINING DATASET ###')
            self.trainer_logger.logger.info('Switching to full dataset triplet mining mode')
            self.train_full_loader_switch = True # from this epoch onward, train from triplets mined over the full training dataset
            self.model.eval()
            pre_epoch_embeddings = []
            all_labels = []
            margin = self.loss_fn.margin
            negative_compatibles_dict = self.loss_fn.negative_compatibles_dict
            eval_loss_fn = OnlineTripletLoss(margin, HardestNegativeTripletSelector(margin, self.label_vector_helper), negative_compatibles_dict, print_interval=0)
            
            
            self.trainer_logger.logger.info('Extracting pre-epoch training set embeddings')
            with torch.no_grad():
                for data, target in tqdm(self.train_eval_loader):
                    target = target if len(target) > 0 else None
                    
                    if self.cuda:
                        data = data.cuda()
                        if target is not None:
                            target = target.cuda()
                    
                    # Extract embeddings using the model's get_embeddings method with autocast
                    with autocast('cuda', enabled=self.use_amp):

                        # shape: [B, D, 1, H, W] → [B, 1, D, H, W]
                        data = data.permute(0, 2, 1, 3, 4)

                        if data.size()[-2] > self.model.input_H:  
                            data = F.interpolate(
                                data,
                                size=[data.size()[-3], self.model.input_H, self.model.input_W],
                                mode='trilinear',
                                align_corners=False
                            )

                        # Back to [B, D, 1, H, W] if needed downstream
                        data = data.permute(0, 2, 1, 3, 4)

                        embeddings = self.model.get_embeddings(data, use_autocast=True)
                    
                    pre_epoch_embeddings.append(embeddings.cpu())
                    all_labels.append(target.cpu())
            
                    if len(all_labels) >= np.inf: # Early stopping for debugging
                        break
                
            # Concatenate all embeddings and labels for metrics calculation
            if pre_epoch_embeddings and all_labels:
                pre_epoch_embeddings = torch.cat(pre_epoch_embeddings, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
            
            self.trainer_logger.logger.info(f'Now getting losses and triplets from {len(pre_epoch_embeddings)} pre-epoch embeddings.')
            pre_epoch_mean_train_loss, pre_epoch_train_triplets, n_nonzero_triplets = eval_loss_fn(query_embeddings=pre_epoch_embeddings, query_target=all_labels, db_embeddings=pre_epoch_embeddings, db_target=all_labels)
            
            self.trainer_logger.logger.info(f'Now generating batches from {len(pre_epoch_train_triplets)} found triplets.')
            train_batches = self.train_full_loader.generate_batches_from_triplets(pre_epoch_train_triplets)
            n_train_batches = min(30, len(self.train_full_loader))
            
            avg_nonzero = round(np.array(self.avg_nonzero_triplets[-5:]).mean(), 1)
            print(f'Last epoch got training loss of {round(self.last_train_loss, 4)} and last 5 epochs got an average of {avg_nonzero} non-zero triplets.')
            self.trainer_logger.logger.info(f'Last training loss: {self.last_train_loss:.4f}, avg nonzero triplets (last 5 epochs): {avg_nonzero}')
            
            print('Pre-epoch triplet-loss on train dataset:', round(pre_epoch_mean_train_loss.item(), 4))
            print('Non-zero triplets on train dataset:', len(pre_epoch_train_triplets))
            print(f'Training on a sample of {n_train_batches} batches')
            self.trainer_logger.logger.info(f'Pre-epoch triplet stats - Loss: {pre_epoch_mean_train_loss.item():.4f}, Triplets found: {len(pre_epoch_train_triplets)}, Non-zero triplets: {n_nonzero_triplets}, Batch sample size: {n_train_batches}')

            self.model.train()
            self.optimizer.zero_grad()
            accumulation_counter = 0
            for data, target in self.train_full_loader.load_from_batches(train_batches, sample_size=n_train_batches):
            #for data, target in tqdm(train_loader):
                total_batches += 1
                accumulation_counter += 1
                target = target if len(target) > 0 else None
                #if not type(data) in (tuple, list):
                #    data = (data,)
                if self.cuda:
                    data = data.cuda()
                    if target is not None:
                        target = target.cuda()

                with autocast('cuda', enabled=self.use_amp):
                    # Apply data shape transformations and GPU augmentations
                    data = data.permute(0, 2, 1, 3, 4)  # [B, D, 1, H, W] → [B, 1, D, H, W]
                    
                    if self.apply_gpu_aug:
                        data = self.gpu_aug(data)
                        
                    if data.size()[-2] > self.model.input_H:
                        data = F.interpolate(
                            data,
                            size=[data.size()[-3], self.model.input_H, self.model.input_W],
                            mode='trilinear',
                            align_corners=False
                        )
                    data = data.permute(0, 2, 1, 3, 4)  # Back to [B, D, 1, H, W]
                    
                    outputs = self.model(data)
                    
                    print(f'\n### BATCH {total_batches} OF {n_train_batches} ###')
                    #print(f'Batch labels: {target}')
                    loss_outputs = self.loss_fn(query_embeddings=outputs, query_target=target, db_embeddings=outputs, db_target=target)
                    #print('train_epoch.loss_outputs', loss_outputs)
                    if type(loss_outputs) in (tuple, list):
                        loss, triplets, n_nonzero_triplets = loss_outputs
                    else:
                        loss = loss_outputs
                        triplets = []
                        n_nonzero_triplets = 0
                
                # Scale loss by accumulation steps
                loss = loss / self.accumulation_steps
                
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                n_triplets = len(triplets)

                # Log batch results with structured logger
                if total_batches % 5 == 0:  # Log every 5th batch to avoid spam
                    self.batch_logger.logger.info(f'Full dataset mode - Batch {total_batches}: Loss={loss.item():.6f}, Triplets={n_triplets}, Non-zero triplets: {n_nonzero_triplets}')
                
                losses.append(loss.item())
                total_loss += loss.item()

                in_epoch_train_embeddings.append(outputs)
                train_labels.append(target)
                epoch_n_triplets.append(n_triplets)
                epoch_n_nonzero_triplets.append(n_nonzero_triplets)
                
                # Perform optimizer step only when accumulation_counter reaches accumulation_steps
                if accumulation_counter % self.accumulation_steps == 0:
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                
                if len(train_labels) >= np.inf: # Early stopping for debugging
                        break
            
            in_epoch_train_embeddings = torch.cat(in_epoch_train_embeddings, dim=0)
            train_labels = torch.cat(train_labels, dim=0)
            
            self.train_embeddings = in_epoch_train_embeddings.cpu()
            self.train_labels = train_labels.cpu()
            
            in_epoch_mean_train_loss = float(np.mean(losses))
            for metric in self.metrics:
                with torch.no_grad():
                    metric(
                        self.current_epoch,
                        in_epoch_train_embeddings.cpu(),
                        train_labels.cpu(),
                        in_epoch_train_embeddings.cpu(),
                        None,
                        train_labels.cpu(),
                        in_epoch_mean_train_loss,
                        epoch_n_nonzero_triplets,
                        self.tensorboard_writer,
                        training=True,
                    )
            return in_epoch_mean_train_loss, self.metrics
        else:
            self.model.train()
            self.optimizer.zero_grad()
            #for data, target in self.train_loader:
            total_loss = 0.0
            accumulation_counter = 0
            for data, target in tqdm(self.train_loader):
                total_batches += 1
                accumulation_counter += 1
                target = target if len(target) > 0 else None
                if not type(data) in (tuple, list):
                    data = (data,)
                if self.cuda:
                    if not isinstance(data, tuple):
                        data = data.cuda()
                    else:
                        data = tuple(d.cuda() for d in data)
                    if target is not None:
                        target = target.cuda()

                with autocast('cuda', enabled=self.use_amp):
                    # Apply data shape transformations and GPU augmentations
                    if not isinstance(data, tuple):
                        # Single tensor input - apply transformations and augmentations
                        data = data.permute(0, 2, 1, 3, 4)  # [B, D, 1, H, W] → [B, 1, D, H, W]
                        
                        if self.apply_gpu_aug:
                            data = self.gpu_aug(data)
                            
                        if data.size()[-2] > self.model.input_H:
                            data = F.interpolate(
                                data,
                                size=[data.size()[-3], self.model.input_H, self.model.input_W],
                                mode='trilinear',
                                align_corners=False
                            )
                        data = data.permute(0, 2, 1, 3, 4)  # Back to [B, D, 1, H, W]
                        outputs = self.model(data)
                    else:
                        # Tuple input - preserve existing behavior
                        outputs = self.model(*data)
                    
                    print(f'\n### BATCH {total_batches} OF {len(self.train_loader)} ###')
                    #print(f'Batch labels: {target}')
                    loss_outputs = self.loss_fn(query_embeddings=outputs, query_target=target, db_embeddings=outputs, db_target=target)
                    #print('train_epoch.loss_outputs', loss_outputs)
                    if type(loss_outputs) in (tuple, list):
                        loss, triplets, n_nonzero_triplets = loss_outputs
                    else:
                        loss = loss_outputs
                        triplets = []
                        n_nonzero_triplets = 0
                
                # Scale loss by accumulation steps
                loss = loss / self.accumulation_steps
                
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                n_triplets = len(triplets)
                
                
                # Log batch results with structured logger  
                if total_batches % 10 == 0:  # Log every 10th batch to avoid spam
                    self.batch_logger.logger.info(f'Regular mode - Batch {total_batches}/{len(self.train_loader)}: Loss={loss.item():.6f}, Triplets={n_triplets}, Non-zero triplets: {n_nonzero_triplets}')
                
                losses.append(loss.item())
                total_loss += loss.item()

                in_epoch_train_embeddings.append(outputs)
                train_labels.append(target)
                epoch_n_triplets.append(n_triplets)
                epoch_n_nonzero_triplets.append(n_nonzero_triplets)

                # Perform optimizer step only when accumulation_counter reaches accumulation_steps
                if accumulation_counter % self.accumulation_steps == 0:
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                # early epoch stop
                if total_batches >= np.inf: # early stop for debugging
                    break
            
            in_epoch_train_embeddings = torch.cat(in_epoch_train_embeddings, dim=0)
            train_labels = torch.cat(train_labels, dim=0)

            in_epoch_mean_train_loss= sum(losses) / len(losses)

            # Now extract embeddings after weights have been updated
            print('### GETTING TRAINING METRICS ###')
            self.trainer_logger.logger.info('Starting training metrics computation')
            print('### GENERATING POST-EPOCH TRAINING SET EMBEDDINGS ###')
            self.model.eval()
            all_embeddings = []
            all_labels = []
            margin = self.loss_fn.margin
            negative_compatibles_dict = self.loss_fn.negative_compatibles_dict
            eval_loss_fn = OnlineTripletLoss(margin, HardestNegativeTripletSelector(margin, self.label_vector_helper), negative_compatibles_dict, print_interval=0)

            self.trainer_logger.logger.info('Extracting training set embeddings')
            with torch.no_grad():
                for data, target in tqdm(self.train_eval_loader):
                    target = target if len(target) > 0 else None
                    
                    if self.cuda:
                        data = data.cuda()
                        if target is not None:
                            target = target.cuda()
                    
                    # Extract embeddings using the model's get_embeddings method with autocast
                    with autocast('cuda', enabled=self.use_amp):

                        # shape: [B, D, 1, H, W] → [B, 1, D, H, W]
                        data = data.permute(0, 2, 1, 3, 4)

                        if data.size()[-2] > self.model.input_H:  
                            data = F.interpolate(
                                data,
                                size=[data.size()[-3], self.model.input_H, self.model.input_W],
                                mode='trilinear',
                                align_corners=False
                            )

                        # Back to [B, D, 1, H, W] if needed downstream
                        data = data.permute(0, 2, 1, 3, 4)

                        embeddings = self.model.get_embeddings(data, use_autocast=True)
                    
                    all_embeddings.append(embeddings.cpu())
                    all_labels.append(target.cpu())
            
                    if len(all_labels) >= np.inf: # Early stopping for debugging
                        break
                
            # Concatenate all embeddings and labels for metrics calculation
            if all_embeddings and all_labels:
                all_embeddings = torch.cat(all_embeddings, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
            
            

            # Store training embeddings and labels for use in test_epoch
            self.train_embeddings = all_embeddings.cpu()
            self.train_labels = all_labels.cpu()

            # Calculate loss using triplet loss
            print('### EVALUATING LOSS USING TRAIN EMBEDDINGS AS QUERIES ON TRAINING SET EMBEDDINGS ###')
            self.trainer_logger.logger.info('Computing training loss using training embeddings as queries on training embeddings')
            post_epoch_mean_train_loss, train_epoch_triplets, n_nonzero_triplets = eval_loss_fn(query_embeddings=self.train_embeddings, query_target=self.train_labels, db_embeddings=self.train_embeddings, db_target=self.train_labels)

            # Log validation results
            self.trainer_logger.logger.info(f'Training triplet stats - Loss: {post_epoch_mean_train_loss.item():.4f}, Triplets found: {len(train_epoch_triplets)}, Non-zero triplets: {n_nonzero_triplets}')

            for metric in self.metrics:
                with torch.no_grad():
                    metric(
                        self.current_epoch,
                        self.train_embeddings,  # Training embeddings
                        self.train_labels,      # Training labels
                        self.train_embeddings,  # Use same for both query and db
                        None,# predicted labels
                        self.train_labels,      # Use same for both query and db
                        post_epoch_mean_train_loss.item(),
                        [n_nonzero_triplets],  # No triplets
                        self.tensorboard_writer,
                        training=True,
                    )
            
            return post_epoch_mean_train_loss, self.metrics
            
            


    def test_epoch(self):
        print('### GETTING VALIDATION METRICS ###')
        self.trainer_logger.logger.info('Starting validation metrics computation')
        self.model.eval()

        for metric in self.metrics:
            metric.reset()

        all_embeddings = []
        all_labels = []
        margin = self.loss_fn.margin
        negative_compatibles_dict = self.loss_fn.negative_compatibles_dict
        eval_loss_fn = OnlineTripletLoss(margin, HardestNegativeTripletSelector(margin, self.label_vector_helper), negative_compatibles_dict, print_interval=0)
        

        self.trainer_logger.logger.info('Extracting validation set embeddings')
        with torch.no_grad():
            for data, target in tqdm(self.val_eval_loader):
                target = target if len(target) > 0 else None
                
                if self.cuda:
                    data = data.cuda()
                    if target is not None:
                        target = target.cuda()
                
                # Extract embeddings using the model's get_embeddings method with autocast
                with autocast('cuda', enabled=self.use_amp):

                    # shape: [B, D, 1, H, W] → [B, 1, D, H, W]
                    data = data.permute(0, 2, 1, 3, 4)

                    if data.size()[-2] > self.model.input_H:  
                        data = F.interpolate(
                            data,
                            size=[data.size()[-3], self.model.input_H, self.model.input_W],
                            mode='trilinear',
                            align_corners=False
                        )

                    # Back to [B, D, 1, H, W] if needed downstream
                    data = data.permute(0, 2, 1, 3, 4)

                    embeddings = self.model.get_embeddings(data, use_autocast=True)
                
                all_embeddings.append(embeddings.cpu())
                all_labels.append(target.cpu())
                
        
                if len(all_labels) >= np.inf: # Early stopping for debugging
                    break
            
            # Concatenate all embeddings and labels for metrics calculation
            if all_embeddings and all_labels:
                all_embeddings = torch.cat(all_embeddings, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
            
            # Store training embeddings and labels for use in test_epoch
            self.val_embeddings = all_embeddings
            self.val_labels = all_labels

            # Calculate loss using triplet loss
            print('### EVALUATING LOSS USING VAL EMBEDDINGS AS QUERIES ON TRAINING SET EMBEDDINGS ###')
            self.trainer_logger.logger.info('Computing validation loss using val embeddings as queries on training embeddings')
            post_epoch_mean_val_loss, val_epoch_triplets, n_nonzero_triplets = eval_loss_fn(query_embeddings=self.val_embeddings, query_target=self.val_labels, db_embeddings=self.train_embeddings, db_target=self.train_labels)
            
            # Log validation results
            self.trainer_logger.logger.info(f'Training triplet stats - Loss: {post_epoch_mean_val_loss:.4f}, Triplets found: {len(val_epoch_triplets)}, Non-zero triplets: {n_nonzero_triplets}')

        
            for metric in self.metrics:
                metric(
                    self.current_epoch,
                    self.train_embeddings.cpu(),
                    self.train_labels.cpu(),
                    self.val_embeddings.cpu(),
                    None,
                    self.val_labels.cpu(),
                    post_epoch_mean_val_loss.item(),
                    [n_nonzero_triplets],
                    self.tensorboard_writer,
                    training=False,
                )
            self.trainer_logger.logger.info(f'Validation metrics computation completed - Mean loss: {post_epoch_mean_val_loss.item():.4f}')
            return post_epoch_mean_val_loss, self.metrics
