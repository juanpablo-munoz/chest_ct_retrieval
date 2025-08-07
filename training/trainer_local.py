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
from losses.losses import OnlineTripletLoss
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
        post_epoch_train_embeddings = []
        train_labels = []
        epoch_n_triplets = []
        total_batches = 0
        activation_conditions = len(self.avg_nonzero_triplets) >= 5 and np.array(self.avg_nonzero_triplets[-5:]).mean() < 5.0
        if self.train_full_loader_switch or activation_conditions:
            self.train_full_loader_switch = True # from this epoch onward, train from triplets mined over the full training dataset
            self.model.eval()
            margin = self.loss_fn.margin
            negative_compatibles_dict = self.loss_fn.negative_compatibles_dict
            eval_loss_fn = OnlineTripletLoss(margin, SemihardNegativeTripletSelector(margin), negative_compatibles_dict, print_interval=0)
            pre_epoch_embeddings, labels = extract_embeddings(self.train_eval_loader, self.model)
            true_loss_outputs, pre_epoch_triplets = eval_loss_fn(query_embeddings=pre_epoch_embeddings, query_target=labels, db_embeddings=pre_epoch_embeddings, db_target=labels)
            train_batches = self.train_full_loader.generate_batches_from_triplets(pre_epoch_triplets)
            n_train_batches = min(30, len(self.train_full_loader))
            print(f'\n### NOW FINDING TRIPLETS ACROSS COMPLETE TRAINING DATASET ###')
            self.trainer_logger.logger.info('Switching to full dataset triplet mining mode')
            
            avg_nonzero = round(np.array(self.avg_nonzero_triplets[-5:]).mean(), 1)
            print(f'Last epoch got training loss of {round(self.last_train_loss, 4)} and last 5 epochs got an average of {avg_nonzero} non-zero triplets.')
            self.trainer_logger.logger.info(f'Last training loss: {self.last_train_loss:.4f}, avg nonzero triplets (last 5 epochs): {avg_nonzero}')
            
            print('True triplet-loss on train dataset:', round(true_loss_outputs.item(), 4))
            print('Non-zero triplets on train dataset:', len(pre_epoch_triplets))
            print(f'Training on a sample of {n_train_batches} batches')
            self.trainer_logger.logger.info(f'Pre-epoch triplet stats - Loss: {true_loss_outputs.item():.4f}, Triplets found: {len(pre_epoch_triplets)}, Batch sample size: {n_train_batches}')

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
                        
                    data = F.interpolate(
                        data,
                        size=[data.size()[-3], self.model.input_H, self.model.input_W],
                        mode='trilinear',
                        align_corners=False
                    )
                    data = data.permute(0, 2, 1, 3, 4)  # Back to [B, D, 1, H, W]
                    
                    outputs = self.model(data)
                    
                    print(f'\n### BATCH {total_batches} TRAINING LOSS ###')
                    print(f'Batch labels: {target}')
                    loss_outputs = self.loss_fn(query_embeddings=outputs, query_target=target, db_embeddings=outputs, db_target=target)
                    #print('train_epoch.loss_outputs', loss_outputs)
                    if type(loss_outputs) in (tuple, list):
                        loss, triplets = loss_outputs
                    else:
                        loss = loss_outputs
                        triplets = []
                
                # Scale loss by accumulation steps
                loss = loss / self.accumulation_steps
                
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                n_triplets = len(triplets)
                print('loss:', loss)
                print('n_triplets:', round(n_triplets, 1))
                
                # Log batch results with structured logger
                if total_batches % 5 == 0:  # Log every 5th batch to avoid spam
                    self.batch_logger.logger.info(f'Full dataset mode - Batch {total_batches}: Loss={loss.item():.6f}, Triplets={n_triplets}')
                
                losses.append(loss.item())
                total_loss += loss.item()

                post_epoch_train_embeddings.append(outputs)
                train_labels.append(target)
                epoch_n_triplets.append(n_triplets)
                
                # Perform optimizer step only when accumulation_counter reaches accumulation_steps
                if accumulation_counter % self.accumulation_steps == 0:
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()
            
            post_epoch_train_embeddings = torch.cat(post_epoch_train_embeddings, dim=0)
            train_labels = torch.cat(train_labels, dim=0)
            #print('epoch_predictions:', epoch_predictions)
            #print('epoch_targets:', epoch_targets)
            mean_train_loss = total_loss
            for metric in self.metrics:
                with torch.no_grad():
                    metric(
                        self.current_epoch,
                        post_epoch_train_embeddings.cpu(),
                        train_labels.cpu(),
                        post_epoch_train_embeddings.cpu(),
                        train_labels.cpu(),
                        mean_train_loss,
                        epoch_n_triplets,
                        self.tensorboard_writer,
                        training=True,
                    )
            return mean_train_loss, self.metrics
        else:
            self.model.train()
            self.optimizer.zero_grad()
            #for data, target in self.train_loader:
            total_loss = 0.0
            resnet_total_loss_grad = 0.0
            reducingconvs_total_loss_grad = 0.0
            fc_total_loss_grad = 0.0
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
                    
                    print(f'\n### BATCH {total_batches} OF {len(self.train_loader)} TRAINING LOSS ###')
                    print(f'Batch labels: {target}')
                    loss_outputs = self.loss_fn(query_embeddings=outputs, query_target=target, db_embeddings=outputs, db_target=target)
                    #print('train_epoch.loss_outputs', loss_outputs)
                    if type(loss_outputs) in (tuple, list):
                        loss, triplets = loss_outputs
                    else:
                        loss = loss_outputs
                        triplets = []
                
                # Scale loss by accumulation steps
                loss = loss / self.accumulation_steps
                
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                n_triplets = len(triplets)
                print('loss:', loss)
                print('n_triplets:', round(n_triplets, 1))
                
                # Log batch results with structured logger  
                if total_batches % 10 == 0:  # Log every 10th batch to avoid spam
                    self.batch_logger.logger.info(f'Regular mode - Batch {total_batches}/{len(self.train_loader)}: Loss={loss.item():.6f}, Triplets={n_triplets}')
                
                losses.append(loss.item())
                total_loss += loss.item()

                # Gradients of layers of interest with respect to the trainig loss
                # resnet_loss_grad, *_ =  self.model.features[0].weight.grad.data
                # resnet_total_loss_grad += resnet_loss_grad
                # reducingconvs_loss_grad, *_ =  self.model.reducingconvs[0].weight.grad.data
                # reducingconvs_total_loss_grad += reducingconvs_loss_grad
                # fc_loss_grad, *_ =  self.model.fc[0].weight.grad.data
                # fc_total_loss_grad += fc_loss_grad

                post_epoch_train_embeddings.append(outputs)
                train_labels.append(target)
                epoch_n_triplets.append(n_triplets)

                # Perform optimizer step only when accumulation_counter reaches accumulation_steps
                if accumulation_counter % self.accumulation_steps == 0:
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                # early epoch stop
                #if total_batches >= 10:
                if total_batches >= np.inf:
                    break
            
            post_epoch_train_embeddings = torch.cat(post_epoch_train_embeddings, dim=0)
            train_labels = torch.cat(train_labels, dim=0)
            #print('post_epoch_train_embeddings:', post_epoch_train_embeddings)
            #print('train_labels:', train_labels)
            #mean_train_loss = total_loss.item() / len(losses)
            mean_train_loss = sum(losses) / len(losses)
            
            # Gradient monitoring
            # print('Mean gradient of the first layer weights in the "features" section with respect to the training loss (dW_0/dL):', resnet_total_loss_grad / total_batches)
            # print('Mean gradient of the first layer weights in the "reducingconvs" with respect to the training loss:', reducingconvs_total_loss_grad / total_batches)
            # print('Mean gradient of the first layer weights in the "fc" with respect to the training loss:', fc_total_loss_grad / total_batches)
            for metric in self.metrics:
                with torch.no_grad():
                    metric(
                        self.current_epoch,
                        post_epoch_train_embeddings.cpu(),
                        train_labels.cpu(),
                        post_epoch_train_embeddings.cpu(),
                        train_labels.cpu(),
                        mean_train_loss,
                        epoch_n_triplets,
                        self.tensorboard_writer,
                        training=True,
                    )
            return mean_train_loss, self.metrics


    def test_epoch(self):
        print('### GETTING VALIDATION METRICS ###')
        self.trainer_logger.logger.info('Starting validation metrics computation')
        with torch.no_grad():
            self.model.eval()
            for metric in self.metrics:
                metric.reset()
            margin = self.loss_fn.margin
            negative_compatibles_dict = self.loss_fn.negative_compatibles_dict
            eval_loss_fn = OnlineTripletLoss(margin, SemihardNegativeTripletSelector(margin), negative_compatibles_dict, print_interval=0)
            print('### GENERATING POST-EPOCH TRAINING SET EMBEDDINGS ###')
            self.trainer_logger.logger.info('Extracting training set embeddings')
            post_epoch_train_embeddings, train_labels = extract_embeddings(self.train_eval_loader, self.model)
            print('### GENERATING POST-EPOCH VALIDATION SET EMBEDDINGS ###')
            self.trainer_logger.logger.info('Extracting validation set embeddings')
            post_epoch_val_embeddings, val_labels = extract_embeddings(self.val_eval_loader, self.model)
            #true_val_loss_outputs, val_epoch_triplets = eval_loss_fn(post_epoch_val_embeddings, val_labels)
            print('### EVALUATING LOSS USING VAL EMBEDDINGS AS QUERIES ON TRAINING SET EMBEDDINGS ###')
            self.trainer_logger.logger.info('Computing validation loss using val embeddings as queries on training embeddings')
            true_val_loss_outputs, val_epoch_triplets = eval_loss_fn(query_embeddings=post_epoch_val_embeddings, query_target=val_labels, db_embeddings=post_epoch_train_embeddings, db_target=train_labels)
            epoch_predictions = []
            epoch_targets = []
            epoch_n_triplets = []
            loss = true_val_loss_outputs.item()
            #loss = sum([l.item() for l in true_val_loss_outputs])
            n_triplets = len(val_epoch_triplets)
            #n_triplets = sum([len(t) for t in val_epoch_triplets])
            epoch_targets.append(val_labels)
            epoch_n_triplets.append(n_triplets)
            
            # Log validation results
            self.trainer_logger.logger.info(f'Validation triplet stats - Loss: {loss:.4f}, Triplets found: {n_triplets}')
            #epoch_predictions = torch.cat(epoch_predictions, dim=0)
            epoch_predictions = torch.empty(0)
            #epoch_targets = torch.cat(epoch_targets, dim=0)
            epoch_targets = torch.empty(0)
            for metric in self.metrics:
                with torch.no_grad():
                    metric(
                        self.current_epoch,
                        post_epoch_train_embeddings.cpu(),
                        train_labels.cpu(),
                        post_epoch_val_embeddings.cpu(),
                        val_labels.cpu(),
                        loss,
                        epoch_n_triplets,
                        self.tensorboard_writer,
                        training=False,
                    )
            mean_val_loss = loss
            self.trainer_logger.logger.info(f'Validation metrics computation completed - Mean loss: {mean_val_loss:.4f}')
            return mean_val_loss, self.metrics
