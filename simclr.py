import logging
import os
import sys
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint, get_linear_classifier, generate_experiment_name
from datetime import datetime
import time
torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args,pca_augmentor = None, eigenvalues = None, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']

        self.pca_augmentor = pca_augmentor
        self.eigenvalues = eigenvalues

        self.classifier, self.classifier_optimizer, self.classifier_criterion = get_linear_classifier(
            out_dim=self.args.out_dim, device=self.args.device)

        experiment_name = generate_experiment_name(self.args)

        #self.writer = SummaryWriter()
        self.writer = SummaryWriter(log_dir=os.path.join("runs", experiment_name))
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device) # applied on contrastive learning logits

    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)


        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader, val_loader = None):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        if getattr(self.args, "vit", False):  # Only log ViT-specific settings if using ViT
            logging.info("Using Vision Transformer with the following configuration:")
            logging.info(f"  Patch size:         {self.args.vit_patch_size}")
            logging.info(f"  Hidden size:        {self.args.vit_hidden_size}")
            logging.info(f"  Num layers:         {self.args.vit_layers}")
            logging.info(f"  Num attention heads:{self.args.vit_heads}")
            logging.info(f"  Intermediate size:  {self.args.vit_intermediate_size or self.args.vit_hidden_size * 4}")
            logging.info(f"  Pooling strategy:   {self.args.vit_pooling}")
        else:
            logging.info("Using ResNet encoder.")

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        
        for epoch_counter in range(self.args.epochs):
            self.model.train()
            """self.pca_augmentor.precompute_masks(self.eigenvalues)"""
            epoch_start_time = time.time()
            total_loss = 0.0
            total_top1 = 0.0
            total_samples = 0
            for images, _ in tqdm(train_loader):
            #for images, _, indices in tqdm(train_loader):
                start_time = time.time()
                load_start = time.time()


                """view1, view2 = self.pca_augmentor.extract_views(images, self.eigenvalues, index=indices)
                images = torch.cat([view1, view2], dim=0)"""
                
                images = torch.cat(images, dim=0)
                start = time.time()
                images = images.to(self.args.device, non_blocking = True)

                torch.cuda.synchronize()
                
                #print("device:", time.time() - start)
                load_end = time.time()

                with autocast(enabled=self.args.fp16_precision):
                    forward_start = time.time()
                    features = self.model(images) # forward pass through resnet
                    logits, labels = self.info_nce_loss(features) # compute contrastive loss
                    loss = self.criterion(logits, labels)
                    forward_end = time.time()

                backward_start = time.time()
                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()
                backward_end = time.time()

                batch_size = labels.size(0)
                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                total_loss += loss.item() * batch_size
                total_top1 += top1[0].item() * batch_size
                total_samples += batch_size

                # Dynamically adjust logging interval: start frequent, reduce over time
                log_interval = max(10, self.args.log_every_n_steps * (1 + epoch_counter // 5))
                if n_iter % log_interval == 0:
                #if n_iter % self.args.log_every_n_steps == 0:
                    #top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    #self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)
                    self.writer.add_scalar('Time/Batch', time.time() - start_time, global_step=n_iter)

                    """wandb.log({
                        'loss': loss.item(),
                        'acc/top1': top1[0].item(),
                        'acc/top5': top5[0].item(),
                        'learning_rate': self.scheduler.get_lr()[0],
                        'Time/Batch': time.time() - start_time,
                        'step': n_iter
                    })"""

                n_iter += 1

            """# warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()"""

            self.scheduler.step(epoch_counter)

            avg_loss = total_loss / total_samples
            avg_top1 = total_top1 / total_samples
            
            #logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {avg_loss:.4f}\tTop1 accuracy: {avg_top1:.2f}")
            self.writer.add_scalar('train/epoch_loss', avg_loss, epoch_counter)
            self.writer.add_scalar('train/epoch_top1', avg_top1, epoch_counter)

            """print(f"[Epoch {epoch_counter}] GPU Max Allocated: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
            print(f"[Epoch {epoch_counter}] GPU Max Reserved: {torch.cuda.max_memory_reserved() / 1e6:.1f} MB")"""


            """wandb.log({
                'train/epoch_loss': avg_loss,
                'train/epoch_top1': avg_top1,
                'epoch': epoch_counter
            })"""

            if val_loader is not None:
                val_contrastive_loss, val_cls_acc, val_top1_acc = self.validate(val_loader)
                self.writer.add_scalar('val/contrastive_loss', val_contrastive_loss, epoch_counter)
                self.writer.add_scalar('val/contrastive_top1', val_top1_acc, epoch_counter)

                """wandb.log({
                    'val/contrastive_loss': val_contrastive_loss,
                    'val/contrastive_top1': val_top1_acc,
                    'epoch': epoch_counter
                })"""
                logging.info(f"Validation Contrastive Loss after epoch {epoch_counter}: {val_contrastive_loss:.4f}, Top1 Accuracy: {val_top1_acc:.2f}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

        

    def validate(self, val_loader, epoch=None):
        """Evaluates the learned representations using contrastive loss and a linear classifier.
        Also logs Top-1 contrastive accuracy on the validation set."""
        self.model.eval()

        contrastive_loss_total = 0.0
        contrastive_batches = 0
        val_top1_accuracies = []

        classifier, classifier_optimizer, classifier_criterion = get_linear_classifier(
                out_dim=self.args.out_dim, device=self.args.device)

        feature_list, label_list = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images = torch.cat(images, dim=0).to(self.args.device)  # [2B, C, H, W]
                
                features = self.model(images)
                logits, contrastive_labels = self.info_nce_loss(features)
                loss = self.criterion(logits, contrastive_labels)

                contrastive_loss_total += loss.item()
                contrastive_batches += 1

                # Compute top-1 accuracy on contrastive task
                top1, _ = accuracy(logits, contrastive_labels, topk=(1, 5))
                val_top1_accuracies.append(top1[0].item())

                # Classifier evaluation (single view)
                single_view = images[:labels.size(0)]  # Use only the first view
                with autocast(enabled=self.args.fp16_precision):
                    extracted_features = self.model(single_view)
                feature_list.append(extracted_features.detach())
                label_list.append(labels.to(self.args.device))

        all_features = torch.cat(feature_list, dim=0)
        all_labels = torch.cat(label_list, dim=0)
        #print("Unique labels:", all_labels.unique())
        #print("Classifier output dim:", logits.shape[1])
        # Train linear classifier
        classifier.train()
        for _ in range(100):
            classifier_optimizer.zero_grad()
            logits = classifier(all_features)
            loss = classifier_criterion(logits, all_labels)
            loss.backward()
            classifier_optimizer.step()

        # Evaluate classifier
        classifier.eval()
        with torch.no_grad():
            logits = classifier(all_features)
            preds = logits.argmax(dim=1)
            classification_acc = (preds == all_labels).float().mean().item()

        contrastive_loss_avg = contrastive_loss_total / contrastive_batches
        contrastive_top1_avg = sum(val_top1_accuracies) / len(val_top1_accuracies)

        # Log validation contrastive top-1 accuracy
        if self.writer is not None and epoch is not None:
            self.writer.add_scalar('val/top1_contrastive', contrastive_top1_avg, epoch)

        return contrastive_loss_avg, classification_acc, contrastive_top1_avg

    def linear_probe_full(self, probe_train, probe_test):
        """
        Full-featured linear probing procedure:
        - Split dataset into train/test
        - Train for multiple epochs
        - Report Top-1 and Top-5 accuracy
        """
        from torch.utils.data import random_split, DataLoader
        import torch.nn.functional as F

        self.model.eval()

        if self.args.vit:
            out_dim = self.args.vit_hidden_size
        else:
            if self.args.arch == "resnet18":
                out_dim = 512
            else:
                out_dim = 2048
        
        classifier, optimizer, criterion = get_linear_classifier(
            out_dim=out_dim, device=self.args.device)

        """# Split dataset: 90% train / 10% test
        probe_size = int(0.8 * len(full_dataset))
        probe_train, probe_test = random_split(full_dataset, [probe_size, len(full_dataset) - probe_size])"""

        train_loader = DataLoader(probe_train, batch_size=self.args.batch_size, shuffle=True,
                                num_workers=self.args.workers, drop_last=False)
        test_loader = DataLoader(probe_test, batch_size=self.args.batch_size, shuffle=False,
                                num_workers=self.args.workers, drop_last=False)


        epochs = 100
        for ep in range(epochs):
            classifier.train()
            top1_train_acc = 0
            for counter, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch[0].to(self.args.device) if isinstance(x_batch, list) else x_batch.to(self.args.device)
                y_batch = y_batch.to(self.args.device)

                with torch.no_grad():
                    features = self.model.get_features(x_batch)

                logits = classifier(features)
                loss = criterion(logits, y_batch)
                top1 = accuracy(logits, y_batch, topk=(1,))
                top1_train_acc += top1[0]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            top1_train_acc /= (counter + 1)

            # Test
            classifier.eval()
            top1_test, top5_test = 0, 0
            for counter, (x_batch, y_batch) in enumerate(test_loader):
                x_batch = x_batch[0].to(self.args.device) if isinstance(x_batch, list) else x_batch.to(self.args.device)
                y_batch = y_batch.to(self.args.device)

                with torch.no_grad():
                    #features = self.model(x_batch)
                    features = self.model.get_features(x_batch)
                    logits = classifier(features)
                    top1, top5 = accuracy(logits, y_batch, topk=(1, 5))
                    top1_test += top1[0]
                    top5_test += top5[0]

            top1_test /= (counter + 1)
            top5_test /= (counter + 1)

            #print(f"Epoch {ep}\tTop1 Train accuracy {top1_train_acc.item():.2f}\tTop1 Test accuracy: {top1_test.item():.2f}\tTop5 test acc: {top5_test.item():.2f}")
            log_msg = f"LINEAR PROBE FULL — Epoch {ep}\tTop1 Train accuracy: {top1_train_acc.item():.2f}\tTop1 Test accuracy: {top1_test.item():.2f}\tTop5 Test Accuracy: {top5_test.item():.2f}"

            """wandb.log({
                "linear_probe/train_top1": top1_train_acc.item(),
                "linear_probe/test_top1": top1_test.item(),
                "linear_probe/test_top5": top5_test.item(),
                "linear_probe/epoch": ep
            })"""

            # Log final values
            if self.writer is not None:
                log_path = os.path.join(self.writer.log_dir, 'training.log')
                with open(log_path, "a") as f:
                    f.write(log_msg + "\n")

        return top1_test.item(), top5_test.item()
