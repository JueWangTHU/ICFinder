# -*- coding: utf-8 -*-
# @Time         : 2025/8/24 18:00
# @Author       : Jue Wang
# @Description  : Classification with cross-entropy guided triplet center loss

import torch
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from data import ProteinLigandData
from model import ContinueModel, CNNOD, RNN, TransformerModel
from losses import TripletCenterLoss, CrossEntropy
import numpy as np
import pytorch_lightning as pl
import time
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

# random seed
pl.seed_everything(42)

class TripletClassificationModel(pl.LightningModule):
    def __init__(self,
                margin,
                clw,  # clw: contrastive learning weight
                clf_lr,
                loss_lr,
                loss,
                batch_size,
                backbone
                ):
        # print("clw is short for Contrastive Learning Weight")
        super(TripletClassificationModel, self).__init__()
        self.save_hyperparameters()
        assert backbone in ['full', 'cnn', 'rnn', 'attention'], 'Not support.'
        self.backbone = backbone
        self.triplet_criterion = TripletCenterLoss(margin=margin)

        if loss == 'ce':
            self.clf_criterion = CrossEntropy()
        else:
            raise Exception

        self.clw = clw
        self.clf_lr = clf_lr
        self.loss_lr = loss_lr
        self.automatic_optimization = False  # pause auto optimizer

        # model definitions
        if backbone == 'full':
            self.full_model = ContinueModel()
        elif backbone == 'cnn':
            self.full_model = CNNOD()
        elif backbone == 'rnn':
            self.full_model = RNN()
        elif backbone == 'attention':
            self.full_model = TransformerModel()

    def training_step(self, batch, batch_idx):
        model_opt, loss_opt = self.optimizers()

        model_opt.zero_grad()
        loss_opt.zero_grad()

        feature, label = batch
        score, embedding = self.full_model(feature)

        clf_loss = self.clf_criterion(score, label)
        triplet_loss = self.triplet_criterion(score, label)
        self.log('classification loss', clf_loss)
        self.log('triplet loss', triplet_loss)
        
        #print(score.shape, label.shape)

        loss = clf_loss + self.clw * triplet_loss
        self.log('loss', loss)
        self.manual_backward(loss)
        model_opt.step()
        if self.clw != 0:
            #self.clip_gradients(loss_opt, gradient_clip_val=0.5)
            loss_opt.step()
        # model_scheduler.step()
        # loss_scheduler.step()

        # self.log('model lr', model_scheduler.get_lr()[0])
        # self.log('loss lr', loss_scheduler.get_lr()[0])

        return {'embedding': embedding.reshape(embedding.size(0) * embedding.size(1), -1), 'label': label.reshape(label.size(0) * label.size(1)), 'score': score, 'loss': loss}

    def training_epoch_end(self, outputs):
        # training embedding is saved as batch, which is not the same
        embedding_list = [out['embedding'].detach().cpu().numpy() for out in outputs]
        label_list = [out['label'].cpu().numpy() for out in outputs]
        score_list = [out['score'].detach().cpu().numpy() for out in outputs]

        self.train_embedding = embedding_list
        self.train_label = label_list
        self.train_score = score_list

    def validation_step(self, batch, batch_idx):
        feature, label = batch
        #embedding = self.encoder(feature)
        #score = self.classifier(embedding)
        score, embedding = self.full_model(feature)
        print(score.shape)
        return {'embedding': embedding.squeeze(0), 'label': label.squeeze(0), 'score': score.squeeze(0)}

    def validation_epoch_end(self, outputs):
        # stack encoded features and labels
        embedding_list = [out['embedding'].detach().cpu().numpy() for out in outputs]
        label_list = [out['label'].cpu().numpy() for out in outputs]
        score_list = [out['score'].detach().cpu().numpy() for out in outputs]

        self.val_embedding = embedding_list
        self.val_label = label_list
        self.val_score = score_list

        # metrics
        score = np.concatenate(score_list)
        label = np.concatenate(label_list)
        auc = roc_auc_score(label, score[:, 1])
        mcc = matthews_corrcoef(label, score.argmax(1))
        
        #print(score.shape, label.shape)
        #print(type(score), type(label))
        score = torch.tensor(score).unsqueeze(0).cuda()
        label = torch.tensor(label).unsqueeze(0).cuda()
        #print(score.shape, label.shape)
        #print(type(score), type(label))
        v_clf_loss = self.clf_criterion(score, label)                                                                           
        #v_triplet_loss = self.triplet_criterion(score, label)
        #loss = v_clf_loss + self.clw * v_triplet_loss
        loss = v_clf_loss
        self.log("AUC", auc)
        self.log("MCC", mcc)
        self.log("loss",loss)

    def configure_optimizers(self):
        model_optimizer = torch.optim.Adam(self.full_model.parameters(), lr=self.clf_lr)
        loss_optimizer = torch.optim.Adam(self.triplet_criterion.parameters(), lr=self.loss_lr)
        return [model_optimizer, loss_optimizer]
    

    def on_save_checkpoint(self, checkpoint):
        checkpoint['train_embedding'] = self.train_embedding
        checkpoint['train_label'] = self.train_label
        checkpoint['train_score'] = self.train_score
        checkpoint['val_embedding'] = self.val_embedding
        checkpoint['val_label'] = self.val_label
        checkpoint['val_score'] = self.val_score

    def on_load_checkpoint(self, checkpoint):
        self.train_embedding = checkpoint['train_embedding']
        self.train_label = checkpoint['train_label']
        self.train_score = checkpoint['train_score']
        self.val_embedding = checkpoint['val_embedding']
        self.val_label = checkpoint['val_label']
        self.val_score = checkpoint['val_score']


if __name__ == '__main__':
    batch_size = 4
    
    data_params = {'batch_size': batch_size, 
                    'train_data_root': '../Embedding/esm_train_ICPermeation.pkl',
                    'val_data_root': '../Embedding/esm_valid_ICPermeation.pkl'}

    data = ProteinLigandData(**data_params)
    
    epochs = 20 # 20, 30, (50, 100, 150)
    gpus = [0]
    model_params = {'margin': 5,
                    'clw': 1.0,
                    'clf_lr': 1e-4,
                    'loss_lr': 0.01,
                    'loss': 'ce',
                    'batch_size': batch_size,
                    'backbone': 'full' # full, cnn
                    }
    model_backbone = model_params['backbone'] + '/'
    training_time = time.strftime('%m-%d-%H-%M-%S', time.localtime())
   
    checkpoint = ModelCheckpoint(dirpath='triplet_classification/' + model_backbone + training_time,
                                 save_top_k=1, monitor='MCC', mode='max')
    tb_logger = pl_loggers.TensorBoardLogger(save_dir='triplet_classification/' + model_backbone + training_time, name='pl_logs',
                                             version='', default_hp_metric=False)
    trainer = pl.Trainer(logger=tb_logger, callbacks=checkpoint, max_epochs=epochs,
                         gpus=gpus, log_every_n_steps=1)
    
    model = TripletClassificationModel(**model_params)

    trainer.fit(model, datamodule=data)







