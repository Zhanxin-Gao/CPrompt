
import logging
import copy
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from network.cprompt_net import CPrompt_Net
from utils.toolkit import target2onehot, tensor2numpy, accuracy
from scipy.spatial.distance import cdist
from utils.toolkit import count_parameters
from .base_learner import BaseLearner
import os
from scipy import stats

dataset_classes = {
    "cifar100_vit": 100,
    "domainnet": 200,
    "imagenetr": 200,
    "stanfordcars":196
}

class CPrompt(BaseLearner):
    def __init__(self, args):
        self.args=args
        self.topk=args["topk"]
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._device = args['device'][0]
        self.dataset_name=args["dataset"]
        self.args["num_classes"] = dataset_classes.get(self.dataset_name, 0) 
        self._network=CPrompt_Net(self.args)
        self.acc=[]
        self.faa_accuracy_table=[]
        
    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        cur_task_nbclasses=data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + cur_task_nbclasses
        self._network.update_fc(self._total_classes,cur_task_nbclasses)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=None)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=8)
        self._train(self.train_loader,self.test_loader)
        
        self._network.fix_branch_layer()
        
    def _train(self,train_loader,test_loader):
        self._network.to(self._device)
        
        enabled = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters()), momentum=0.9,lr=self.args["lr"],weight_decay=self.args["weight_decay"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.args["epochs"])
        self._classifier_train(train_loader,test_loader,optimizer,scheduler)

    def _classifier_train(self,train_loader,test_loader,optimizer,scheduler):
        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                new_targets=targets-self._known_classes
                logits,features = self._network.aux_forward(inputs)
                loss_aux=F.cross_entropy(logits,new_targets)
                loss=loss_aux
                
                if self._cur_task>0:
                    for k in range(self._cur_task):
                        old_logit=self._network.clas_w[k](features)['logits']
                        c1_logits=self._network.clas_w[self._cur_task](features)['logits']
                        bool_=torch.max(c1_logits,dim=1)[0]>torch.max(old_logit,dim=1)[0]+self.args["margin"]
                        t=torch.ones((bool_.shape)).to(self._device)
                        t[bool_==False]=self.args["tau"]
                        t=t.unsqueeze(1).repeat(1,self.args["increment"])
                        ground=F.softmax(old_logit/t,dim=1).detach().clone()
                        loss_ccl = -torch.sum(ground * torch.log(F.softmax(old_logit,dim=1)), dim=1).mean()
                        loss+=self.args["alpha"]*loss_ccl/self._cur_task
                        
                gen_p=[]
                x_querry = self._network.image_encoder(inputs, returnbeforepool=True)[:,0,:]
                K=self._network.keys
                
                s=self._cur_task*self.args["increment"]
                f=(self._cur_task+1)*self.args["increment"]
                if self._cur_task==0:
                    K = K[s:f]
                else:
                    K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
                n_K = nn.functional.normalize(K, dim=1)
                q = nn.functional.normalize(x_querry, dim=1)
                mk = torch.einsum('bd,kd->bk', q, n_K)
                loss_mk=F.cross_entropy(mk,targets)
                loss+=loss_mk
                
                m=torch.randint(0,self._cur_task+1,(len(mk),1))
                ts_prompts_1=self._network.ts_prompts_1
                P1=torch.cat([ts_prompts_1[j].weight.unsqueeze(0) for j in m],dim=0)
                gen_p.append(P1)
                ts_prompts_2=self._network.ts_prompts_2
                P2=torch.cat([ts_prompts_2[j].weight.unsqueeze(0) for j in m],dim=0)
                gen_p.append(P2)
                out_gen=self._network(inputs,gen_p,train=True)
                loss_ce=F.cross_entropy(out_gen,new_targets)
                loss+=loss_ce

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(new_targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            
            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch+1, self.args["epochs"], losses/len(train_loader), train_acc)
            
            prog_bar.set_description(info)
        logging.info(info)

    def _eval_cnn(self, loader): 
        faa_y_true=[]
        total = 0

        cor=0
        faa_pred=[]

        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            
            gen_p=[]
            with torch.no_grad():
                x_querry = self._network.image_encoder(inputs, returnbeforepool=True)[:,0,:]
            
            K=self._network.keys
            
            f=(self._cur_task+1)*self.args["increment"]
            K = K[:f]
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_querry, dim=1)
            mk = torch.einsum('bd,kd->bk', q, n_K)
            
            m=torch.max(mk,dim=1,keepdim=True)[1]//self.args["increment"]
            
            ts_prompts_1=self._network.ts_prompts_1
            P1=torch.cat([ts_prompts_1[j].weight.detach().clone().unsqueeze(0) for j in m],dim=0)
            gen_p.append(P1)
            ts_prompts_2=self._network.ts_prompts_2
            P2=torch.cat([ts_prompts_2[j].weight.detach().clone().unsqueeze(0) for j in m],dim=0)
            gen_p.append(P2)
            
            with torch.no_grad():
                out_logits=self._network(inputs,gen_p,train=False)
            
            preds=torch.max(out_logits, dim=1)[1]
            
            logits_preds=torch.max(out_logits, dim=1)[1]
            cor+=preds.eq(targets.expand_as(preds)).cpu().sum().numpy()
            predicts = torch.topk(out_logits, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            
            faa_pred.append(preds.cpu().numpy())
            faa_y_true.append(targets.cpu().numpy())
            
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            total+=len(targets)
        faa_pred=np.concatenate(faa_pred)
        faa_y_true=np.concatenate(faa_y_true)
        faa_tempacc=[]
        for class_id in range(0, np.max(faa_y_true), self.args["increment"]):
            idxes = np.where(np.logical_and(faa_y_true >= class_id, faa_y_true < class_id + self.args["increment"]))[0]
            faa_tempacc.append(np.around((faa_pred[idxes] == faa_y_true[idxes]).sum() * 100 / len(idxes), decimals=3))
        
        self.faa_accuracy_table.append(faa_tempacc)
        
        acctable = np.zeros([self._cur_task + 1, self._cur_task + 1])

        for idxx, line in enumerate(self.faa_accuracy_table):
            idxy = len(line)
            acctable[idxx, :idxy] = np.array(line)
        
        acctable = acctable.T
        
        forgetting = np.mean((np.max(acctable, axis=1) - acctable[:, self._cur_task])[:self._cur_task])
        
        self.acc.append(np.around(cor*100 / total, decimals=2))
        print("######################################")
        print("Last-acc:{}".format(self.acc[-1]))
        print("Avg-acc:{:.3f}".format(np.mean(self.acc)))
        print("FF: {}".format(np.around(forgetting, decimals=2)))
        print("test acc:{}".format(self.acc))
        print("FF table Last:{}".format(acctable[:,-1]))
        print("FF table:")
        print(acctable)
        print("################## next run ####################")
        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def normal_eval_cnn(self,loader):
        self._network.eval()
        
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                logits = self._network(inputs)
                
            predicts = torch.topk(logits, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]
