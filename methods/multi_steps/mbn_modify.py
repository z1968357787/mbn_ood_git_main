from methods.multi_steps.finetune_il import Finetune_IL
from backbone.mbn_net_ori_refer_adapter import mbn_net_ori_refer_adapter
from utils.toolkit import count_parameters, tensor2numpy
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import torch
import numpy as np
from argparse import ArgumentParser
from torch.nn import functional as F

def add_special_args(parser: ArgumentParser)->ArgumentParser:
    parser.add_argument('--epochs_finetune', type=int, default=None, help='The number of epochs for finetuning(Alignement stage)')
    parser.add_argument('--lrate_finetune', type=float, default=None, help='The learning rate for finetuning(Alignement stage)')
    # parser.add_argument('--milestones_finetune', type=list, default=None, help='The milestones for finetuning(Alignement stage)')   
    parser.add_argument('--milestones_finetune', nargs='+', type=int, default=None, help='for multi step learning rate decay scheduler')
    return parser

class mbn_modify(Finetune_IL):
    def __init__(self, logger, config):
        super().__init__(logger, config)
        
        self._epochs_finetune = config.epochs_finetune
        self._lrate_finetune = config.lrate_finetune
        self._milestones_finetune = config.milestones_finetune
        
        
    def prepare_model(self, checkpoint=None):
        if self._network == None:
            self._network = mbn_net_ori_refer_adapter(self._logger, self._config.backbone, self._config.pretrained, self._config.pretrain_path)
        self._network._cur_task = self._cur_task
        
        self._network.update_fc(self._total_classes)  #
        
        # self._network.freeze_previous_FE_HEAD() #freeze all the previous bn
        self._network.freeze_previous_FE() #freeze all the previous bn
        
        self._network = self._network.cuda()
        self.set_train_bn_head() 
        
        self._network = self._network.cuda()

    def set_train_bn_head(self): #only bn and head are trainable
        #if change backbone, change this function
        cur_fe = self._network.feature_extractors[-1]
        cur_fc = self._network.aux_fc
        if isinstance(self._network, nn.DataParallel):
            cur_fe.module.train()
            cur_fc.module.train()
        else:
            cur_fe.train()
            cur_fc.train()
        #需要设置对应的feature extractor为train，而不是全部的self._network为train()
        if "resnet" in self._backbone:
            for name, param in cur_fe.named_parameters():
                if "bn" in name or "downsample.1" in name or "conv1.1" in name: #they are both BN
                    self._logger.info(name)
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            for name, param in cur_fc.named_parameters():
                self._logger.info(name)
                param.requires_grad = True
        elif "efficientnet_b0" in self._backbone:
            for name, param in cur_fe.named_parameters():
                if "bn" in name: 
                    self._logger.info(name)
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            for name, param in cur_fc.named_parameters():
                self._logger.info(name)
                param.requires_grad = True

                
    def set_all_train_head(self): #only head is trainable

        for i in range(self._cur_task + 1):
            
            cur_fe = self._network.feature_extractors[i]
            cur_fc = self._network.seperate_fcs[i]
            cur_fe.eval()
            cur_fc.train()
            if self._backbone == "resnet18":
                for name, param in cur_fe.named_parameters():
                    param.requires_grad = False
                for name, param in cur_fc.named_parameters():
                    self._logger.info(name)
                    param.requires_grad = True

        #需要设置对应的feature extractor为train，而不是全部的self._network为train()
        

    
    def store_samples(self):
        self._network.eval()  #存储样本的时候，设置为eval模式
        if self._memory_bank is not None:
            self._memory_bank.store_samples(self._sampler_dataset, self._network)
            
    def prepare_task_data(self, data_manager):
        self._cur_task += 1
        self._cur_classes = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self._cur_classes

        if self._cur_task > 0:
            self._train_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes), 
                    source='train', mode='train', appendent=self._memory_bank.get_memory()) #直接用memory的话，就是这个
        else:
            self._train_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                    source='train', mode='train')
        self._test_dataset = data_manager.get_dataset(indices=np.arange(0, self._total_classes), source='test', mode='test')
        self._openset_test_dataset = data_manager.get_openset_dataset(known_indices=np.arange(0, self._total_classes), source='test', mode='test')
        #从测试集之外，拿出来除了已学习的任务外的dataset，例如已学习了0-19类，则拿出来20-99类的样本（总类别数 - 已学习的类别数）
        self._logger.info('Train dataset size: {}'.format(len(self._train_dataset)))
        self._logger.info('Test dataset size: {}'.format(len(self._test_dataset)))

        self._train_loader = DataLoader(self._train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers, pin_memory= True, persistent_workers=True)
        self._test_loader = DataLoader(self._test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers, pin_memory= True, persistent_workers=True)
        self._openset_test_loader = DataLoader(self._openset_test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
        self._sampler_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                    source='train', mode='test')
        
    def incremental_train(self):
        if self._gpu_num > 1:
            self._network = nn.DataParallel(self._network, list(range(self._gpu_num)))
        self._logger.info('-'*10 + ' Learning on task {}: {}-{} '.format(self._cur_task, self._known_classes, self._total_classes-1) + '-'*10)
        optimizer = self._get_optimizer(filter(lambda p: p.requires_grad, self._network.parameters()), self._config, self._cur_task==0)
        scheduler = self._get_scheduler(optimizer, self._config, self._cur_task==0)
        self._is_training_bn = True #use this to judge whether it is the first stage or not( first stage: train bn and fc, second stage: alignement fcs only)

        self._network.train_bn_mode()
        
        if self._cur_task == 0:
            epochs = self._init_epochs
        else:
            epochs = self._epochs
            
        self._network = self._train_model(self._network, self._train_loader, self._test_loader, optimizer, scheduler,
            task_id=self._cur_task, epochs=epochs, note='stage1')
        self._is_training_bn = False
        
        #以上是训练完了的模式
        if self._cur_task == 0:
            self._network.seperate_fcs[0].load_state_dict(self._network.aux_fc.state_dict())
            self._is_training_bn = False
        else:
            ### stage 2: Retrain fc (begin) ###
            self._logger.info('Retraining fc with the balanced dataset!')
            finetune_train_dataset = self._memory_bank.get_unified_sample_dataset(self._train_dataset, self._network)
            finetune_train_loader = DataLoader(finetune_train_dataset, batch_size=self._batch_size,
                                            shuffle=True, num_workers=self._num_workers, pin_memory= True, persistent_workers=True)
            self._network.test_mode()

            self._network.freeze_bn(mode="all") #freeze all bn, only fcs
            
            ft_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self._network.parameters()), lr=self._lrate_finetune)
            ft_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=ft_optimizer, milestones=self._milestones_finetune, gamma=self._config.lrate_decay)
            
            for name, param in self._network.named_parameters():
                if param.requires_grad:
                    self._logger.info('second stage {} require grad!'.format(name))

            self._network = self._train_model(self._network, finetune_train_loader, self._test_loader, ft_optimizer, ft_scheduler, 
                task_id=self._cur_task, epochs=self._epochs_finetune, note='stage2') #直接用所有样本的测试集来测试，而不是用什么balanced的
        
               #从而让测试的时候，是所有任务的测试数据 + 所有任务训练得到的模型做测试
               
        
        if self._gpu_num > 1:
            self._network = self._network.module
    
    
    def min_others_test(self, logits, targets, task_id): #need to modify，using metric to get final prediction

        unknown_scores = []
        known_scores = []
        cnn_max_scores = []
        known_class_num, total_class_num = 0, 0
        task_id_targets = torch.zeros(targets.shape[0], dtype=int).cuda()
        for id, cur_class_num in enumerate(self._increment_steps[:task_id+1]): #find every head prediction result
            total_class_num += cur_class_num + 1
            task_logits = logits[:,known_class_num:total_class_num] #including k+1 class
            task_scores = torch.softmax(task_logits, dim=1)

            unknown_scores.append(task_scores[:, -1])
            
            known_task_scores = torch.zeros((targets.shape[0], max(self._increment_steps))).cuda()
            known_task_scores[:, :(task_scores.shape[1]-1)] = task_scores[:, :-1] #exclude k+1 class
            known_scores.append(known_task_scores)

            # generate task_id_targets
            task_data_idxs = torch.argwhere(torch.logical_and(targets>=known_class_num-id, targets<total_class_num-id-1)).squeeze(-1)
            if len(task_data_idxs) > 0:
                task_id_targets[task_data_idxs] = id

            known_class_num = total_class_num

        known_scores = torch.stack(known_scores, dim=0) # task num, b, max(task_sizes)
        unknown_scores = torch.stack(unknown_scores, dim=-1) # b, task num
    
        min_scores, task_id_predict = torch.min(unknown_scores, dim=-1) #find task id prediction result within the batch
        # cnn_max_scores = 1 - min_scores  #different  from mbn_ori_refer_to_adapter
        ###
        
 #need to modify，using metric to get final prediction
        
            ###
                ### predict class based on task id and known scores ###
        cnn_preds = torch.zeros(targets.shape[0], dtype=int).cuda()
        cnn_max_scores = torch.zeros(targets.shape[0]).cuda()
        
        known_class_num, total_class_num = 0, 0
        for id, cur_class_num in enumerate(self._increment_steps[:task_id+1]): #检索每一个任务id，
            total_class_num += cur_class_num # do not have others category !
            task_logits_idxs = torch.argwhere(task_id_predict==id).squeeze(-1) #找到任务标识预测为id的样本
            if len(task_logits_idxs) > 0:
                cnn_preds[task_logits_idxs] = torch.argmax(known_scores[id, task_logits_idxs], dim=1) + known_class_num 
                #make those with task id prediction correctly sample, add bias to get the final prediction
                cnn_max_scores[task_logits_idxs] = known_scores[id, task_logits_idxs].max(dim=1).values
            known_class_num = total_class_num

        task_id_correct = task_id_predict.eq(task_id_targets).cpu().sum()

        return cnn_preds, cnn_max_scores, task_id_correct

    
    def _epoch_test(self, model, test_loader, ret_task_acc=False, ret_pred_target=False, task_begin=None, task_end=None, task_id=None):
        cnn_correct, cnn_task_correct, total, task_total = 0, 0, 0, 0
        task_id_correct = 0
        cnn_pred_all, target_all = [], []
        cnn_max_scores_all = []
        features_all = []
        for _, inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            # model forward has two mode which shoule be noticed before forward!
            if self._is_training_bn: #only focus on the current bn(current feature extractor ), output is : batch_size * (class in the current task + 1)

                aux_logits, feature_outputs  = model(inputs)
                
                targets = torch.where(targets-task_begin+1>0, targets, task_end) #consider ood to be K+1 class
                cnn_preds = torch.argmax(aux_logits, dim=1) + task_begin
                
            else: #test based on all the trained bn and fc (forward_all model), output: batch_size * (n * 任务数量 + n)
                # task_test_acc 有意义(反映当前task的预测准确率), test_acc 有意义(反映模型最终的预测结果)
                ### predict task id based on the unknown scores ###
                aux_logits, feature_outputs  = model.forward_all_model(inputs)
                
                cnn_preds, cnn_max_scores, task_id_pred_correct= self.min_others_test(logits=aux_logits, targets=targets, task_id=task_id) #need to modify
                task_id_correct += task_id_pred_correct #aux_logits: batch_size * (n * task_num + n), 因此targets无需修改。

            if ret_pred_target: # only apply when self._is_training_bn is True
                cnn_pred_all.append(tensor2numpy(cnn_preds))
                target_all.append(tensor2numpy(targets))
                cnn_max_scores_all.append(tensor2numpy(cnn_max_scores))
                features_all.append(tensor2numpy(feature_outputs['features']))
            else:#training bn is True
                if ret_task_acc:
                    task_data_idxs = torch.argwhere(torch.logical_and(targets>=task_begin, targets<task_end)) 
                    #task_begin:0, task_end:10 , normal class: 0-9, ood class:10 ;only focus on 0-9, do not care about 10
                    cnn_task_correct += cnn_preds[task_data_idxs].eq(targets[task_data_idxs]).cpu().sum() 
                    task_total += len(task_data_idxs)
                cnn_correct += cnn_preds.eq(targets).cpu().sum()
            
            # for print out task id predict acc
            total += len(targets)
        
        if not self._is_training_bn:
            self._logger.info('Test task id predict acc (CNN) : {:.2f}'.format(np.around(task_id_correct*100 / total, decimals=2)))

        if ret_pred_target:
            cnn_pred_all = np.concatenate(cnn_pred_all)
            target_all = np.concatenate(target_all)
            cnn_max_scores_all = np.concatenate(cnn_max_scores_all)
            features_all = np.concatenate(features_all)
            return cnn_pred_all, None, cnn_max_scores_all, None, target_all, features_all
        else:
            test_acc = np.around(tensor2numpy(cnn_correct)*100 / total, decimals=2)
            if ret_task_acc:
                test_task_acc = np.around(tensor2numpy(cnn_task_correct)*100 / task_total, decimals=2)
                return test_acc, test_task_acc #test_acc: with k+1, test_task_acc: without k+1
            else:
                return test_acc
    
    def _epoch_train(self, model, train_loader, optimizer, scheduler, task_begin=None, task_end=None, task_id=None):
        losses = 0. #训练fc和训练
        correct, total = 0, 0
        ce_losses = 0.
        if self._is_training_bn:
            model.new_bn_train()
        else:
            model.eval()
        
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            
            if self._is_training_bn: #first stage, only focus on train the current bn and head

                
                targets = torch.where(targets-task_begin+1>0, targets, task_end) #in the first stage, consider the ood class in the memory to be k+1
                
                aux_logits, features = model(inputs)
                
                loss = F.cross_entropy(aux_logits, targets-task_begin)


                preds = torch.argmax(aux_logits, dim=1) + task_begin
                correct += preds.eq(targets).cpu().sum() 
                #including the correct of k+1 class
                
            else: #train all the fcs
                
                loss = torch.tensor(0.0)
                aux_logits, features = model.forward_all_model(inputs)
                
                known_class_num, total_class_num = 0, 0
                for id, cur_class_num in enumerate(self._increment_steps[:task_id+1]): #for each batch, train every fc as alignment
                    total_class_num += cur_class_num + 1
                    task_logits = aux_logits[:,known_class_num:total_class_num]
                    
                    task_targets = (torch.ones(targets.shape[0], dtype=int) * cur_class_num).cuda() # class label: [0, cur_class_num]
                    task_data_idxs = torch.argwhere(torch.logical_and(targets>=known_class_num-id, targets<total_class_num-id-1)).squeeze(-1)
                    # current_task: 11-20
                    # task_targets:[10,10,10]
                    # targets : [9, 11 ,14]
                    # task_data_idxs: [0]
                    # task_targets will become：[10, 1, 4], task targets map the original targets to the current task, ood as k+1
                    if len(task_data_idxs) > 0:
                        task_targets[task_data_idxs] = targets[task_data_idxs] - known_class_num + id
                        loss = loss + F.cross_entropy(task_logits, task_targets)
                        #loss acummulate
                    if id == task_id: # 
                        preds = torch.argmax(aux_logits[:,known_class_num:total_class_num], dim=1)

                    known_class_num = total_class_num
                
                ce_losses += loss.item()
                #因此在外面只需要计算loss，它叠加了所有分类头的分类loss
                aux_targets = torch.where(targets-task_begin+1>0, targets-task_begin, task_end - task_begin)
                correct += preds.eq(aux_targets).cpu().sum()
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            
            total += len(targets)
    
        if scheduler != None:
            scheduler.step()
            
        train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
        train_loss = ['Loss', losses/len(train_loader), 'ce_loss', ce_losses/len(train_loader)]
        return model, train_acc, train_loss
    