from backbone.inc_net import IncrementalNet
from typing import Callable, Iterable
from torch import nn
from backbone.inc_net import get_backbone
import torch

class mmd_aug_net(IncrementalNet):
    def __init__(self, logger, backbone_type, pretrained, pretrain_path=None, layer_names:Iterable[str]=[], mode=None):
        super().__init__(logger, backbone_type, pretrained, pretrain_path)
        self._logger = logger
        self.backbone_type = backbone_type
        self.feature_extractors = nn.ModuleList()
        
        self.pretrained = pretrained
        self.pretrain_path = pretrain_path
        self.out_dim = None
        self.seperate_fcs = nn.ModuleList()
        
        self.task_sizes = []
    
    def freeze_fe_fc(self): #在任务后进行，使得新添加的fc和fe才是可训练的
        for fe in self.feature_extractors:
            for param in fe.parameters():
                param.requires_grad = False
        for fc in self.seperate_fcs:
            for param in fc.parameters():
                param.requires_grad = False
        self.eval()
    
    def set_cur_train(self): #只设置当前的fe和fc为训练模式
        for fe in self.feature_extractors:
            fe.eval()
        for fc in self.seperate_fcs:
            fc.eval()
        self.feature_extractors[-1].train()
        self.seperate_fcs[-1].train()
        
    
    def update_fc_no_ood_cls(self, nb_classes): #更新分类头，但是分类头上没有ood类别，比较少用。
        new_task_size = nb_classes - sum(self.task_sizes)

        self._logger.info('Created aux others classifier head: {} => {}'.format(self.feature_dim, new_task_size))

        self.seperate_fcs.append(self.generate_fc(self.feature_dim, new_task_size)) #加到self.seperate_fc里面去
    
    def update_fc(self, nb_classes): #传入的需要是total_classes
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)

        # self.aux_fc = self.generate_fc(self.feature_dim, new_task_size+1) #others类别被写在这，注意，aux_fc是在训练第一阶段使用的
        #mbn的时候，第一阶段用的是aux_fc，仅使用训练；
        # 第二阶段用的是seperate_fc。
        #但我这里因为解耦开了第一阶段和第二阶段，所以目前都只使用seperate_fc
        self._logger.info('Created aux others classifier head: {} => {}'.format(self.feature_dim, new_task_size+1))

        self.seperate_fcs.append(self.generate_fc(self.feature_dim, new_task_size+1)) #加到self.seperate_fc里面去
        #而seperate_fc是第二阶段训练各个分类头的时候使用的。
    
    def forward(self, x):  #如果是训练的时候，那么只在当前任务所对应的模型的基础上进行前向
        features = self.feature_extractors[-1](x)

        aux_logits=self.seperate_fcs[-1](features) 
        #seperate_fc其实就是fc，只不过是多了一个logit的fc
        #
        
        return aux_logits, {"features": features} #必须统一通过这种形式包裹

    def forward_all_model(self, x):  #forward test是：使用所有的任务对应的模型进行前向
        self.forward_batch_size = x.shape[0]
        
        task_feature_list = []
        task_output_list = []
        for i in range(len(self.task_sizes)): # features: [b * number  of adapters, feature_dim]
            self.feature_extractors[i].eval()
            self.seperate_fcs[i].eval()
            task_feature = self.feature_extractors[i](x)
            task_feature_list.append(task_feature) # features: [b, feature_dim]
            task_output_list.append(self.seperate_fcs[i](task_feature))
        
        features = torch.cat(task_feature_list, dim=1) # features: [b, feature_dim * number of adapters]
        aux_logits = torch.cat(task_output_list, dim=1) # seperate logits: [b, known_class_num]
        return aux_logits, {"features": features} #必须统一通过这种形式包裹


    def append_fe(self):
        ft = get_backbone(self._logger, self.backbone_type, pretrained=self.pretrained, pretrain_path=self.pretrain_path)
        ft.fc = nn.Identity()
        
        self.feature_extractors.append(ft)
        