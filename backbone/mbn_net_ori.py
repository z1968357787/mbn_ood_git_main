import torch
import copy
from torch import nn

from backbone.inc_net import get_backbone

class mbn_net_ori(nn.Module):
    def __init__(self, logger, backbone_type, pretrained, pretrain_path=None):
        super(mbn_net_ori,self).__init__()
        self._logger = logger
        self.backbone_type = backbone_type
        self.feature_extractors = nn.ModuleList()
        
        self.pretrained = pretrained
        self.pretrain_path = pretrain_path
        self.out_dim = None
        self.aux_fcs = nn.ModuleList()
        
        self.aux_fc = None
        self.task_sizes = []

    def extract_features(self, x): #need to modify
        features = self.feature_extractors[-1](x)

        return features

    def forward(self, x):  #如果是训练的时候，那么只在当前任务所对应的模型的基础上进行前向
        features = self.feature_extractors[-1](x)

        aux_logits=self.aux_fcs[-1](features)  #aux_fc其实就是fc，只不过是多了一个logit的fc
        return aux_logits, {"features": features} #必须统一通过这种形式包裹

    def forward_all_model(self, x):  #forward test是：使用所有的任务对应的模型进行前向
        self.forward_batch_size = x.shape[0]
        
        task_feature_list = []
        task_output_list = []
        for i in range(len(self.task_sizes)): # features: [b * number  of adapters, feature_dim]
            self.feature_extractors[i].eval()
            self.aux_fcs[i].eval()
            task_feature = self.feature_extractors[i](x)
            task_feature_list.append(task_feature) # features: [b, feature_dim]
            task_output_list.append(self.aux_fcs[i](task_feature))
        
        features = torch.cat(task_feature_list, dim=1) # features: [b, feature_dim * number of adapters]
        aux_logits = torch.cat(task_output_list, dim=1) # seperate logits: [b, known_class_num]
        return aux_logits, {"features": features} #必须统一通过这种形式包裹

    def train_bn_mode(self):
        self._training_mode = 'train_bn'
        self._logger.info("Training mode 'train_bn' is set !")
        
    # def forward_test(self, x): 
    #     #如果是测试的时候，需要在所有任务对应的模型的基础上进行前向
    #     features = self.feature_extractors[-1](x)

    #     aux_logits=self.aux_fcs[-1](features)  #aux_fc其实就是fc，只不过是多了一个logit的fc
    #     return aux_logits, {"features": features} #必须统一通过这种形式包裹


    def update_fc(self, nb_classes): 
        #not only update the fc, also update the feature extractor(in fact, only train bn)
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)

        # ft = get_backbone(self._logger, self.backbone_type, pretrained=self.pretrained, pretrain_path=self.pretrain_path)
        ft = get_backbone(self._logger, self.backbone_type, pretrained = True, pretrain_path=self.pretrain_path) #预训练模型是成功载入的。
        
        if 'resnet' in self.backbone_type:
            feature_dim = ft.fc.in_features
            self._logger.info("feature dim is {}".format(feature_dim))
            ft.fc = nn.Identity()
        elif 'efficientnet' in self.backbone_type:
            feature_dim = ft.classifier[1].in_features
            ft.classifier = nn.Dropout(p=0.4, inplace=True)
        elif 'mobilenet' in self.backbone_type:
            feature_dim = ft.classifier[-1].in_features
            ft.classifier = nn.Dropout(p=0.2, inplace=False)
        else:
            raise ValueError('{} did not support yet!'.format(self.backbone_type))

        if len(self.feature_extractors)==0:
            self.feature_extractors.append(ft)
            self._feature_dim = feature_dim
        else:
            self.feature_extractors.append(ft)
            # self.feature_extractors[-1].load_state_dict(self.feature_extractors[-2].state_dict()) \
                #to be done，加载的时候，需要把BN的参数也进行加载。
        
        self.aux_fc=self.generate_fc(self._feature_dim, new_task_size + 1)
        self.aux_fcs.append(self.aux_fc)    

    def generate_fc(self, in_dim, out_dim):
        # fc = SimpleLinear(in_dim, out_dim)
        fc = nn.Linear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze_feature_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

    def reset_fc_parameters(self):
        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='linear')
        nn.init.constant_(self.fc.bias, 0)