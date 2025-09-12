# 参考adapter_cl_net，通过魔改的方式来获得多BN
# adapter_cl_net： register_hook来获得多BN的效果。
# adapter_cl_net： register_hook的方式并不适用，因为register hook是在每个nn.module进行注册
# 而本方法是在nn.module内部，改每个bn层。所以说直接修改basic block，bottleneck block的内容，会更好。
# 也就是修改feature extractor的部分
# 那么如何随着新任务的到来不断地添加BN呢？
# 第一种方式：
#     训练时候：
#         随着任务的不断到来的时候，register pre hook，把block内部的涉及bn的地方都进行新添加
#         每次forward的时候，就是用最新的来进行forward
#     测试时候：
        #循环如果在外面做的话，那么每设置一个t，就需要前向一次。
        #循环如果在里面，那么只需要前向一次，就能够得到多个bn作用后的结果了
        # 像adapter这样做的话，那么问题在于要设置module里面的内容，而不是像他这样，直接作用于input

#如果一次前向，想获得多个bn的结果的话，不太可能。因为我是要修改每个block内部的bn的
#而acl的话，是固定了backbone的，所以说可以获得多个adapte的结果，直接通过adapter作用于backbone即可。
#因此，第一种方式是不能实现的

# 而第二种方式，通过魔改backbone的方式的话，那么关键点在于：
# 可以提升前向速度，但是：载入预训练的部分，就需要修改。除非的话除了key一样的部分，其他的都不载入
        
from typing import Callable, Iterable
from backbone.inc_net import IncrementalNet
from torch import nn
import timm.models as timm_models
import timm
import torch
import torch.nn.functional as F

class CNN_MBN_Net_CIL_V2(nn.Module):  #不继承incremental_net
    def __init__(self, config, cur_task, logger, backbone_type, pretrained=True, pretrain_path=None):
        super(CNN_MBN_Net_CIL_V2, self).__init__() 
        self.task_sizes = []
        self._logger = logger 
        self._backbone = backbone_type
        self._cur_task = cur_task
        self.forward_batch_size = None
        # get_backbone(self._logger, backbone_type, pretrained, pretrain_path)
        self.feature_extractor = self.get_ft(config)
        if pretrained and pretrain_path is None: #从默认的地方进行pretrain
            self.load_pretrain(config)
        # self.get_backbone(config, backbone_type, pretrained, pretrained_path)
        if backbone_type =='resnet18':
            self.feature_dim = 512 
        self.seperate_fc = nn.ModuleList()
    
    def load_pretrain(self, config):
        if config.backbone == "resnet18":
            import timm.models as timm_models
            
            timm_resnet18 = timm_models.create_model('resnet18',pretrained=True)
            modified_state_dict = {}
            
            for key, value in timm_resnet18.state_dict().items():
                if "fc" in key:
                    continue
                if "layer" in key and "bn" in key and "downsample" not in key:
                    # layer1.0.bn1.weight ->  'layer1.0.bn1.task_0.weight'
                    split_list = key.split(".")
                    assert len(split_list) == 4
                    # split_list.insert(-1, "task_0")
                    for i in range(config.nb_tasks):  #use pretrain model to init every bn
                        split_list.insert(-1, f"task_{i}")
                            
                        now_key = ".".join(split_list)
                        
                        modified_state_dict[now_key] = value
                        split_list.pop(-2)
                
                elif "bn" in key and "layer" not in key and "downsample" not in key:
                    # bn1.weight  -> bn1.task_0.weight
                    split_list = key.split(".")
                    # split_list.insert(-1, "task_0")
                    
                    for i in range(config.nb_tasks):   #use pretrain model to init every bn
                        split_list.insert(-1, f"task_{i}")
                            
                        now_key = ".".join(split_list)
                        modified_state_dict[now_key] = value
                        split_list.pop(-2)
                    
                elif "downsample" in key and "bn" not in key and "layer" in key:
                    # layer2.0.downsample.0.weight -> layer2.0.downsample.conv1.weight
                    # layer2.0.downsample.1.weight -> layer2.0.downsample.bn1.task_0.weight
                    split_list = key.split(".")
                    if split_list[-2] == "0": #convolution layer
                        split_list[-2] = "conv1"
                        now_key = ".".join(split_list)
                        modified_state_dict[now_key] = value
                        #not need to pop
                    elif split_list[-2] == "1": #bn layer
                        split_list[-2] = "bn1"
                        # split_list.insert(-1, "task_0")
                        for i in range(config.nb_tasks):
                            split_list.insert(-1, f"task_{i}")
                                
                            now_key = ".".join(split_list)
                            modified_state_dict[now_key] = value
                            split_list.pop(-2)
                            
                            modified_state_dict[now_key] = value
                else:
                    modified_state_dict[key] = value
                                
            mbn_cl_net_keys = set(self.feature_extractor.state_dict().keys())
            modified_state_dict_keys = set(modified_state_dict.keys())
            difference_keys = mbn_cl_net_keys.difference(modified_state_dict_keys)
            print(difference_keys)
            self.feature_extractor.load_state_dict(modified_state_dict)
            print("load pretrain model successfully")
                # Check if all values are loaded




    def generate_fc(self, in_dim, out_dim):
        return nn.Linear(in_dim, out_dim)
    
    def update_fc(self, nb_classes):  #bn这些，是初始化的时候就弄的
        #但是fc的话， 是
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self._cur_task = len( self.task_sizes) -1  #在这里更新cur_task
        self.aux_fc = self.generate_fc(self.feature_dim, new_task_size+1) #重新生成一个aux_fc
        self._logger.info('Created aux others classifier head: {} => {}'.format(self.feature_dim, new_task_size+1))

        self.seperate_fc.append(self.generate_fc(self.feature_dim, new_task_size+1)) #加到self.seperate_fc里面去
    #而seperate_fc是第二阶段训练各个分类头的时候使用的。
    
    def freeze_bn(self, mode): #freeze all the bn, only head is trainable
        if mode == "all":
            cur_fe = self.feature_extractor
            for name, param in cur_fe.named_parameters():
                    param.requires_grad = False
            cur_fe.eval()
    
    def set_train_bn_head(self): #only bn and head are trainable
        #if change backbone, change this function
        if self._backbone == "resnet18": #冻结过去的
            for name, param in self.feature_extractor.named_parameters():
                if "bn" in name and f"task_{self._cur_task}" in name: 
                    self._logger.info(name)
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            for name, param in self.aux_fc.named_parameters(): #训练的时候，训练的是aux_fc
                param.requires_grad = True
                
                print(name)
            #     if "bn" in name or "downsample.1" in name or "conv1.1" in name: 
            #         self._logger.info(name)
            #         param.requires_grad = True
            #     else:
            #         param.requires_grad = False
            # for name, param in cur_fc.named_parameters():
            #     self._logger.info(name)
            #     param.requires_grad = True
            

    def new_bn_train(self):
        self.eval()  # 将整个模型设置为评估模式
        if self._backbone == "resnet18":
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and f"task_{self._cur_task}" in name:
                    module.train()  # 只有当前任务的BN层设置为训练模式
                    # print("trainable mode" + name)

    
    def get_ft(self, config):
        if config.backbone == 'resnet18':
            feature_extractor = ResNet18(config)
        return feature_extractor
    
    
    # def get_backbone(self, logger, backbone_type, pretrained, pretrain_path):
    #     if backbone_type == 'resnet18':
    #         feature_extractor = ResNet18(P)
    #     return feature_extractor
    
    
    def forward(self, x, task_id = None):
        # self.forward_batch_size = x.shape[0]
        output_features = {}
        if task_id == None:
            task_id = len(self.task_sizes) - 1   #默认使用最新的BN
        features = self.feature_extractor(x, task_id)
        out = self.aux_fc(features) #在训练特征提取器的时候，用了aux_fc，和seperate_fc不是同一个东西

        output_features['features'] = features
        return out, output_features

    def forward_all_model(self, x):  #forward test是：使用所有的任务对应的模型进行前向
        self.forward_batch_size = x.shape[0]
        feature_list = []
        out_list = []
        for task_id in range(self._cur_task + 1):
            task_features = self.feature_extractor(x, task_id)
            task_logits = self.seperate_fc[task_id](task_features)
            feature_list.append(task_features)
            out_list.append(task_logits)
        return torch.cat(out_list, dim=1), {"features": torch.cat(feature_list, dim=1)} #必须统一通过这种形式包裹
        



def conv3x3(in_planes, out_planes, stride=1): #默认kernel_size是3的
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class Downsample(nn.Module):
    def __init__(self, Config, stride, in_planes, expansion, planes):
        super(Downsample, self).__init__()
        self.identity = True
        self.downsample = nn.Sequential()

        if stride != 1 or in_planes != expansion*planes:
            self.identity = False
            self.conv1 = nn.Conv2d(in_planes, expansion*planes, kernel_size=1, stride=stride, bias=False)
            # self.bn1 = nn.ModuleList()

            self.bn1 = nn.ModuleDict()
            for i in range(Config.nb_tasks):
                self.bn1[f'task_{i}'] = nn.BatchNorm2d(expansion*planes)
                
    def forward(self, x, t):
        if self.identity:
            out = self.downsample(x)
        else:
            out = self.conv1(x)
            out = self.bn1[f'task_{t}'](out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, Config, in_planes, planes, stride=1, pooling=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        # self.bn1 = nn.ModuleList()
        self.bn1 = nn.ModuleDict()
        self.bn2 = nn.ModuleDict()
        
        for t in range(Config.nb_tasks): #不是通过动态控制的方式，而是初始化的时候，就设定了所有的bn
            # self.bn1.append(nn.BatchNorm2d(planes))
            self.bn1[f'task_{t}'] = nn.BatchNorm2d(planes)
            self.bn2[f'task_{t}'] = nn.BatchNorm2d(planes)


        self.downsample = Downsample(Config, stride, in_planes, self.expansion, planes)

        self.pooling = pooling
        
    def forward(self, x, t):

        out = F.relu(self.bn1[f'task_{t}'](self.conv1(x)))

        out = self.bn2[f'task_{t}'](self.conv2(out))
        out += self.downsample(x, t)
        out = F.relu(out)

        if self.pooling:
            out = F.avg_pool2d(out, 4)

        return out

class ResNet(nn.Module):
    def __init__(self, Config, block, num_blocks):
        # if 'cifar10' in Config.dataset and 'cifar100' not in Config.dataset:
        #     init_dim, last_dim = 64, 512 # last_dim = init_dim * 8
        # elif 'cifar100' in Config.dataset or 'tinyImagenet' in Config.dataset:
            # init_dim, last_dim = 128, 1024
        if Config.backbone == "resnet18":
            init_dim, last_dim = 64, 512
        self.in_planes = init_dim
        # super(ResNet, self).__init__(last_dim, Config)
        super(ResNet, self).__init__()
        
        # self.conv1 = conv3x3(3, init_dim)
        if "cifar100" in Config.dataset: #现在都是转为224*224的了
            self.conv1 = nn.Conv2d(3, init_dim, kernel_size=7, stride=2, padding=3, bias=False)

            self.bn1 = nn.ModuleDict()
            
            for i in range(Config.nb_tasks):
                self.bn1[f'task_{i}'] = nn.BatchNorm2d(init_dim)
        
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        
        self.layer1 = self._make_layer(Config, block, init_dim, num_blocks[0], stride=1, pooling=False)
        self.layer2 = self._make_layer(Config, block, init_dim * 2, num_blocks[1], stride=2, pooling=False) 
        self.layer3 = self._make_layer(Config, block, init_dim * 4, num_blocks[2], stride=2, pooling=False) 
        self.layer4 = self._make_layer(Config, block, last_dim, num_blocks[3], stride=2, pooling=True) 
    
    def _make_layer(self, Config, block, planes, num_blocks, stride, pooling=False):
        strides = [stride] + [1]*(num_blocks-1)
        pooling_ = False
        layers = nn.ModuleList()
        for i, stride in enumerate(strides):
            if i == len(strides) - 1:
                pooling_ = pooling
            layers.append(block(Config, self.in_planes, planes, stride, pooling_))
            self.in_planes = planes * block.expansion
        return layers

    def forward(self, x, t = None): #是不是只要底层的实现， 是需要传t的，那么forward的时候就传就可以了？
        if t is None:
            t = self._cur_task
        x = self.conv1(x)
        x = self.bn1[f"task_{t}"](x)
        x = F.relu(x)
        x = self.maxpool(x) #conv, bn, relu, maxpool for 224*224 images
        
        for op in self.layer1:
            x = op( x, t)
        for op in self.layer2:
            x = op( x, t)
        for op in self.layer3:
            x = op( x, t)
        for op in self.layer4:
            x = op( x, t)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


def ResNet18(Config):
    return ResNet(Config, BasicBlock, [2,2,2,2])


