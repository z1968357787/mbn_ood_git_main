import os
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix, roc_curve, auc, average_precision_score
import diffdist.functional as distops
import torch.distributed as dist
# import transform_layers as TL
import backbone.clom_transform_layers as TL
# import transform_layers as TL
from torch.utils.data import Sampler


def get_similarity_matrix(outputs, chunk=2, multi_gpu=False):
    '''
        Compute similarity matrix
        - outputs: (B', d) tensor for B' = B * chunk
        - sim_matrix: (B', B') tensor
    '''

    if multi_gpu:
        outputs_gathered = []
        for out in outputs.chunk(chunk):
            gather_t = [torch.empty_like(out) for _ in range(dist.get_world_size())]
            gather_t = torch.cat(distops.all_gather(gather_t, out))
            outputs_gathered.append(gather_t)
        outputs = torch.cat(outputs_gathered)

    sim_matrix = torch.mm(outputs, outputs.t())  # (B', d), (d, B') -> (B', B')

    return sim_matrix

def get_simclr_augmentation(Config, image_size): #place here

    # parameter for resizecrop
    resize_scale = (Config.resize_factor, 1.0) # resize scaling factor
    if Config.resize_fix: # if resize_fix is True, use same scale
        resize_scale = (Config.resize_factor, Config.resize_factor)

    # Align augmentation
    color_jitter = TL.ColorJitterLayer(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8)
    color_gray = TL.RandomColorGrayLayer(p=0.2)
    resize_crop = TL.RandomResizedCropLayer(scale=resize_scale, size=image_size)

    transform = nn.Sequential(
        color_jitter,
        color_gray,
        resize_crop,
    )

    return transform


class WeightLoss(nn.Module):
    def __init__(self, target_norm):
        super(WeightLoss, self).__init__()
        self.target_norm = target_norm

    def forward(self, weights):
        # 计算权重向量的L2范数
        norm_diff = torch.norm(weights, p=2, dim=1) - self.target_norm
        # weighted_norm_diff = norm_diff * torch.norm(weights, p=2, dim=1)
        norm_differences_squared = torch.square(norm_diff)
        return norm_differences_squared.mean()

def Supervised_NT_xent(sim_matrix, labels, temperature=0.5, chunk=2, eps=1e-8, multi_gpu=False):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''

    device = sim_matrix.device

    if multi_gpu:
        gather_t = [torch.empty_like(labels) for _ in range(dist.get_world_size())]
        labels = torch.cat(distops.all_gather(gather_t, labels))
    labels = labels.repeat(2)

    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    sim_matrix = sim_matrix - logits_max.detach()

    B = sim_matrix.size(0) // chunk  # B = B' / chunk

    eye = torch.eye(B * chunk).cuda()  # (B', B')
    sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)  # remove diagonal

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)
    sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)  # loss matrix

    labels = labels.contiguous().view(-1, 1)
    Mask = torch.eq(labels, labels.t()).float().cuda()
    #Mask = eye * torch.stack([labels == labels[i] for i in range(labels.size(0))]).float().cuda()
    Mask = Mask / (Mask.sum(dim=1, keepdim=True) + eps)
    loss = torch.sum(Mask * sim_matrix) / (2 * B)
    return loss

def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad) #可训练的参数的数量
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if hasattr(x, 'is_cuda') and x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).cuda()
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.)
    return onehot


def check_makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, increment):
    assert len(y_pred) == len(y_true), 'Data length error.'
    total_acc = np.around((y_pred == y_true).sum()*100 / len(y_true), decimals=2)
    known_classes = 0
    task_acc_list = []

    # Grouped accuracy
    for cur_classes in increment:
        idxes = np.where(np.logical_and(y_true >= known_classes, y_true < known_classes + cur_classes))[0]
        task_acc_list.append(np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes), decimals=2))
        known_classes += cur_classes
        if known_classes >= nb_old:
            break

    return total_acc, task_acc_list #total_acc: cil total acc of all tasks, task acc_list: cil acc of all tasks


def cal_openset_test_metrics(pred_max_score, y_true):
    fpr, tpr, thresholds = roc_curve(y_true, pred_max_score)
    roc_auc = auc(fpr, tpr)
    fpr95_idx = np.where(tpr>=0.95)[0]
    fpr95 = fpr[fpr95_idx[0]]

    ap = average_precision_score(y_true, pred_max_score)
    return roc_auc*100, fpr95*100, ap*100


def cal_ece(y_pred, pred_max_score, y_true, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins+1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    accuracies = (y_pred==y_true)
    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # calculate |confidence - accuracy| in each bin
        in_bin = (pred_max_score>bin_lower.item()) * (pred_max_score <= bin_upper.item())
        prop_in_bin = in_bin.astype(float).mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].astype(float).mean()
            avg_confidence_in_bin = pred_max_score[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()

def cal_class_avg_acc(y_pred, y_true):
    class_acc_list = []
    for class_id in np.unique(y_true):
        class_mask = np.where(y_true==class_id)[0]
        class_acc_list.append((y_pred[class_mask]==class_id).sum() / len(class_mask))
    return np.mean(class_acc_list)*100

def mean_class_recall(y_pred, y_true, nb_old, increment):
    assert len(y_pred) == len(y_true), 'Data length error.'
    total_mcr = cal_mean_class_recall(y_pred, y_true)
    known_classes = 0
    task_mcr_list = []

    # Grouped accuracy
    for cur_classes in increment:
        idxes = np.where(np.logical_and(y_true >= known_classes, y_true < known_classes + cur_classes))[0]
        task_mcr_list.append(cal_mean_class_recall(y_pred[idxes], y_true[idxes], cur_classes))
        known_classes += cur_classes
        if known_classes >= nb_old:
            break
    return total_mcr, task_mcr_list

def cal_mean_class_recall(y_pred, y_true, task_size=None):
    """ Calculate the mean class recall for the dataset X 
    Note: should take task_size input seriously when calculating task-specific MCR !!!
    """
    cm = confusion_matrix(y_true, y_pred)
    right_of_class = np.diag(cm)
    num_of_class = cm.sum(axis=1)
    if task_size is None:
        task_size = cm.shape[0]
    # Can not use (right_of_class*100 / (num_of_class+1e-8)).mean() to calculate !!!
    # This is WRONG for task-specific MCR !!!
    mcr = np.around((right_of_class*100 / (num_of_class+1e-8)).sum() / task_size, decimals=2)
    return mcr

def md(data, mean, mat, inverse=False):
    if data.ndim == 1:
        data.reshape(1, -1)
    delta = (data - mean)

    if not inverse:
        mat = np.linalg.inv(mat)

    dist = np.dot(np.dot(delta, mat), delta.T)
    return np.sqrt(np.diagonal(dist)).reshape(-1, 1)

def cal_bwf(task_metric_curve, cur_task):
    """cur_task in [0, T-1]"""
    bwf = 0.
    if cur_task > 0:
        for i in range(cur_task):
            task_result = 0.
            for j in range(cur_task-i):
                task_result += task_metric_curve[i][cur_task - j] - task_metric_curve[i][i]
            bwf += task_result / (cur_task-i)
        bwf /= cur_task
    return bwf

def cal_avg_forgetting(task_metric_curve, cur_task):
    """cur_task in [0, T-1]"""
    avg_forget = 0.
    if cur_task > 0:
        avg_forget = (task_metric_curve[:cur_task, :cur_task].max(axis=1) - task_metric_curve[:cur_task, cur_task]).mean()
    return avg_forget

def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)

def pil_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    '''
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class MoreSampler(Sampler):
    """
        Sampler for dataset whose length is different from that of the target dataset.
        This can be particularly useful when we need oversampling/undersampling because
        the target dataset has more/less samples than the dataset of interest.
        Generate indices whose length is same as that of target length.
    """
    def __init__(self, current_length, target_length):
        self.current = current_length
        self.target = target_length

    def __iter__(self): #将current_length采样到和target_length一样长。
        self.indices = np.array([], dtype=int)
        while len(self.indices) < self.target:
            idx = np.random.permutation(self.current)
            self.indices = np.concatenate([self.indices, idx])
        self.indices = self.indices[:self.target]
        return iter(self.indices)

    def __len__(self):
        return self.target
    



class DummyDataset(Dataset):
    def __init__(self, data, targets, transform, use_path=False, two_view=False, ret_origin=False):
        assert len(data) == len(targets), 'Data size error!'
        self.data = data
        self.targets = targets
        self.ret_origin = ret_origin

        self.transform = transform
        self.use_path = use_path
        self.two_view = two_view

        self.no_transform = transforms.PILToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.use_path:
            origin_img = pil_loader(self.data[idx])
        else:
            origin_img = Image.fromarray(self.data[idx])
        image = self.transform(origin_img)
        label = self.targets[idx]
        addition_info = self.no_transform(origin_img) if self.ret_origin else idx
        if self.two_view:
            if self.use_path:
                image2 = self.transform(pil_loader(self.data[idx]))
            else:
                image2 = self.transform(Image.fromarray(self.data[idx]))
            image = [image, image2]
            
        return addition_info, image, label
    
    def set_ret_origin(self, mode:bool):
        self.ret_origin = mode

class bn_track_stats:
    def __init__(self, module: nn.Module, condition=True):
        self.module = module
        self.enable = condition

    def __enter__(self):
        if not self.enable:
            for m in self.module.modules():
                if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                    m.track_running_stats = False

    def __exit__(self ,type, value, traceback):
        if not self.enable:
            for m in self.module.modules():
                if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                    m.track_running_stats = True
                    