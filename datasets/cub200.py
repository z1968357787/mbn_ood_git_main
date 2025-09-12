from torchvision import transforms
from datasets.idata import iData
import os
import numpy as np
from PIL import Image

import os
import shutil
import numpy as np

class CUB200(iData):
    '''
    Dataset Name:   CUB200-2011
    Task:           fine-grain birds classification
    Data Format:    224x224 color images. (origin imgs have different w,h)
    Data Amount:    5,994 images for training and 5,794 for validationg/testing
    Class Num:      200
    Label:          

    Reference:      https://opendatalab.com/CUB-200-2011
    '''
    def __init__(self, img_size=None) -> None:
        super().__init__()
        self.use_path = True
        self.img_size = 224 if img_size is None else img_size
        self.train_trsf = [
            transforms.RandomResizedCrop(224, (0.6, 1)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            ]
        self.strong_trsf = [
            transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
        ]
        self.test_trsf = []

        
        self.common_trsf = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        self.class_order = np.arange(200).tolist()

    # def getdata(self, train:bool, img_dir):
    #     data, targets = [], []
    #     with open(os.path.join(self.root, 'train_test_split.txt')) as f:
    #         for line in f:
    #             image_id, is_train = line.split()
    #             if int(is_train) == int(train):
    #                 data.append(os.path.join(img_dir, self.images_path[image_id]))
    #                 targets.append(self.class_ids[image_id])
            
    #     return np.array(data), np.array(targets)

    # def download_data(self):
    #     os.environ["DATA"] = "/home/enroll2024/zhiping/Datasets"
    #     root_dir = os.path.join(os.environ["DATA"], 'CUB_200_2011')
    #     img_dir = os.path.join(root_dir, 'images')
    #     self.root = root_dir

    #     self.images_path = {}
    #     with open(os.path.join(root_dir, 'images.txt')) as f:
    #         for line in f:
    #             image_id, path = line.split()
    #             self.images_path[image_id] = path

    #     self.class_ids = {}
    #     with open(os.path.join(root_dir, 'image_class_labels.txt')) as f:
    #         for line in f:
    #             image_id, class_id = line.split()
    #             self.class_ids[image_id] = class_id

    #     self.train_data, self.train_targets = self.getdata(True, img_dir)
    #     self.test_data, self.test_targets = self.getdata(False, img_dir)

    #     print(len(np.unique(self.train_targets))) # output: 
    #     print(len(np.unique(self.test_targets))) # output: 

    def getdata(self, train:bool, img_dir):
        data, targets = [], []
        with open(os.path.join(self.root, 'train_test_split.txt')) as f:
            for line in f:
                image_id, is_train = line.split()
                if int(is_train) == int(train):
                    data.append(os.path.join(img_dir, self.images_path[image_id]))
                    targets.append(int(self.class_ids[image_id])-1)
        # print(targets)    
        return np.array(data), np.array(targets)

    def download_data(self):
        os.environ['DATA'] = "/home/enroll2024/zhiping/Datasets"
        root_dir = os.path.join(os.environ["DATA"], 'CUB_200_2011')
        img_dir = os.path.join(root_dir, 'images')
        self.root = root_dir

        self.images_path = {}
        with open(os.path.join(root_dir, 'images.txt')) as f:
            for line in f:
                image_id, path = line.split()
                self.images_path[image_id] = path

        self.class_ids = {}
        with open(os.path.join(root_dir, 'image_class_labels.txt')) as f:
            for line in f:
                image_id, class_id = line.split()
                self.class_ids[image_id] = class_id

        self.train_data, self.train_targets = self.getdata(True, img_dir)
        # if self._shots != None:
        #     self.train_data, self.train_targets = self.stratified_sample(
        #         self.train_data, 
        #         self.train_targets, 
        #         samples_per_class=self._shots, 
        #         random_seed=self._config.seed
        #     )
        self.test_data, self.test_targets = self.getdata(False, img_dir)
        min_side, max_side = get_image_dimensions_stats(self.train_data)
        print(f"所有图片中的最短边长: {min_side} 像素")
        print(f"所有图片中的最长边长: {max_side} 像素")
        min_side, max_side = get_image_dimensions_stats(self.test_data)
        print(f"所有图片中的最短边长: {min_side} 像素")
        print(f"所有图片中的最长边长: {max_side} 像素")

        # organize_images_by_class("/home/enroll2024/zhiping/Storage/code/cub200_dataset/train",self.train_data,self.train_targets)
        # organize_images_by_class("/home/enroll2024/zhiping/Storage/code/cub200_dataset/test",self.test_data,self.test_targets)

def organize_images_by_class(output_dir, train_data, train_targets):
    """
    根据类别整理图片到不同的文件夹
    
    参数:
        output_dir: 输出目录路径
        train_data: numpy数组，包含图片路径(string类型)
        train_targets: numpy数组，包含对应的类别标签(int类型)
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查输入数组长度是否一致
    if len(train_data) != len(train_targets):
        raise ValueError("train_data和train_targets的长度必须相同")
    
    # 遍历所有样本
    for img_path, target in zip(train_data, train_targets):
        # 创建目标类别目录
        class_dir = os.path.join(output_dir, str(target))
        os.makedirs(class_dir, exist_ok=True)
        
        # 获取图片文件名
        img_name = os.path.basename(img_path)
        
        # 构建目标路径
        dst_path = os.path.join(class_dir, img_name)
        
        try:
            # 复制文件
            shutil.copy2(img_path, dst_path)
        except FileNotFoundError:
            print(f"警告: 文件 {img_path} 不存在，跳过")
        except Exception as e:
            print(f"复制文件 {img_path} 时出错: {str(e)}")

def get_image_dimensions_stats(image_paths):
    """
    统计给定图片路径列表中所有图片的最短边长和最长边长
    
    参数:
    image_paths: numpy数组，包含图片路径字符串
    
    返回:
    tuple: (最短边长, 最长边长)
    """
    min_side = float('inf')
    max_side = 0
    
    for img_path in image_paths:
        # 确保路径是字符串且文件存在
        if not isinstance(img_path, str) or not os.path.exists(img_path):
            print(f"警告: 路径不存在或无效: {img_path}")
            continue
            
        try:
            # 打开图片并获取尺寸
            with Image.open(img_path) as img:
                width, height = img.size
                # 计算当前图片的最小边长
                current_min_side = min(width, height)
                
                # 更新全局最小边长
                if current_min_side < min_side:
                    min_side = current_min_side
                
                # 更新全局最大边长
                current_max_side = max(width, height)
                if current_max_side > max_side:
                    max_side = current_max_side
                    
        except Exception as e:
            print(f"错误: 无法处理图片 {img_path}: {e}")
            continue
    
    # 如果没有找到有效的图片
    if min_side == float('inf'):
        return None, None
    
    return min_side, max_side