from torchvision import transforms
from datasets.idata import iData
import os
import numpy as np
import random

class path16_strong(iData):
    '''
    Dataset Name:   Path16
    Task:           Diverse classification task (binary/multi-class)
    Data Format:    224x224 color images.
    Data Amount:    800 each class for training , 100 each class for test.
    
    Reference: 
    '''
    def __init__(self, img_size=None) -> None:
        super().__init__()
        # 以下表示中，字符为子数据集名字, 数字为子数据集中包含的类别数
        self.has_valid = False
        
        #breast cancer: BR, Colon: CO, Lung:LU, oral cancer: OR, TCGA_STD: ST, MHIST: CP, Patch_Camelyon: LY
        
        self._dataset_info = [('colon',2), ('lung',2), ('Patch_Camelyon',2), ('TCGA_STD',4),
                                ('oral_cancer',2), ('MHIST',2), ('breast_cancer',2)] # order 4
        #对应的顺序是：CO，LU，LY，ST，OR，CP，BR，也就是论文第一行的子图
        
        # self._dataset_info = [('MHIST',2), ('breast_cancer',2), ('oral_cancer',2), ('Patch_Camelyon',2),
        #                         ('TCGA_STD',4), ('lung',2), ('colon',2)] # order 5
        #对应的顺序是：CP, BR, OR, LY, ST, LU, CO, 也就是论文第二行的子图



        self.use_path = True
        self.img_size = img_size if img_size != None else 224 # original img size is 28
        # self.train_trsf = [
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     ]
        
        # self.test_trsf = []


        self.train_trsf =  [
            transforms.Resize(256),  # Resize the shortest side to 256 pixels
            transforms.CenterCrop(224),  # Randomly crop a 224x224 area
            transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
            transforms.RandomRotation(degrees=15),  # Randomly rotate the image by ±15 degrees
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Randomly jitter color properties
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5, fill=0),  # Random perspective transformation
            # Adding additional strong augmentations for medical imaging
            # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # Apply Gaussian Blur
            # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),  # Random affine transformation
            # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=True)  # Random erasing
        ]

        self.test_trsf = []

        self.common_trsf = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
        ]

        self._dataset_inc = [data_flag[1] for data_flag in self._dataset_info]
        self.class_order = list(range(sum(self._dataset_inc)))
    
    def shuffle_order(self, seed):
        random.seed(seed)
        random.shuffle(self._dataset_info)
        self._dataset_inc = [data_flag[1] for data_flag in self._dataset_info]

    def getdata(self, src_dir, mode):
        assert mode == 'train' or mode == 'test', 'Unkown mode: {}'.format(mode)
        known_class = 0
        data, targets = [], []
        for sub_dataset_name, class_num in self._dataset_info:
            sub_dataset_dir = os.path.join(src_dir, sub_dataset_name, mode)
            for class_id, class_name in enumerate(os.listdir(sub_dataset_dir)):
                class_dir = os.path.join(sub_dataset_dir, class_name)
                for img_name in os.listdir(class_dir):
                    data.append(os.path.join(class_dir, img_name))
                    targets.append(known_class+class_id)
            known_class += class_num
        
        return np.array(data), np.array(targets, dtype=int)


    def download_data(self):
        # src_dir = os.path.join(os.environ["DATA"], "Pathology16_root")
        src_dir = os.path.join(os.environ["Pathology16"])
        self.train_data, self.train_targets = self.getdata(src_dir, 'train')
        self.test_data, self.test_targets = self.getdata(src_dir, 'test')