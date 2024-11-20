import torch
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import Dataset

random.seed(0)

class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""
    "包装器"

    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:
            初始化类的实例。它接受三个参数：
            user_tensor：包含用户ID的张量。
            item_tensor：包含物品ID的张量。
            target_tensor：预测此项目与用户间产生交互的概率。
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)
    
"""
它提供了一种标准的方式来访问数据集中的样本。Dataset 类是 PyTorch 数据加载框架的基础，它允许你自定义数据加载方式，使得数据可以被 DataLoader 以批量、多进程的方式加载。

以下是 Dataset 类的基本结构和用法：

继承 Dataset 类：你需要创建一个类，继承自 torch.utils.data.Dataset。

实现 __init__ 方法：在这个方法中，初始化你的数据集，比如加载数据文件、预处理数据等。

实现 __len__ 方法：这个方法返回数据集中样本的数量。

实现 __getitem__ 方法：这个方法用于获取数据集中的单个样本。它接受一个索引值，并返回该索引对应的样本。
"""
