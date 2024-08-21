from loader.baseloader import BaseLoader
from loader.img_flist import ImageFilelist
from torchvision.datasets import ImageFolder

import logging

logger = logging.getLogger('mylogger')


class FileDataLoader(BaseLoader):
    """Function to build data loader(s) for the specified splits given the parameters.
    """

    def __init__(self, cfg, splits, batch_size):
        super().__init__(cfg, splits, batch_size)

    def getDataset(self, root_dir, flist=None, transform=None, loader=None):
        return ImageFilelist(root_dir=root_dir, flist=flist, transform=transform)

class ImageDataLoader(BaseLoader):
    """Function to build data loader(s) for the specified splits given the parameters.
    """

    def __init__(self, cfg, splits, batch_size):
        super().__init__(cfg, splits, batch_size)

    def getDataset(self, data_root, data_list=None, trans=None, loader=None):
        dataset = ImageFolder(root_dir=data_root, transform=trans, loader=loader)    
        dataset.data = [imgs[0] for imgs in dataset.imgs]
        dataset.target = [imgs[1] for imgs in dataset.imgs]
        dataset.root_dir = dataset.root

        return dataset