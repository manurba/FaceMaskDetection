from torch import long, tensor
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
import numpy as np
import cv2

"""
This script converts the Dataset to a more easily manageable format
PIL Images of 100x100
"""

class MaskDataset(Dataset):
    """
        This class transforms and resizes the input dataset
        mantaining the original labelling of the Masked Faces
        Dataset
        0 = 'No Mask'
        1 = 'Mask'
    """
    def __init__(self, dataFrame):
        self.dataFrame = dataFrame
        
        # Transform to PIL Image with 100x100 size
        self.transformation = Compose([
            ToPILImage(),
            Resize((100, 100)),
            ToTensor(), # Depending on no-mask/mask [0, 1]
        ])
    
    def __len__(self):
        return len(self.dataFrame.index)

    def __getitem__(self, key):
        '''
        Retrieve transformed images
        '''
        if isinstance(key, slice):
            raise NotImplementedError('Slicing not supported')
        
        row = self.dataFrame.iloc[key]
        image = cv2.imdecode(np.fromfile(row['image'], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        return {
            'image': self.transformation(image),
            'mask': tensor([row['mask']], dtype=long),
        }