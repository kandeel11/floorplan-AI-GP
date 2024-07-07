from .floorplan_train import LoadFloorplanTrain
from torch.utils import data
import torch as t
import utils
import os 

class LivingDataset(data.Dataset):
    def __init__(self, data_root, mask_size):
        self.mask_size = mask_size
        self.floorplans = [os.path.join(data_root, pth_path) for pth_path in os.listdir(data_root)]

    def __len__(self):
        return len(self.floorplans)

    def __getitem__(self, index):
        floorplan_path = self.floorplans[index]
        floorplan = LoadFloorplanTrain(floorplan_path, self.mask_size)
        living_h, living_w = floorplan.living_node['centroid']

        input = floorplan.get_composite_living(num_extra_channels=0)
        target = t.zeros(2)
        target[0] = living_h
        target[1] = living_w
        
        return input, target