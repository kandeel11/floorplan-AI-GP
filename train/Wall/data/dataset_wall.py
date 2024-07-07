from .floorplan_train import LoadFloorplanTrain
from torch.utils import data
import torch as t
import random
import utils
import os

class WallDataset(data.Dataset):
    def __init__(self, data_root, mask_size):
        self.mask_size = mask_size
        self.floorplans = [os.path.join(data_root, pth_path) for pth_path in os.listdir(data_root)]

    def __len__(self):
        return len(self.floorplans)

    def __getitem__(self, index):          
        floorplan_path = self.floorplans[index]
        floorplan = LoadFloorplanTrain(floorplan_path, self.mask_size)
        OUTSIDE = 2
        NOTHING = 1
        INTERIORWALL = 0

        living_node = floorplan.living_node
        floorplan.add_room(living_node)
        continue_node = floorplan.continue_node
        for node in continue_node:
            floorplan.add_room(node)

        input = floorplan.get_composite_wall(num_extra_channels=0)
        target = t.zeros((floorplan.data_size, floorplan.data_size), dtype=t.long)
        target[target == 0] = OUTSIDE
        target[floorplan.inside != 0] = NOTHING
        target[floorplan.interiorWall == 1] = INTERIORWALL
        target[floorplan.interiordoor == 1] = INTERIORWALL

        return input, target