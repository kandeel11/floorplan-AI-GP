from .floorplan_train import LoadFloorplanTrain
from torch.utils import data
import torch as t
import random
import utils
import os 

class ContinueDataset(data.Dataset):
    def __init__(self, data_root, mask_size, complete_prob=0.5):
        self.mask_size = mask_size
        self.complete_prob = complete_prob
        self.floorplans = [os.path.join(data_root, pth_path) for pth_path in os.listdir(data_root)]

    def __len__(self):
        return len(self.floorplans)

    def __getitem__(self, index):       
        floorplan_path = self.floorplans[index]
        floorplan = LoadFloorplanTrain(floorplan_path, self.mask_size)
        num_category = utils.num_category 
        existing = t.zeros(num_category)        
        continue_node = floorplan.continue_node

        living_node = floorplan.living_node 
        floorplan.add_room(living_node)
        existing[utils.label2index(living_node["category"])] += 1
        is_complete = random.random() < self.complete_prob
        if not is_complete:  
            num_rooms = random.randint(0, len(continue_node)-1)
        else:
            num_rooms = len(continue_node)
        for i in range(num_rooms):
            node = continue_node[i]
            floorplan.add_room(node)
            existing[utils.label2index(node["category"])] += 1

        input = floorplan.get_composite_continue(num_extra_channels=0)
        target = not is_complete

        return input, target, existing