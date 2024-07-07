from PIL import Image
import numpy as np
import torch as t
import random
import pickle
import utils

class LoadFloorplanTrain():
    """
    Loading a floorplan for train
    """ 
    def __init__(self, floorplan_path, mask_size, random_shuffle=True):
        "Read data from pickle"
        with open(floorplan_path, 'rb') as pkl_file:
            [inside, boundary, _, _, room_node] = pickle.load(pkl_file)
       
        "randomly order rooms"
        if random_shuffle:
            random.shuffle(room_node)

        self.inside = t.from_numpy(inside)
        self.boundary = t.from_numpy(boundary)
        self.data_size = self.inside.shape[0]
        self.mask_size = mask_size
        self.continue_node = []
        for node in room_node:
            if node['category'] == 0:
                self.living_node = node  
            else:
                self.continue_node.append(node)           

        "inside_mask"
        self.inside_mask = t.zeros((self.data_size, self.data_size))
        self.inside_mask[self.inside != 0] = 1.0    
        
        "boundary_mask"
        self.boundary_mask = t.zeros((self.data_size, self.data_size))    
        self.boundary_mask[self.boundary == 127] = 1.0               
        self.boundary_mask[self.boundary == 255] = 0.5                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

        "front_door_mask"
        self.front_door_mask = t.zeros((self.data_size, self.data_size))
        self.front_door_mask[self.boundary == 255] = 1.0
        
    def get_composite_living(self, num_extra_channels=0):
        composite = t.zeros((num_extra_channels+3, self.data_size, self.data_size))  
        composite[0] = self.inside_mask
        composite[1] = self.boundary_mask
        composite[2] = self.front_door_mask
        return composite