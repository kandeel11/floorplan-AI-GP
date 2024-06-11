from PIL import Image
import numpy as np
import torch as t
import utils
num_category = utils.num_category
class LoadFloorplanSynth():
    """
    Loading a floorplan for synth
    """ 
    def __init__(self, temp, mask_size=9):
        "Read data from Image"
        floorplan = np.asarray(temp, dtype=np.uint8)
        self.input_map = floorplan.copy()
        self.boundary = t.from_numpy(floorplan[:,:,0]) 
        self.inside = t.from_numpy(floorplan[:,:,3])  
        self.data_size = self.inside.shape[0]
        self.mask_size = mask_size
        
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

        "category_mask"
        self.category_mask = t.zeros((utils.num_category, self.data_size, self.data_size))

        "room_node"  
        self.room_node = []

        "existing_category"  
        self.existing_category = t.zeros(utils.num_category)
 
    def get_composite_living(self, num_extra_channels=0):
        composite = t.zeros((num_extra_channels+3, self.data_size, self.data_size))
        composite[0] = self.inside_mask
        composite[1] = self.boundary_mask
        composite[2] = self.front_door_mask     
        return composite

    def get_composite_continue(self, num_extra_channels=0):
        composite = t.zeros((utils.num_category+num_extra_channels+4, self.data_size, self.data_size))
        composite[0] = self.inside_mask
        composite[1] = self.boundary_mask
        composite[2] = self.front_door_mask 
        composite[3] = self.category_mask.sum(0)
        for i in range(utils.num_category):
            composite[i+4] = self.category_mask[i]
        return composite

    def get_composite_location(self, num_extra_channels=0):
        composite = t.zeros((utils.num_category+num_extra_channels+4, self.data_size, self.data_size))
        composite[0] = self.inside_mask
        composite[1] = self.boundary_mask
        composite[2] = self.front_door_mask  
        composite[3] = self.category_mask.sum(0)
        for i in range(utils.num_category):
            composite[i+4] = self.category_mask[i]
        return composite

    def get_composite_wall(self, num_extra_channels=0):
        composite = t.zeros((utils.num_category+num_extra_channels+4, self.data_size, self.data_size))
        composite[0] = self.inside_mask
        composite[1] = self.boundary_mask
        composite[2] = self.front_door_mask
        composite[3] = self.category_mask.sum(0)
        for i in range(utils.num_category):
            composite[i+4] = self.category_mask[i]
        return composite

    def add_room(self, node):
        index = utils.label2index(node['category']) 
        h, w = node['centroid']
        min_h = max(h - self.mask_size, 0)
        max_h = min(h + self.mask_size, self.data_size - 1)
        min_w = max(w - self.mask_size, 0)
        max_w = min(w + self.mask_size, self.data_size - 1)
        self.category_mask[index, min_h:max_h+1, min_w:max_w+1] = 1.0
        self.room_node.append(node)
        self.existing_category[index] += 1