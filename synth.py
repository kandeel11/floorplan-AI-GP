from floorplan_synth import LoadFloorplanSynth
from PIL import Image
import numpy as np
import torch as t
import random
import models
import shutil
import utils
import time
import os

mask_size = 9
data_size = 256
sample_density = 8
num_category = utils.num_category

class Synth():
    def __init__(self, model_dir,  
                living_epoch=0, from_multi_GPU_living=False,
                continue_epoch=0, from_multi_GPU_continue=False, 
                location_epoch=0, from_multi_GPU_location=False, 
                wall_epoch=0, from_multi_GPU_wall=False):

        self.model_dir = model_dir
        self.living_model, self.living_connect = self._load_living_model(living_epoch, from_multi_GPU_living) 
        self.continue_model, self.continue_connect = self._load_continue_model(continue_epoch, from_multi_GPU_continue)
        self.location_model, self.location_connect = self._load_location_model(location_epoch, from_multi_GPU_location)
        self.wall_model, self.wall_connect = self._load_wall_model(wall_epoch, from_multi_GPU_wall) 
        self.softmax = t.nn.Softmax(dim=1) 
        self.continue_adding = True

    def _load_model(self, model, connect, epoch, from_multi_GPU):
        if from_multi_GPU: 
            model_path = f"{self.model_dir}/{model.name}_parallel_{epoch}.pth"
            connect_path = f"{self.model_dir}/{connect.name}_parallel_{epoch}.pth"
        else:
            model_path = f"{self.model_dir}/{model.name}_{epoch}.pth"
            connect_path = f"{self.model_dir}/{connect.name}_{epoch}.pth"
        model.load_model(model_path, from_multi_GPU)
        connect.load_model(connect_path, from_multi_GPU)            

        model.cuda()
        connect.cuda()
        model.eval()
        connect.eval()   
        return model, connect    

    def _load_living_model(self, epoch, from_multi_GPU):
        model = models.model(
            module_name="living",
            model_name="resnet34_fc1",
            input_channel=3,
            output_channel=2,
            pretrained=False,
            dilation_state=False,
            use_pool=False
        )
        input_channel = 512
        connect = models.connect(
            module_name="living",
            model_name="resnet34_fc1",
            input_channel=input_channel, 
            output_channel=2,
            reshape=True
        )
        if epoch == 0:
            return model, connect
        else:
            return self._load_model(model, connect, epoch, from_multi_GPU)

    def _load_continue_model(self, epoch, from_multi_GPU):
        model = models.model(
            module_name="continue",
            model_name="resnet34_fc2",
            input_channel=utils.num_category+4, 
            output_channel=2,
            pretrained=False,
            dilation_state=False,
            use_pool=True
        )
        input_channel = 512
        connect = models.connect(
            module_name="continue",
            model_name="resnet34_fc2",
            input_channel=input_channel+utils.num_category, 
            output_channel=2,
            reshape=False
        )
        if epoch == 0:
            return model, connect
        else:
            return self._load_model(model, connect, epoch, from_multi_GPU)
    
    def _load_location_model(self, epoch, from_multi_GPU):
        model = models.model(
            module_name="location",
            model_name="resnet34_up1",
            input_channel=utils.num_category+4,
            output_channel=utils.num_category+3, 
            pretrained=False,
            dilation_state=True,
            use_pool=False
        )
        input_channel = 512
        connect = models.connect(
            module_name="location",
            model_name="resnet34_up1",
            input_channel=input_channel, 
            output_channel=utils.num_category+3,
            reshape=False
        )
        if epoch == 0:
            return model, connect
        else:
            return self._load_model(model, connect, epoch, from_multi_GPU)

    def _load_wall_model(self, epoch, from_multi_GPU):
        model = models.model(
            module_name="wall",
            model_name="resnet34_up1",
            input_channel=utils.num_category+4,      
            output_channel=3,  
            pretrained=False,
            dilation_state=True,
            use_pool=False
        )
        input_channel = 512
        connect = models.connect(
            module_name="wall",
            model_name="resnet34_up1",
            input_channel=input_channel, 
            output_channel=3,
            reshape=False
        )
        if epoch == 0:
            return model, connect
        else:
            return self._load_model(model, connect, epoch, from_multi_GPU)

    def add_living(self):
        self.composite = self.floorplan.get_composite_living(num_extra_channels=0)
        with t.no_grad():
            input = self.composite.unsqueeze(0).cuda()
            score_model = self.living_model(input)
            score_connect = self.living_connect(score_model)
            output = score_connect.cpu().numpy().astype(int)

        new_node = {}
        new_node['category'] = 0
        new_node['centroid'] = (output[0, 0], output[0, 1])
        self.floorplan.add_room(new_node)

    def should_continue(self):
        self.composite = self.floorplan.get_composite_continue(num_extra_channels=0)
        with t.no_grad():
            input = self.composite.unsqueeze(0).cuda()
            score_model = self.continue_model(input)
            existing = self.floorplan.existing_category.unsqueeze(0).cuda()
            score_model = t.cat([score_model, existing], 1)
            score_connect = self.continue_connect(score_model)
            output = self.softmax(score_connect).cpu().numpy()
        
        if output[0,1] < 0.6 or len(self.floorplan.room_node) > 10:
            self.continue_adding = False

    def add_room(self):
        self.composite = self.floorplan.get_composite_location(num_extra_channels=0)
        with t.no_grad():
            input = self.composite.unsqueeze(0).cuda()
            score_model = self.location_model(input)
            score_connect = self.location_connect(score_model)
            
        score_softmax = self.softmax(score_connect)
        output = score_softmax.cpu().numpy()
        predict = np.argmax(output, axis=1)[0]
        new_nodes = []
        new_nums = []
        for cat in range(num_category):
            if self.floorplan.existing_category[cat] == 0:
                index_point = []
                shape_array = predict.shape
                min_h = data_size
                max_h = 0
                min_w = data_size
                max_w = 0
                for h in range(shape_array[0]):  
                    for w in range(shape_array[1]):
                        if predict[h, w] == cat:
                            index_point.append((h, w))
                            min_h = h if(min_h > h) else min_h
                            max_h = h if(max_h < h) else max_h
                            min_w = w if(min_w > w) else min_w
                            max_w = w if(max_w < w) else max_w

                if len(index_point) > 0:
                    num_point = 0
                    if max_h - min_h + 1 <= 2 * mask_size and max_w - min_w + 1 <= 2 * mask_size:
                        predict_h = (min_h + max_h) // 2
                        predict_w = (min_w + max_w) // 2
                        num_point = len(index_point)
                    else:
                        predict_h, predict_w = index_point[0]
                        for point in index_point:
                            new_num_point = 0
                            for other_point in index_point:
                                if abs(other_point[0] - point[0]) <= mask_size and \
                                    abs(other_point[1] - point[1]) <= mask_size:
                                    new_num_point += 1
                            if new_num_point > num_point:
                                predict_h, predict_w = point
                                num_point = new_num_point

                    if num_point > 0:       
                        new_node = {}
                        new_node['category'] = utils.index2label(cat)
                        new_node['centroid'] = (predict_h, predict_w)
                        new_nodes.append(new_node)
                        new_nums.append(num_point)

        if len(new_nodes):   
            index_max= new_nums.index(max(new_nums))
            self.floorplan.add_room(new_nodes[index_max])
        else:
            self.continue_adding = False

    def add_wall(self):
        self.composite = self.floorplan.get_composite_wall(num_extra_channels=0)
        with t.no_grad():
            input = self.composite.unsqueeze(0).cuda()
            score_model = self.wall_model(input)
            score_connect = self.wall_connect(score_model)
            score_softmax = self.softmax(score_connect)
            output = score_softmax.cpu().numpy()

        predict = np.argmax(output, axis=1)
        self.map[:,:,1][predict[0] == 0] = 127

    def save_synth_result(self, synth_output_dir, output_path):     
        for node in self.floorplan.room_node:
            self.map[:,:,2][node['centroid']] = node['category'] + 100
        output = Image.fromarray(self.map)  
        return output
def LiSynth(MaskImage):
    model_dir = "//kaggle/input/synthtrained/trained_model"
    synthesizer = Synth(
        model_dir=model_dir, 
        living_epoch=150, from_multi_GPU_living=False,
        continue_epoch=70, from_multi_GPU_continue=True,
        location_epoch=32, from_multi_GPU_location=True,
        wall_epoch=50, from_multi_GPU_wall=False
    ) 
    name = time.strftime('synth_%Y%m%d_%H%M%S')
    log_file = open(f"{name}.txt", 'w')
    test_number = 0
    start_time = time.process_time()
    temp_time = start_time
    test_number = test_number + 1
    synthesizer.floorplan = LoadFloorplanSynth(MaskImage)
    synthesizer.map = synthesizer.floorplan.input_map
    synthesizer.map.flags.writeable = True
    synthesizer.add_living()
    synthesizer.continue_adding = True
    while synthesizer.continue_adding:
        synthesizer.add_room()
        synthesizer.should_continue()
    synthesizer.add_wall()
    synth_output_dir=""
    Img=synthesizer.save_synth_result(synth_output_dir, output_path="result")
    end_time = time.process_time()
    #utils.log(log_file, f'result: {(end_time-temp_time):.2f}s')
    temp_time = end_time
    end_time = time.process_time()
    cost_time = end_time-start_time
    #utils.log(log_file)
    #utils.log(log_file, f'Total test time: {cost_time:.2f}s')
    #utils.log(log_file, f'Total test number: {test_number}')
    #utils.log(log_file, f'Average time: {(cost_time/test_number):.2f}s')
    #utils.log_file.close()
    return Img