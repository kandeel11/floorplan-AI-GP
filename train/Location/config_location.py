import warnings
import utils

class LocationConfig(object):
    train_data_root = '../pickle/train_dataset'
    val_data_root = '../pickle/val_dataset'
    save_log_root = 'log'
    result_file = 'result_location.csv'
    module_name = 'location'
    model_name = 'resnet34_up1'
    load_model_path = None
    load_connect_path = None 
    mask_size = 9

    multi_GPU = False
    batch_size = 16  
    num_workers = 2
    print_freq = 300

    max_epoch = 100
    current_epoch = 0
    save_freq = 10
    val_freq = 1

    update_lr = True
    lr_decay_freq = 1
    lr_base = 1e-4
    weight_decay = 1e-4

    def parse(self, kwargs, file):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                utils.log(file, f'{k}: {getattr(self, k)}')