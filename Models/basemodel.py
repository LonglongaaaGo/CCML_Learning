
import os
import torch
import torch.nn as nn
from Tools.utils import check_data_parallel

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.model = None

    def init(self, *input, **kwargs):
        pass

    def forward(self, *input, **kwargs):
        pass

    def optimize_parameters(self,*input, **kwargs):
        pass

    def get_current_visuals(self,*input, **kwargs):
        pass

    def get_current_losses(self,*input, **kwargs):
        pass

    def update_learning_rate(self,*input, **kwargs):
        pass

    def test(self,*input, **kwargs):
        pass

    def train(self,*input, **kwargs):
        pass

    def save_network(self,epoch,test_acc,file_name):
        print('Saving..')
        state = {
            'model': self.model.state_dict(),
            'acc': test_acc,
            'epoch': epoch,
        }
        torch.save(state, file_name)

    def load_network(self,path,name):
        checkpoint = torch.load(os.path.join(path,name), map_location=self.device)
        if check_data_parallel(checkpoint):
            self.model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model'].items()})
        else:
            self.model.load_state_dict(checkpoint['model'])
        return


    def adjust_learning_rate(self, optimizer, epoch, args):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = args.learning_rate * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def calculate_IoU(self):
        print("calculate_IoU function is not implemented !")
        return

    def visualization(self):
        print("visualization function is not implemented !")
        return

    def test_speed(self):
        print("test_speed function is not implemented !")
        return

