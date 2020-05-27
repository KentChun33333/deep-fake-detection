

from meso_net_torch import Meso4
from ffdnet import FFDNet

class FFD_Meso_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffd_model = FFDNet()
        self.meso_model = Meso4()

    def forward(self, imgs):
        noise_img = self.ffd_model(imgs) 
        y =  self.meso_model(noise_img)
        y, ind = torch.max(y, dim=0)
        return y