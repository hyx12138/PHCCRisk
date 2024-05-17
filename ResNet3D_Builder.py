import torch
import torch.nn as nn
import monai

class ResNet3D(nn.Module):
    def __init__(self,
                 base_encoder,
                 spatial_dims: int = 3,
                 n_input_channels: int = 1,
                 num_classes: int = 1, 
                 pretrained_dict_path: str = None,):

        super(ResNet3D, self).__init__()

        self.spatial_dims = spatial_dims
        self.n_input_channels = n_input_channels
        self.num_classes = num_classes
        self.pretrained_dict_path = pretrained_dict_path

        encoder = base_encoder(spatial_dims=self.spatial_dims, n_input_channels=self.n_input_channels, num_classes=self.num_classes)
        if self.pretrained_dict_path is not None:
                net_dict = encoder.state_dict()
                pretrain = torch.load(self.pretrained_dict_path)
                pretrain['state_dict'] = {k.replace('module.', ''): v for k, v in pretrain['state_dict'].items()}
                missing = tuple({k for k in net_dict.keys() if k not in pretrain['state_dict']})

                inside = tuple({k for k in pretrain['state_dict'] if k in net_dict.keys()})

                unused = tuple({k for k in pretrain['state_dict'] if k not in net_dict.keys()})

                pretrain['state_dict'] = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
                encoder.load_state_dict(pretrain['state_dict'], strict=False)
                print(f'Missing: {len(missing)}\tInside: {len(inside)}\tUnused: {len(unused)}')
        
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.encoder(x).unsqueeze(0)

        return x
    
if __name__ == "__main__":
     model = ResNet3D(
          base_encoder=monai.networks.nets.__dict__['resnet18'],
          spatial_dims=3,
          n_input_channels=1,
          num_classes=1,
          pretrained_dict_path = './PHCCRisk/resnet_18_23dataset.pth') # pretrained weights from MedicalNet https://github.com/Tencent/MedicalNet
     print(model)