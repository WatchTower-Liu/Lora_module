import torch
from torch import nn
import re
import math

class loraConv(nn.Module):
    def __init__(self, name, org_module, lora_dim, alpha, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = name
        input_channel = org_module.in_channels
        output_channel = org_module.out_channels
        kernel_size = org_module.kernel_size
        padding = org_module.padding
        strid = org_module.stride

        self.scale = alpha / lora_dim

        self.org_forward = org_module.forward
        # del org_module
        if padding == (0, 0):
            self.down_conv = nn.Conv2d(input_channel, lora_dim, kernel_size, strid, bias=False)
        else:
            self.down_conv = nn.Conv2d(input_channel, lora_dim, kernel_size, padding=padding, stride=strid, bias=False)
        self.up_conv = nn.Conv2d(lora_dim, output_channel, kernel_size=(1, 1), stride=(1, 1), bias=False)

        torch.nn.init.kaiming_uniform_(self.down_conv.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.up_conv.weight)  # 防止初始参数破环原始信息


    def forward(self, x):
        org_x = self.org_forward(x)
        down_x = self.down_conv(x)
        up_x = self.up_conv(down_x)
        return up_x * self.scale + org_x
    
class loraLinear(nn.Module):
    def __init__(self, name, org_module, lora_dim, alpha, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = name
        input_channel = org_module.in_features
        output_channel = org_module.out_features

        self.scale = alpha / lora_dim

        self.org_forward = org_module.forward
        # del org_module
        self.down_linear = nn.Linear(input_channel, lora_dim, bias=False)
        self.up_linear = nn.Linear(lora_dim, output_channel, bias=False)

        torch.nn.init.kaiming_uniform_(self.down_linear.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.up_linear.weight)    # 防止初始参数破环原始信息

    def forward(self, x):
        org_x = self.org_forward(x)
        down_x = self.down_linear(x)
        up_x = self.up_linear(down_x)
        return up_x * self.scale + org_x
    
    
class loraModle(nn.Module):
    def __init__(self, target_model, lora_dim, alpha, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.target_model = target_model
        self.lora_dim = lora_dim
        self.alpha = alpha

    @staticmethod
    def check_target_module(name:str, target_model:list):
        for target in target_model:
            layer_name = name.split(".")
            layer_num = len(layer_name)
            target_name = target.split(".")
            target_num = len(target_name)
            last_layer_name = layer_name[-target_num:]
            if last_layer_name == target_name:
                return True
        return False

    def inject(self, model: nn.Module):
        self.org_hit_module = []
        self.lora_module = []
        lora_idx = 0
        for name, module in model.named_modules():
            if self.check_target_module(name, self.target_model):
                self.org_hit_module.append({name:module})
                if isinstance(module, nn.Conv2d):
                    Lora_module = loraConv("lora_{}".format(lora_idx), module, self.lora_dim, self.alpha).to(module.weight.device)
                    lora_idx += 1
                elif isinstance(module, nn.Linear):
                    Lora_module = loraLinear("lora_{}".format(lora_idx), module, self.lora_dim, self.alpha).to(module.weight.device)
                    lora_idx += 1

                self.lora_module.append({name:Lora_module})
        print("lora module number: {}".format(len(self.lora_module)))
        for M in self.lora_module:
            for name, module in M.items():
                # module.apply_to()
                nameList = name.split(".")
                change_module = model
                for i in range(len(nameList)-1):
                    change_module = getattr(change_module, nameList[i])
                setattr(change_module, nameList[-1], module)
                self.add_module(module.name, module)
        return model

    @property
    def lora_modules(self):
        return self.lora_module
    
    @property
    def lora_parameter(self):
        parameters = []
        for M in self.lora_module:
            for name, module in M.items():
                # print(name)
                parameters.extend(module.parameters())
        return parameters
    
    def save_lora(self, path):
        torch.save(self.state_dict(), path)
        return path
    
    def load_lora(self, path):
        self.load_state_dict(torch.load(path))
        return self

def main():
    """
    测试一下是否正常注入
    """
    class testNet(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.conv1 = nn.Conv2d(3, 3, 3, 1, 1)
            self.conv2 = nn.Conv2d(3, 3, 3, 1, 1)
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            return x
    data = torch.randn(1, 3, 8, 8)
    model = testNet()
    print(model)
    lora = loraModle(["conv1"], 16)
    model = lora.inject(model)
    print(model)
    print(lora.lora_modules)



if __name__ == '__main__':
    main()