import torch
import torch.nn as nn

def build_custom_net(layers):
    class CustomNet(nn.Module):
        def __init__(self, layers):
            super(CustomNet, self).__init__()
            self.layers = nn.ModuleList()

            for layer_params in layers:
                layer_type = layer_params['type']
                activation = layer_params.get('activation_function', None)
                io_shape = layer_params.get('io_shape', None)
                
                if layer_type == 'dense':
                    in_features = io_shape[0]
                    out_features = io_shape[1]
                    self.layers.append(nn.Linear(in_features, out_features))

                elif layer_type == 'conv2d':
                    conv_params = layer_params['conv_params']
                    in_channels = io_shape[0]
                    out_features = io_shape[1]
                    self.layers.append(nn.Conv2d(in_channels, out_features, kernel_size=conv_params['kernel_size'], stride=conv_params['stride'], padding=conv_params['padding']))

                elif layer_type == 'flatten':
                    self.layers.append(nn.Flatten(start_dim=1))
                    
                elif layer_type == 'maxpool2d':
                    maxpool_params = layer_params['maxpool_params']
                    kernel_size = maxpool_params['kernel_size']
                    stride = maxpool_params['stride']
                    padding = maxpool_params['padding']
                    self.layers.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding))
                    
                elif layer_type == 'batchnorm2d':
                    batchnorm_params = layer_params['batchnorm_params']
                    num_features = batchnorm_params['num_features']
                    self.layers.append(nn.BatchNorm2d(num_features))
                    
                elif layer_type == 'averagepool2d':
                    averagepool_params = layer_params['averagepool_params']
                    kernel_size = averagepool_params['kernel_size']
                    stride = averagepool_params['stride']
                    padding = averagepool_params['padding']
                    self.layers.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))
                    
                elif layer_type == 'dropout':
                    probability = layer_params['p']
                    self.layers.append(nn.Dropout(p=probability))
                    
                if activation == 'relu':
                    self.layers.append(nn.ReLU())
                elif activation == 'softmax':
                    self.layers.append(nn.Softmax(dim=1))

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    return CustomNet(layers)

def print_network_architecture(net):
    print(net)
