import constructor
import inference
import training
import json
from parameter_parser import get_inference_params, get_training_params

import torch
import onnx

def read_config_file(config_file_path):
    with open(config_file_path, 'r') as config_file:
        config_data = json.load(config_file)
    return config_data

#TODO add choice between custom and well-known nets
config_data = read_config_file('../configs/resnet34_config.json')
    
network = constructor.build_custom_net(config_data['network']['layers'])
constructor.print_network_architecture(network)
    
device = config_data['network']['device']

if config_data['network']['mode'] == 'inference':
    inference_params = get_inference_params(config_data)
    inference_time = inference.profile_custom(network, device, *inference_params)
    
    total_time = inference_time * pow(10, -6)
    print(f"Ran {config_data['network']['inference_params']['no_inferences']} inferences in {total_time} ms.")
    print(f"Time spent per inference: {str(total_time / config_data['network']['inference_params']['no_inferences'])} ms on average.")
elif config_data['network']['mode'] == 'training':
    training_params = get_training_params(config_data)
    training_time = training.train_network(network, device, *training_params)
    total_time = training_time * pow(10, -6)
    print(f"Ran {config_data['network']['training_params']['epochs']} epochs with batch size {config_data['network']['training_params']['batch_size']} in {total_time} ms.")
else:
    print("Invalid mode specified in the configuration file.")