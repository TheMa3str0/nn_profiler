#include <torch/torch.h>
#include "CustomNetworks.h"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include "InferenceProfiler.cpp"
#include "TrainingProfiler.cpp"
#include "ConfigParser.h"

using json = nlohmann::json;

int main() {
    ConfigParser configParser("../../network_config.json");
    if (!configParser.parseConfig()) {
        return -1;
    }
    std::vector<NetworkLayer> network_layers = configParser.getNetworkLayers();
    CustomNetwork network(network_layers);

    // Print the network architecture
    std::cout << "Network Architecture:\n" << network << std::endl;

    // Example: Perform inference with a sample input
    const std::vector<int64_t> input_shape = configParser.getInputShape();
    const std::string device = configParser.getDevice();
    const std::string task = configParser.getTask();
    if (configParser.getMode() == "inference") {
        InferenceParameters params = configParser.getInferenceParameters();
        InferenceProfiler inference_profiler(network, device, input_shape, params.no_inferences);
        inference_profiler.profile();
    } else if (configParser.getMode() == "training") {
        TrainingParameters params = configParser.getTrainingParameters();
        TrainingProfiler training_profiler(
            network, device, input_shape,
            params.learning_rate,
            params.optimizer_choice,
            params.loss_function,
            params.batch_size,
            params.epochs,
            params.num_samples,
            params.num_classes,
            task);
        training_profiler.train();
    } else {
        std::cout << "Invalid mode specified in the configuration file." << std::endl;
    }

    return 0;
}
