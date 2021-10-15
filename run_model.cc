#include <cstdint>
#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char *argv[]) {

  std::int32_t batch_size = 1;
  std::int32_t seq_len = 20;
  std::int32_t num_heads = 12;
  std::int32_t hidden_layer_size = 768;

  torch::jit::script::Module module;
  try {

    module = torch::jit::load("bert-12-heads-768-hidden.pt");

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({batch_size, seq_len, hidden_layer_size}));
    inputs.push_back(torch::ones({batch_size, seq_len, seq_len}));

    
    while (true) {
        at::Tensor output = module.forward(inputs).toTensor();
    }

  } catch (const c10::Error &e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
}