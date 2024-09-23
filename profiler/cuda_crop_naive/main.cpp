#include <torch/extension.h>
torch::Tensor crop_image(torch::Tensor image, torch::Tensor crop_centers, int crop_size)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("crop_image", torch::wrap_pybind_function(crop_image), "crop_image");
}