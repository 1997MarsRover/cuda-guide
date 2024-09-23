import torch
from torch.utils.cpp_extension import load_inline
from PIL import Image
from torchvision import transforms
from config.config  import IMAGE_PATH
# Updated CUDA kernel source code
cuda_source = '''
#include <c10/cuda/CUDAStream.h>

template <typename T>
__global__ void CropCudaKernel(
    const T* image_ptr,
    const int* crop_centers_ptr,
    const int image_size_x,
    const int image_size_y,
    const int image_size_z,
    const int channels,
    int crop_size,
    const int num_crops,
    T* crops_ptr
) {
    const int crop_id = blockIdx.x;
    const int center_x = crop_centers_ptr[crop_id * 3];
    const int center_y = crop_centers_ptr[1 + 3 * crop_id];
    const int center_z = crop_centers_ptr[2 + 3 * crop_id];
    
    for (int id = threadIdx.x; id < crop_size * crop_size * crop_size * channels; id += blockDim.x) {
        int id_temp = id;
        const int c = id_temp % channels;
        id_temp /= channels;
        const int z = id_temp % crop_size;
        id_temp /= crop_size;
        const int y = id_temp % crop_size;
        const int x = id_temp / crop_size;
        
        int image_x = x + (center_x - crop_size / 2);
        int image_y = y + (center_y - crop_size / 2);
        int image_z = z + (center_z - crop_size / 2);

        // Handle 2D images by clamping image_z
        image_z = max(0, min(image_z, image_size_z - 1));

        if (image_x >= 0 && image_x < image_size_x &&
            image_y >= 0 && image_y < image_size_y &&
            image_z >= 0 && image_z < image_size_z) {

            int img_idx = c + channels * (image_z + image_size_z * (image_y + image_size_y * image_x));
            int crop_idx = c + channels * (z + crop_size * (y + crop_size * (x + crop_size * crop_id)));
            crops_ptr[crop_idx] = image_ptr[img_idx];
        }
    }
}

torch::Tensor crop_image(
    torch::Tensor image,
    torch::Tensor crop_centers,
    int crop_size
) {
    TORCH_CHECK(image.is_cuda(), "Input image must be a CUDA tensor");
    TORCH_CHECK(crop_centers.is_cuda(), "Crop centers must be a CUDA tensor");
    TORCH_CHECK(image.dim() == 4 || image.dim() == 5, "Input image must be 4D (B, C, H, W) or 5D (B, C, D, H, W)");
    TORCH_CHECK(crop_centers.dim() == 2, "Crop centers must be 2D (num_crops, 3)");

    const int batch_size = image.size(0);
    const int channels = image.size(1);
    const int image_size_z = (image.dim() == 5) ? image.size(2) : 1;
    const int image_size_y = image.size(image.dim() - 2);
    const int image_size_x = image.size(image.dim() - 1);
    const int num_crops = crop_centers.size(0);

    auto crops = torch::empty({batch_size, num_crops, channels, crop_size, crop_size, crop_size},
                              image.options());

    for (int b = 0; b < batch_size; ++b) {
	const auto stream = at::cuda::getCurrentCUDAStream();
        AT_DISPATCH_FLOATING_TYPES(image.type(), "crop_image_cuda", ([&] {
            CropCudaKernel<scalar_t><<<num_crops, 1024, 0, stream.stream()>>>(
                image[b].data_ptr<scalar_t>(),
                crop_centers.data_ptr<int>(),
                image_size_x,
                image_size_y,
                image_size_z,
                channels,
                crop_size,
                num_crops,
                crops[b].data_ptr<scalar_t>()
            );
        }));
    }

    return crops;
}
'''

cpp_source = "torch::Tensor crop_image(torch::Tensor image, torch::Tensor crop_centers, int crop_size);"

# Load the CUDA kernel as a PyTorch extension
cuda_cropping = load_inline(
    name='cuda_cropping_extension',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['crop_image'],
    with_cuda=True,
    extra_cuda_cflags=["-O2"],
    build_directory='./cuda_crop_naive'
)

image = Image.open(IMAGE_PATH)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_tensor = transform(image).unsqueeze(0).cuda()

# Pad image tensor with depth of 1 for 2D compatibility
image_tensor = image_tensor.unsqueeze(2) 

crop_centers = torch.tensor([
    [100, 100, 0],  
    [200, 200, 0],  
    [300, 300, 0]  
], dtype=torch.int32, device='cuda')

crop_size = 64  

crops = cuda_cropping.crop_image(image_tensor, crop_centers, crop_size)

import matplotlib.pyplot as plt
import torchvision.transforms as T
reverse_transform = T.Compose([
    T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1/0.229, 1/0.224, 1/0.225]),
    T.ToPILImage()
])

crops = crops.cpu()
num_crops = crops.size(1)  # Assuming the crops tensor has shape (batch_size, num_crops, channels, crop_size, crop_size, crop_size)
for i in range(num_crops):

    crop_2d = crops[0, i, :, 0, :, :]  
    crop_image = reverse_transform(crop_2d)  
    plt.figure()
    plt.imshow(crop_image)
    plt.title(f'Crop {i+1}')
    plt.axis('off')  