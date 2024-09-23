#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void CropCudaKernel(
    const T* image_ptr,
    const int* crop_centers_ptr,
    const int image_size,
    const int channels,
    int crop_size,
    const int num_crops,
    T* crops_ptr
) {
    // a pointer to the (flattened) big image, an array of (flattened) crop centers coordinates, as well as the image size, the number of channels, the crop size, and the total number of crops
    const int crop_id = blockIdx.x;
    const int center_x = crop_centers_ptr[crop_id*3];
    const int center_y = crop_centers_ptr[1 + 3*crop_id];
    const int center_z = crop_centers_ptr[2 + 3*crop_id];
    
    // retrieve the current crop center coordinates with the block index blockIdx.x info
    for (int id = threadIdx.x; id < crop_size*crop_size*crop_size*channels; id += blockDim.x) {
        // Coordinates inside the crop (0 <= coords < crop_size)
        int id_temp = id;
        const int c = id_temp % channels;
        id_temp /= channels;
        const int z = id_temp % crop_size;
        id_temp /= crop_size;
        const int y = id_temp % crop_size;
        const int x = id_temp / crop_size;
        
        
        // Corresponding coordinates in original image
        int image_x = x + (center_x - crop_size / 2);
        int image_y = y + (center_y - crop_size / 2);
        int image_z = z + (center_z - crop_size / 2);
        int img_idx = c + channels * (image_z + image_size * (image_y + image_size * image_x ));
        
        if ((img_idx >= image_size * image_size * image_size * channels) || (img_idx < 0)) continue;

            int crop_idx = c + channels * (z + crop_size * (y + crop_size * (x + crop_size * crop_id)));
            crops_ptr[crop_idx] = image_ptr[img_idx];
        }
    }
torch::Tensor crop_image(
    torch::Tensor image,
    torch::Tensor crop_centers,
    int crop_size
) {
    // Input validation
    TORCH_CHECK(image.is_cuda(), "Input image must be a CUDA tensor");
    TORCH_CHECK(crop_centers.is_cuda(), "Crop centers must be a CUDA tensor");
    TORCH_CHECK(image.dim() == 4, "Input image must be 4-dimensional (B, C, H, W)");
    TORCH_CHECK(crop_centers.dim() == 2, "Crop centers must be 2-dimensional (num_crops, 3)");
    TORCH_CHECK(crop_centers.size(1) == 3, "Crop centers must have 3 coordinates per crop");

    const int batch_size = image.size(0);
    const int channels = image.size(1);
    const int image_size = image.size(2);
    const int num_crops = crop_centers.size(0);

    // Create output tensor
    auto crops = torch::empty({batch_size, num_crops, channels, crop_size, crop_size, crop_size},
                              image.options());

    // Launch kernel for each image in the batch
    for (int b = 0; b < batch_size; ++b) {
	const auto stream = c10::cuda::getCurrentCUDAStream();
        AT_DISPATCH_FLOATING_TYPES(image.type(), "crop_image_cuda", ([&] {
            CropCudaKernel<scalar_t><<<num_crops, 1024, 0, stream.stream()>>>(
                image[b].data_ptr<scalar_t>(),
                crop_centers.data_ptr<int>(),
                image_size,
                channels,
                crop_size,
                num_crops,
                crops[b].data_ptr<scalar_t>()
            );
        }));
    }

    return crops;
} 
    
