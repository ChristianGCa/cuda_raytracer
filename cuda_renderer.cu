// CUDA Renderer Implementation

#include "cuda_renderer.h"
#include "cuda_types.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <memory>

// Host version of vec3_cuda for host code
struct vec3_cuda_host {
    double x, y, z;
    vec3_cuda_host() : x(0), y(0), z(0) {}
    vec3_cuda_host(double x, double y, double z) : x(x), y(y), z(z) {}
    operator vec3_cuda() const {
        vec3_cuda v;
        v.x = x;
        v.y = y;
        v.z = z;
        return v;
    }
};

// Forward declaration of kernel - definition is in raytracing_cuda.cu
__global__ void render_kernel(unsigned char* image, int image_width, int image_height, int samples_per_pixel,
                               int max_depth, point3_cuda lookfrom, point3_cuda lookat, vec3_cuda vup,
                               double vfov, double aspect_ratio, double defocus_angle, double focus_dist,
                               Sphere* spheres, int num_spheres, MaterialData* materials, int num_materials,
                               unsigned long long seed);

void CudaRenderer::convert_world_to_cuda(const hittable_list& world,
                                         std::vector<Sphere>& spheres,
                                         std::vector<MaterialData>& materials) {
    spheres.clear();
    materials.clear();
    
    // Create a mapping from shared_ptr<material> to material index
    std::vector<shared_ptr<material>> material_map;
    
    for (const auto& obj : world.objects) {
        auto sphere_ptr = std::dynamic_pointer_cast<sphere>(obj);
        if (!sphere_ptr) continue;
        
        // Find or create material
        int mat_index = -1;
        shared_ptr<material> mat = sphere_ptr->get_material();
        
        if (!mat) {
            std::cerr << "Warning: sphere has no material, skipping\n";
            continue;
        }
        
        for (size_t i = 0; i < material_map.size(); i++) {
            if (material_map[i] == mat) {
                mat_index = i;
                break;
            }
        }
        
        if (mat_index == -1) {
            mat_index = materials.size();
            material_map.push_back(mat);
            
            MaterialData mat_data;
            
            if (auto lambertian_ptr = std::dynamic_pointer_cast<lambertian>(mat)) {
                mat_data.type = LAMBERTIAN;
                color albedo = lambertian_ptr->get_albedo();
                mat_data.albedo.x = albedo.x();
                mat_data.albedo.y = albedo.y();
                mat_data.albedo.z = albedo.z();
                mat_data.fuzz = 0.0;
                mat_data.refraction_index = 0.0;
            } else if (auto metal_ptr = std::dynamic_pointer_cast<metal>(mat)) {
                mat_data.type = METAL;
                color albedo = metal_ptr->get_albedo();
                mat_data.albedo.x = albedo.x();
                mat_data.albedo.y = albedo.y();
                mat_data.albedo.z = albedo.z();
                mat_data.fuzz = metal_ptr->get_fuzz();
                mat_data.refraction_index = 0.0;
            } else if (auto dielectric_ptr = std::dynamic_pointer_cast<dielectric>(mat)) {
                mat_data.type = DIELECTRIC;
                mat_data.albedo.x = 1.0;
                mat_data.albedo.y = 1.0;
                mat_data.albedo.z = 1.0;
                mat_data.fuzz = 0.0;
                mat_data.refraction_index = dielectric_ptr->get_refraction_index();
            } else {
                std::cerr << "Warning: unknown material type, using default lambertian\n";
                mat_data.type = LAMBERTIAN;
                mat_data.albedo.x = 0.5;
                mat_data.albedo.y = 0.5;
                mat_data.albedo.z = 0.5;
                mat_data.fuzz = 0.0;
                mat_data.refraction_index = 0.0;
            }
            
            materials.push_back(mat_data);
        }
        
        Sphere sphere;
        point3 center = sphere_ptr->get_center();
        sphere.center.x = center.x();
        sphere.center.y = center.y();
        sphere.center.z = center.z();
        sphere.radius = sphere_ptr->get_radius();
        sphere.material_index = mat_index;
        
        spheres.push_back(sphere);
    }
    
    std::clog << "Converted " << spheres.size() << " spheres and " << materials.size() << " materials\n";
}

void CudaRenderer::render(const hittable_list& world, int image_width, int image_height,
                          int samples_per_pixel, int max_depth,
                          point3 lookfrom, point3 lookat, vec3 vup,
                          double vfov, double aspect_ratio,
                          double defocus_angle, double focus_dist) {
    
    // Convert world to CUDA structures
    std::vector<Sphere> spheres;
    std::vector<MaterialData> materials;
    convert_world_to_cuda(world, spheres, materials);
    
    if (spheres.empty()) {
        std::cerr << "No spheres to render!\n";
        return;
    }
    
    // Allocate device memory
    Sphere* d_spheres;
    MaterialData* d_materials;
    unsigned char* d_image;
    
    size_t spheres_size = spheres.size() * sizeof(Sphere);
    size_t materials_size = materials.size() * sizeof(MaterialData);
    size_t image_size = image_width * image_height * 3 * sizeof(unsigned char);
    
    cudaError_t err;
    
    err = cudaMalloc(&d_spheres, spheres_size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc error (spheres): " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    err = cudaMalloc(&d_materials, materials_size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc error (materials): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_spheres);
        return;
    }
    
    err = cudaMalloc(&d_image, image_size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc error (image): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_spheres);
        cudaFree(d_materials);
        return;
    }
    
    // Initialize image to zero
    cudaMemset(d_image, 0, image_size);
    
    // Copy data to device
    err = cudaMemcpy(d_spheres, spheres.data(), spheres_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy error (spheres): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_spheres);
        cudaFree(d_materials);
        cudaFree(d_image);
        return;
    }
    
    err = cudaMemcpy(d_materials, materials.data(), materials_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy error (materials): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_spheres);
        cudaFree(d_materials);
        cudaFree(d_image);
        return;
    }
    
    // Convert camera parameters
    point3_cuda cuda_lookfrom;
    cuda_lookfrom.x = lookfrom.x();
    cuda_lookfrom.y = lookfrom.y();
    cuda_lookfrom.z = lookfrom.z();
    
    point3_cuda cuda_lookat;
    cuda_lookat.x = lookat.x();
    cuda_lookat.y = lookat.y();
    cuda_lookat.z = lookat.z();
    
    vec3_cuda cuda_vup;
    cuda_vup.x = vup.x();
    cuda_vup.y = vup.y();
    cuda_vup.z = vup.z();
    
    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((image_width + blockSize.x - 1) / blockSize.x,
                  (image_height + blockSize.y - 1) / blockSize.y);
    
    unsigned long long seed = 12345;
    
    render_kernel<<<gridSize, blockSize>>>(
        d_image, image_width, image_height, samples_per_pixel, max_depth,
        cuda_lookfrom, cuda_lookat, cuda_vup,
        vfov, aspect_ratio, defocus_angle, focus_dist,
        d_spheres, spheres.size(), d_materials, materials.size(), seed
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_spheres);
        cudaFree(d_materials);
        cudaFree(d_image);
        return;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA device synchronize error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_spheres);
        cudaFree(d_materials);
        cudaFree(d_image);
        return;
    }
    
    // Copy result back
    unsigned char* image = new unsigned char[image_width * image_height * 3];
    err = cudaMemcpy(image, d_image, image_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy error (image back): " << cudaGetErrorString(err) << std::endl;
        delete[] image;
        cudaFree(d_spheres);
        cudaFree(d_materials);
        cudaFree(d_image);
        return;
    }
    
    // Output PPM image
    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    for (int j = 0; j < image_height; j++) {
        std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
        for (int i = 0; i < image_width; i++) {
            int idx = (j * image_width + i) * 3;
            std::cout << (int)image[idx] << ' ' 
                      << (int)image[idx+1] << ' ' 
                      << (int)image[idx+2] << '\n';
        }
    }
    std::clog << "\rDone.                 \n";
    
    // Cleanup
    delete[] image;
    cudaFree(d_spheres);
    cudaFree(d_materials);
    cudaFree(d_image);
}

