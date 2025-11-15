#ifndef CUDA_TYPES_H
#define CUDA_TYPES_H

// Shared CUDA types between .cu files
#ifdef __CUDACC__
    #define CUDA_HOST_DEVICE __host__ __device__
#else
    #define CUDA_HOST_DEVICE
#endif

// Basic structure definition - methods are in raytracing_cuda.cu
struct vec3_cuda {
    double x, y, z;
};

using point3_cuda = vec3_cuda;

enum MaterialType {
    LAMBERTIAN,
    METAL,
    DIELECTRIC
};

struct MaterialData {
    MaterialType type;
    vec3_cuda albedo;
    double fuzz;
    double refraction_index;
};

struct Sphere {
    point3_cuda center;
    double radius;
    int material_index;
};

#endif

