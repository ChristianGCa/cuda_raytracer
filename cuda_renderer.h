#ifndef CUDA_RENDERER_H
#define CUDA_RENDERER_H

#include "rtweekend.h"
#include "hittable_list.h"
#include "sphere.h"
#include "material.h"
#include <vector>

// Forward declarations for CUDA structures
struct Sphere;
struct MaterialData;

class CudaRenderer {
public:
    void render(const hittable_list& world, int image_width, int image_height,
                int samples_per_pixel, int max_depth,
                point3 lookfrom, point3 lookat, vec3 vup,
                double vfov, double aspect_ratio,
                double defocus_angle, double focus_dist);

private:
    void convert_world_to_cuda(const hittable_list& world,
                                std::vector<Sphere>& spheres,
                                std::vector<MaterialData>& materials);
};

#endif

