// CUDA Ray Tracing Implementation
// Based on the original code by Peter Shirley

#include "cuda_types.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iostream>

// Using vec3_cuda from cuda_types.h - add device functions
using point3_cuda = vec3_cuda;
using color_cuda = vec3_cuda;

// Device helper functions for vec3_cuda
__device__ vec3_cuda vec3_cuda_neg(const vec3_cuda& v) {
    return vec3_cuda{-v.x, -v.y, -v.z};
}

__device__ vec3_cuda vec3_cuda_add(const vec3_cuda& u, const vec3_cuda& v) {
    return vec3_cuda{u.x+v.x, u.y+v.y, u.z+v.z};
}

__device__ vec3_cuda vec3_cuda_sub(const vec3_cuda& u, const vec3_cuda& v) {
    return vec3_cuda{u.x-v.x, u.y-v.y, u.z-v.z};
}

__device__ vec3_cuda vec3_cuda_mul(const vec3_cuda& v, double t) {
    return vec3_cuda{t*v.x, t*v.y, t*v.z};
}

__device__ vec3_cuda vec3_cuda_mul_vec(const vec3_cuda& u, const vec3_cuda& v) {
    return vec3_cuda{u.x*v.x, u.y*v.y, u.z*v.z};
}

__device__ vec3_cuda vec3_cuda_div(const vec3_cuda& v, double t) {
    return vec3_cuda{v.x/t, v.y/t, v.z/t};
}

__device__ double vec3_cuda_length_squared(const vec3_cuda& v) {
    return v.x*v.x + v.y*v.y + v.z*v.z;
}

__device__ double vec3_cuda_length(const vec3_cuda& v) {
    return sqrt(vec3_cuda_length_squared(v));
}

__device__ bool vec3_cuda_near_zero(const vec3_cuda& v) {
    const auto s = 1e-8;
    return (fabs(v.x) < s) && (fabs(v.y) < s) && (fabs(v.z) < s);
}

__device__ double dot_cuda(const vec3_cuda& u, const vec3_cuda& v) {
    return u.x*v.x + u.y*v.y + u.z*v.z;
}

__device__ vec3_cuda cross_cuda(const vec3_cuda& u, const vec3_cuda& v) {
    return vec3_cuda{u.y*v.z - u.z*v.y,
                     u.z*v.x - u.x*v.z,
                     u.x*v.y - u.y*v.x};
}

__device__ vec3_cuda unit_vector_cuda(const vec3_cuda& v) {
    double len = vec3_cuda_length(v);
    if (len < 1e-8) {
        // Return a default unit vector if input is too small
        return vec3_cuda{1.0, 0.0, 0.0};
    }
    return vec3_cuda_div(v, len);
}

__device__ vec3_cuda random_unit_vector_cuda(curandState* state) {
    while (true) {
        auto p = vec3_cuda{curand_uniform_double(state)*2.0 - 1.0,
                          curand_uniform_double(state)*2.0 - 1.0,
                          curand_uniform_double(state)*2.0 - 1.0};
        auto lensq = vec3_cuda_length_squared(p);
        if (1e-160 < lensq && lensq <= 1.0)
            return vec3_cuda_div(p, sqrt(lensq));
    }
}

__device__ vec3_cuda random_in_unit_disk_cuda(curandState* state) {
    while (true) {
        auto p = vec3_cuda{curand_uniform_double(state)*2.0 - 1.0,
                          curand_uniform_double(state)*2.0 - 1.0,
                          0};
        if (vec3_cuda_length_squared(p) < 1)
            return p;
    }
}

__device__ vec3_cuda reflect_cuda(const vec3_cuda& v, const vec3_cuda& n) {
    return vec3_cuda_sub(v, vec3_cuda_mul(n, 2.0 * dot_cuda(v, n)));
}

__device__ vec3_cuda refract_cuda(const vec3_cuda& uv, const vec3_cuda& n, double etai_over_etat) {
    auto cos_theta = fmin(dot_cuda(vec3_cuda_neg(uv), n), 1.0);
    vec3_cuda r_out_perp = vec3_cuda_mul(vec3_cuda_add(uv, vec3_cuda_mul(n, cos_theta)), etai_over_etat);
    double perp_len_sq = vec3_cuda_length_squared(r_out_perp);
    if (perp_len_sq >= 1.0) {
        // Total internal reflection
        return reflect_cuda(uv, n);
    }
    vec3_cuda r_out_parallel = vec3_cuda_mul(n, -sqrt(1.0 - perp_len_sq));
    return vec3_cuda_add(r_out_perp, r_out_parallel);
}

struct ray_cuda {
    point3_cuda orig;
    vec3_cuda dir;
    
    __device__ ray_cuda() {}
    __device__ ray_cuda(const point3_cuda& origin, const vec3_cuda& direction) : orig(origin), dir(direction) {}
    __device__ point3_cuda at(double t) const { return vec3_cuda_add(orig, vec3_cuda_mul(dir, t)); }
};

using color_cuda = vec3_cuda;

struct HitRecord {
    point3_cuda p;
    vec3_cuda normal;
    double t;
    bool front_face;
    int material_index;
    
    __device__ void set_face_normal(const ray_cuda& r, const vec3_cuda& outward_normal) {
        front_face = dot_cuda(r.dir, outward_normal) < 0;
        normal = front_face ? outward_normal : vec3_cuda_neg(outward_normal);
    }
};

__device__ bool sphere_hit(const Sphere& sphere, const ray_cuda& r, double t_min, double t_max, HitRecord& rec) {
    vec3_cuda oc = vec3_cuda_sub(sphere.center, r.orig);
    auto a = vec3_cuda_length_squared(r.dir);
    
    // Avoid division by zero
    if (a < 1e-8) return false;
    
    auto h = dot_cuda(r.dir, oc);
    auto c = vec3_cuda_length_squared(oc) - sphere.radius * sphere.radius;
    
    auto discriminant = h*h - a*c;
    if (discriminant < 0) return false;
    
    auto sqrtd = sqrt(discriminant);
    auto root = (h - sqrtd) / a;
    if (root <= t_min || t_max <= root) {
        root = (h + sqrtd) / a;
        if (root <= t_min || t_max <= root)
            return false;
    }
    
    rec.t = root;
    rec.p = r.at(rec.t);
    vec3_cuda outward_normal = vec3_cuda_div(vec3_cuda_sub(rec.p, sphere.center), sphere.radius);
    rec.set_face_normal(r, outward_normal);
    rec.material_index = sphere.material_index;
    
    return true;
}

__device__ bool world_hit(const Sphere* spheres, int num_spheres, const ray_cuda& r, double t_min, double t_max, HitRecord& rec) {
    HitRecord temp_rec;
    bool hit_anything = false;
    auto closest_so_far = t_max;
    
    for (int i = 0; i < num_spheres; i++) {
        if (sphere_hit(spheres[i], r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    
    return hit_anything;
}

__device__ double reflectance_cuda(double cosine, double refraction_index) {
    auto r0 = (1.0 - refraction_index) / (1.0 + refraction_index);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
}

__device__ bool scatter_lambertian(const MaterialData& mat, const ray_cuda& r_in, const HitRecord& rec, 
                                    color_cuda& attenuation, ray_cuda& scattered, curandState* state) {
    auto scatter_direction = vec3_cuda_add(rec.normal, random_unit_vector_cuda(state));
    
    if (vec3_cuda_near_zero(scatter_direction))
        scatter_direction = rec.normal;
    
    scattered = ray_cuda(rec.p, scatter_direction);
    attenuation = mat.albedo;
    return true;
}

__device__ bool scatter_metal(const MaterialData& mat, const ray_cuda& r_in, const HitRecord& rec,
                               color_cuda& attenuation, ray_cuda& scattered, curandState* state) {
    vec3_cuda reflected = reflect_cuda(unit_vector_cuda(r_in.dir), rec.normal);
    reflected = vec3_cuda_add(unit_vector_cuda(reflected), vec3_cuda_mul(random_unit_vector_cuda(state), mat.fuzz));
    scattered = ray_cuda(rec.p, reflected);
    attenuation = mat.albedo;
    return (dot_cuda(scattered.dir, rec.normal) > 0);
}

__device__ bool scatter_dielectric(const MaterialData& mat, const ray_cuda& r_in, const HitRecord& rec,
                                    color_cuda& attenuation, ray_cuda& scattered, curandState* state) {
    attenuation = color_cuda{1.0, 1.0, 1.0};
    double ri = rec.front_face ? (1.0 / mat.refraction_index) : mat.refraction_index;
    
    vec3_cuda unit_direction = unit_vector_cuda(r_in.dir);
    double cos_theta = fmin(dot_cuda(vec3_cuda_neg(unit_direction), rec.normal), 1.0);
    double sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    
    bool cannot_refract = ri * sin_theta > 1.0;
    vec3_cuda direction;
    
    if (cannot_refract || reflectance_cuda(cos_theta, ri) > curand_uniform_double(state))
        direction = reflect_cuda(unit_direction, rec.normal);
    else
        direction = refract_cuda(unit_direction, rec.normal, ri);
    
    scattered = ray_cuda(rec.p, direction);
    return true;
}

__device__ bool scatter_material(const MaterialData* materials, int num_materials, const ray_cuda& r_in, const HitRecord& rec,
                                  color_cuda& attenuation, ray_cuda& scattered, curandState* state) {
    if (rec.material_index < 0 || rec.material_index >= num_materials)
        return false;
    
    const MaterialData& mat = materials[rec.material_index];
    
    if (mat.type == LAMBERTIAN)
        return scatter_lambertian(mat, r_in, rec, attenuation, scattered, state);
    else if (mat.type == METAL)
        return scatter_metal(mat, r_in, rec, attenuation, scattered, state);
    else if (mat.type == DIELECTRIC)
        return scatter_dielectric(mat, r_in, rec, attenuation, scattered, state);
    
    return false;
}

__device__ color_cuda ray_color_cuda(const ray_cuda& r, int max_depth, const Sphere* spheres, int num_spheres,
                                     const MaterialData* materials, int num_materials, curandState* state) {
    ray_cuda current_ray = r;
    color_cuda accumulated_color{1.0, 1.0, 1.0};
    int depth = max_depth;
    
    while (depth > 0) {
        HitRecord rec;
        
        if (world_hit(spheres, num_spheres, current_ray, 0.001, 1e10, rec)) {
            ray_cuda scattered;
            color_cuda attenuation;
            if (scatter_material(materials, num_materials, current_ray, rec, attenuation, scattered, state)) {
                // Multiply accumulated color by attenuation
                accumulated_color = vec3_cuda_mul_vec(accumulated_color, attenuation);
                // Continue with scattered ray
                current_ray = scattered;
                depth--;
            } else {
                // Ray was absorbed
                return color_cuda{0, 0, 0};
            }
        } else {
            // Ray hit nothing - return sky color
            vec3_cuda unit_direction = unit_vector_cuda(current_ray.dir);
            auto a = 0.5 * (unit_direction.y + 1.0);
            color_cuda c1 = vec3_cuda_mul(color_cuda{1.0, 1.0, 1.0}, 1.0 - a);
            color_cuda c2 = vec3_cuda_mul(color_cuda{0.5, 0.7, 1.0}, a);
            color_cuda sky_color = vec3_cuda_add(c1, c2);
            return vec3_cuda_mul_vec(accumulated_color, sky_color);
        }
    }
    
    // Max depth reached - return black
    return color_cuda{0, 0, 0};
}

__global__ void render_kernel(unsigned char* image, int image_width, int image_height, int samples_per_pixel,
                               int max_depth, point3_cuda lookfrom, point3_cuda lookat, vec3_cuda vup,
                               double vfov, double aspect_ratio, double defocus_angle, double focus_dist,
                               Sphere* spheres, int num_spheres, MaterialData* materials, int num_materials,
                               unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= image_width || j >= image_height) return;
    
    curandState state;
    curand_init(seed + i * image_height + j, 0, 0, &state);
    
    // Camera setup
    auto theta = vfov * 3.1415926535897932385 / 180.0;
    auto h = tan(theta / 2.0);
    auto viewport_height = 2.0 * h * focus_dist;
    auto viewport_width = viewport_height * aspect_ratio;
    
    vec3_cuda w_dir = vec3_cuda_sub(lookfrom, lookat);
    if (vec3_cuda_length_squared(w_dir) < 1e-8) {
        // If lookfrom and lookat are too close, use default direction
        w_dir = vec3_cuda{0.0, 0.0, -1.0};
    }
    vec3_cuda w = unit_vector_cuda(w_dir);
    vec3_cuda u = unit_vector_cuda(cross_cuda(vup, w));
    vec3_cuda v = cross_cuda(w, u);
    
    vec3_cuda viewport_u = vec3_cuda_mul(u, viewport_width);
    vec3_cuda viewport_v = vec3_cuda_mul(v, viewport_height * -1.0);
    
    vec3_cuda pixel_delta_u = vec3_cuda_div(viewport_u, image_width);
    vec3_cuda pixel_delta_v = vec3_cuda_div(viewport_v, image_height);
    
    auto defocus_radius = focus_dist * tan((defocus_angle / 2.0) * 3.1415926535897932385 / 180.0);
    vec3_cuda defocus_disk_u = vec3_cuda_mul(u, defocus_radius);
    vec3_cuda defocus_disk_v = vec3_cuda_mul(v, defocus_radius);
    
    point3_cuda viewport_upper_left = vec3_cuda_sub(vec3_cuda_sub(vec3_cuda_sub(lookfrom, vec3_cuda_mul(w, focus_dist)), vec3_cuda_mul(viewport_u, 0.5)), vec3_cuda_mul(viewport_v, 0.5));
    point3_cuda pixel00_loc = vec3_cuda_add(viewport_upper_left, vec3_cuda_mul(vec3_cuda_add(pixel_delta_u, pixel_delta_v), 0.5));
    
    color_cuda pixel_color{0, 0, 0};
    
    for (int sample = 0; sample < samples_per_pixel; sample++) {
        auto offset = vec3_cuda{curand_uniform_double(&state) - 0.5,
                                curand_uniform_double(&state) - 0.5,
                                0};
        point3_cuda pixel_sample = vec3_cuda_add(pixel00_loc, vec3_cuda_add(vec3_cuda_mul(pixel_delta_u, i + offset.x), vec3_cuda_mul(pixel_delta_v, j + offset.y)));
        
        point3_cuda ray_origin = lookfrom;
        if (defocus_angle > 0) {
            auto p = random_in_unit_disk_cuda(&state);
            ray_origin = vec3_cuda_add(lookfrom, vec3_cuda_add(vec3_cuda_mul(defocus_disk_u, p.x), vec3_cuda_mul(defocus_disk_v, p.y)));
        }
        
        vec3_cuda ray_dir = vec3_cuda_sub(pixel_sample, ray_origin);
        // Skip if ray direction is too small
        if (vec3_cuda_length_squared(ray_dir) < 1e-8) {
            continue;
        }
        ray_cuda r(ray_origin, ray_dir);
        pixel_color = vec3_cuda_add(pixel_color, ray_color_cuda(r, max_depth, spheres, num_spheres, materials, num_materials, &state));
    }
    
    if (samples_per_pixel > 0) {
        pixel_color = vec3_cuda_div(pixel_color, samples_per_pixel);
    }
    
    // Gamma correction
    auto r = sqrt(pixel_color.x);
    auto g = sqrt(pixel_color.y);
    auto b = sqrt(pixel_color.z);
    
    // Clamp to [0, 0.999]
    r = fmax(0.0, fmin(0.999, r));
    g = fmax(0.0, fmin(0.999, g));
    b = fmax(0.0, fmin(0.999, b));
    
    // Verify index is within bounds
    int idx = (j * image_width + i) * 3;
    int max_idx = image_width * image_height * 3;
    if (idx >= 0 && idx + 2 < max_idx) {
        image[idx + 0] = (unsigned char)(256 * r);
        image[idx + 1] = (unsigned char)(256 * g);
        image[idx + 2] = (unsigned char)(256 * b);
    }
}

__global__ void init_curand_states(curandState* states, unsigned long long seed, int num_threads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_threads) {
        curand_init(seed + idx, 0, 0, &states[idx]);
    }
}

