#ifndef CAMERAH
#define CAMERAH

#include <curand.h>
#include <curand_kernel.h>

#include "ray.h"
#include "utils.h"

__device__ vec3 random_in_unit_disk(curandState& randState)
{
    vec3 p;
    do
    {
        float rx = curand_uniform(&randState);
        float ry = curand_uniform(&randState);
        p = 2.0f * vec3(rx, ry, 0) - vec3(1.f, 1.f,0.f);
    } while (dot(p,p) >= 1.0f);
    return p;
}

class camera
{
    public:
         __host__ __device__ camera()
         {}

        // vfov is top to bottom in degrees
         __host__ __device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focus_dist)
        { 
            lens_radius = aperture / 2.f;
            float theta = vfov * M_PI/180.f;
            float half_height = tan(theta/2);
            float half_width  = aspect * half_height;
            origin = lookfrom;
            w = unit_vector(lookfrom - lookat);
            u = unit_vector(cross(vup, w));
            v = cross(w, u);
            lower_left_corner = origin  - half_width*focus_dist*u -half_height*focus_dist*v - focus_dist*w;
            horizontal = 2*half_width*focus_dist*u;
            vertical   = 2*half_height*focus_dist*v;
        }

         __device__ ray get_ray(float s, float t, curandState& randState)
        {
            vec3 rd = lens_radius*random_in_unit_disk(randState);
            vec3 offset = u * rd.x() + v * rd.y();
            return ray(origin + offset, lower_left_corner + s*horizontal + t*vertical - origin - offset); 
        }

        vec3 origin;
        vec3 lower_left_corner;
        vec3 horizontal;
        vec3 vertical;
        vec3 u, v, w;
        float lens_radius;
};
#endif




