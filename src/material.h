#ifndef MATERIALH
#define MATERIALH 

#include <curand.h>
#include <curand_kernel.h>

struct hit_record;

#include "ray.h"
#include "hitable.h"

__device__ 
float schlick(float cosine, float ref_idx)
{
    float r0 = (1-ref_idx) / (1+ref_idx);
    r0 = r0*r0;
    return r0 + (1-r0)*pow((1 - cosine),5);
}

__device__ 
bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted)
{
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0 - ni_over_nt*ni_over_nt*(1-dt*dt);
    if (discriminant > 0)
    {
        refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
        return true;
    }
    else 
        return false;
}


__device__ 
vec3 reflect(const vec3& v, const vec3& n)
{
     return v - 2*dot(v,n)*n;
}


// sample a unit sphere in a [-1, 1] cube
__device__ 
vec3 random_in_unit_sphere(curandState& randState)
{
    vec3 p;
    do
    {
        float rx = curand_uniform(&randState);
        float ry = curand_uniform(&randState);
        float rz = curand_uniform(&randState);
        p = 2.0 * vec3(rx, ry,rz) - vec3(1.f,1.f,1.f);
    } while (p.squared_length() >= 1.0);
    return p;
}


class material 
{
    public:

        // calculate "scatterec ray" and "attenuation to ray" at hit point
        __device__ 
        virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState& randState) const = 0;
};

class lambertian : public material
{
    public:
        
        __host__ __device__ 
        lambertian(const vec3& a) : albedo(a) {}

        // sample a unit sphere tangent to hit point
        __device__ 
        virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState& randState) const 
        {
             vec3 target = rec.p + rec.normal + random_in_unit_sphere(randState);
             scattered = ray(rec.p, target-rec.p);
             attenuation = albedo;
             return true;
        }

        vec3 albedo;
};

class metal : public material
{
    public:

        __host__ __device__ 
        metal(const vec3& a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }

        __device__ 
        virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState& randState) const 
        {
            vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
            scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere(randState));
            attenuation = albedo;
            return (dot(scattered.direction(), rec.normal) > 0);
        }

        vec3 albedo;
        float fuzz;
};

class dielectric : public material
{ 
    public:

        __host__ __device__ 
        dielectric(float ri) : ref_idx(ri) {}

        __device__ 
        virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState& randState) const 
        {
             vec3 outward_normal;
             vec3 reflected = reflect(r_in.direction(), rec.normal);
             float ni_over_nt;
             attenuation = vec3(1.0, 1.0, 1.0); 
             vec3 refracted;
             float reflect_prob;
             float cosine;
             if (dot(r_in.direction(), rec.normal) > 0)
             {
                  outward_normal = -rec.normal;
                  ni_over_nt = ref_idx;
         //         cosine = ref_idx * dot(r_in.direction(), rec.normal) / r_in.direction().length();
                  cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
                  cosine = sqrt(1 - ref_idx*ref_idx*(1-cosine*cosine));
             }
             else
             {
                  outward_normal = rec.normal;
                  ni_over_nt = 1.0 / ref_idx;
                  cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
             }
             if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted)) 
                reflect_prob = schlick(cosine, ref_idx);
             else 
                reflect_prob = 1.0;

             float guess = curand_uniform(&randState);
#if 1
             if (guess < reflect_prob) 
                scattered = ray(rec.p, reflected);
             else 
                scattered = ray(rec.p, refracted);
#endif
             return true;
        }

        float ref_idx;
};

#endif




