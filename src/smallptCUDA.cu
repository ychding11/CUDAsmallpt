// based on smallpt, a path tracer by Kevin Beason, 2008  
 
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <chrono>  // for high_resolution_clock
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <vector_types.h>
#include "device_launch_parameters.h"
#include "cutil_math.h" // from http://www.icmc.usp.br/~castelo/CUDA/common/inc/cutil_math.h
#include "helper_string.h"


#define CUDA_CALL_CHECK(x)                             \
do{                                                    \
    if((x) != cudaSuccess)                             \
    {                                                  \
        cudaError_t cudaStatus = x;                    \
        printf("Error at %s:%d\t",__FILE__,__LINE__);  \
        printf("%s %d\t",#x, (cudaStatus));            \
        printf("%s\n",cudaGetErrorString(cudaStatus)); \
        system("pause");                               \
        return EXIT_FAILURE;                           \
    }                                                  \
} while(0)


#ifndef M_PI
#define M_PI 3.14159265359f  //< PI 
#endif

inline float clamp(float x)
{
	return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x;
} 

// convert RGB float in range [0.0, 1.0] to [0, 255] and perform gamma correction
inline int toInt(float x)
{
	return int(pow(clamp(x), 1 / 2.2) * 255 + .5);
} 

void SaveToPPM(float3* output, int w, int h, int count);

// random number generator from https://github.com/gz/rust-raytracer
__device__ static float getrandom(unsigned int *seed0, unsigned int *seed1)
{
 *seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16);  // hash the seeds using bitwise AND and bitshifts
 *seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);

 unsigned int ires = ((*seed0) << 16) + (*seed1);

 // Convert to float
 union
 {
  float f;
  unsigned int ui;
 } res;
 res.ui = (ires & 0x007fffff) | 0x40000000;  // bitwise AND, bitwise OR

 return (res.f - 2.f) / 2.f;
}

//! __device__ : executed on the device (GPU) and callable only from the device
struct Ray
{ 
 float3 o; //< ray origin
 float3 d;  //< ray direction 
 __device__ Ray(float3 o_, float3 d_) : o(o_), d(d_) {} 
};

enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance(), only DIFF used here

struct Sphere
{
	float rad;            // radius 
	float3 pos, emi, col; // position, emission, colour 
	Refl_t refl;          // reflection type (e.g. diffuse)

	// ray/sphere intersection
	// returns distance t to intersection point, 0 if no hit  
	// ray equation: p(x,y,z) = ray.orig + t*ray.dir
	// general sphere equation: x^2 + y^2 + z^2 = rad^2 
	// classic quadratic equation of form ax^2 + bx + c = 0 
	// solution x = (-b +- sqrt(b*b - 4ac)) / 2a
	// solve t^2*ray.dir*ray.dir + 2*t*(orig-p)*ray.dir + (orig-p)*(orig-p) - rad*rad = 0 
	// more details in "Realistic Ray Tracing" book by P. Shirley or Scratchapixel.com
	__device__ float intersect_sphere(const Ray &r) const 
	{ 
		float3 op = pos - r.o;    // distance from ray.orig to center sphere 
		float t, epsilon = 1e-4;  // epsilon required to prevent floating point precision artefacts
		float b = dot(op, r.d);    // b in quadratic equation
		float disc = b*b - dot(op, op) + rad*rad;  // discriminant quadratic equation
		if (disc<0) return 0;       // if disc < 0, no real solution (we're not interested in complex roots) 
		else disc = sqrtf(disc);    // if disc >= 0, check for solutions using negative and positive discriminant
		return (t = b - disc)>epsilon ? t : ((t = b + disc)>epsilon ? t : 0); // pick closest point in front of ray origin
	}
};

//! SCENE: small enough to be in constant GPU memory
//!		   9 spheres Cornell box
// { float radius, { float3 position }, { float3 emission }, { float3 colour }, refl_type }
__constant__ Sphere spheres[] =
{
 { 1e5f, { 1e5f + 1.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { 0.75f, 0.25f, 0.25f }, DIFF }, //Left 
 { 1e5f, { -1e5f + 99.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .25f, .25f, .75f }, DIFF }, //Rght 
 { 1e5f, { 50.0f, 40.8f, 1e5f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Back 
 { 1e5f, { 50.0f, 40.8f, -1e5f + 600.0f }, { 0.0f, 0.0f, 0.0f }, { 1.00f, 1.00f, 1.00f }, DIFF }, //Frnt 
 { 1e5f, { 50.0f, 1e5f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Botm 
 { 1e5f, { 50.0f, -1e5f + 81.6f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Top 
 { 16.5f, { 27.0f, 16.5f, 47.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, SPEC }, // small sphere 1
 { 16.5f, { 73.0f, 16.5f, 78.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, .0f, .0f }, DIFF }, // small sphere 2
 { 600.0f, { 50.0f, 681.6f - .77f, 81.6f }, { 2.0f, 1.8f, 1.6f }, { 0.0f, 0.0f, 0.0f }, DIFF }  // Light
};

// param t is distance to closest intersection, initialise t to a huge number outside scene
// param i is the intersected sphere id.
__device__ inline bool intersect_scene(const Ray &r, float &t, int &id)
{
	float n = sizeof(spheres) / sizeof(Sphere),
		  d,
		  inf = t = 1e20;  
	for (int i = int(n); i--;)  // test all scene objects for intersection
		if ((d = spheres[i].intersect_sphere(r)) && d<t) {  t = d; id = i; }
	return t < inf; // returns true if an intersection with the scene occurred, false when no hit
}

// radiance function, the meat of path tracing 
// solves the rendering equation: 
// outgoing radiance (at a point) = emitted radiance + reflected radiance
// reflected radiance is sum (integral) of incoming radiance from all directions in hemisphere above point, 
// multiplied by reflectance function of material (BRDF) and cosine incident angle 
// returns ray color
__device__ float3 radiance(Ray &r, curandState *randstate)
{ 
	float3 accucolor = make_float3(0.0f, 0.0f, 0.0f); // accumulates ray colour with each iteration through bounce loop
	float3 cf = make_float3(1.0f, 1.0f, 1.0f); 

	 for (int bounces = 0; bounces < 4; bounces++)
	 {  
		float t;           // distance to closest intersection 
		int id = 0;        // index of closest intersected sphere 

		// test ray for intersection with scene
		if (!intersect_scene(r, t, id))
		{
			return make_float3(0.0f, 0.0f, 0.0f); // if miss, return black
		}

		const Sphere &obj = spheres[id];  // hitobject
		float3 x = r.o+ r.d*t;          // hitpoint 
		float3 n = normalize(x - obj.pos);    // normal
		float3 nl = dot(n, r.d) < 0 ? n : n * -1; // front facing normal
		float3 d;

		accucolor += cf * obj.emi;

		if (obj.refl == DIFF)
		{
			float r1 = 2 * M_PI * curand_uniform(randstate); // pick random number on unit circle
			float r2 = curand_uniform(randstate);  // pick random number for elevation
			float r2s = sqrtf(r2); 

			float3 w = nl; 
			float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));  
			float3 v = cross(w,u);
			d = normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrtf(1 - r2));

			cf *= obj.col;    // multiply with colour of object       
			//cf *= dot(d,nl);  // weigh light contribution using cosine of angle between incident light and normal
			//cf *= 2;          // fudge factor
		}
		else if (obj.refl == SPEC)
		{
			d = r.d - n * 2 * dot(n, r.d);
		}
		else
		{

		}

		r.o = x + nl * 0.05f; // offset ray origin slightly to prevent self intersection
		r.d = d;
	 }
	 return accucolor;
}

//! __global__ : executed on the device (GPU) and callable only from host (CPU) 
__global__ void render_kernel(float3 *output, int width, int height, int samps, uint64_t hashedSeed)
{
    // blockIdx, blockDim and threadIdx are CUDA specific keywords
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int i = (height - 1 - y) * width + x; // index of current pixel (calculated using thread index) 

	// global threadId, see richiesams blogspot
	int gThreadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	// create random number generator, see RichieSams blogspot
	curandState randState; // state of the random number generator, to prevent repetition
	curand_init(hashedSeed + gThreadId, 0, 0, &randState);

    Ray cam(make_float3(50, 52, 295.6), normalize(make_float3(0, -0.042612, -1)));
    float3 cx = make_float3(width * .5135 / height, 0.0f, 0.0f);
    float3 cy = normalize(cross(cx, cam.d)) * .5135;
    float3 r = make_float3(0.0f);

	//// super sampling
	for (int sy = 0; sy < 2; sy++)     // 2x2 subpixel rows
		for (int sx = 0; sx < 2; sx++, r = make_float3(0.0f))			  // 2x2 subpixel cols
		{

			for (int s = 0; s < samps; s++)
			{  
				// compute primary ray direction
				//float3 d = cam.dir + cx*((.25 + x) / width - .5) + cy*((.25 + y) / height - .5);
				double r1 = 2 * curand_uniform(&randState), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
				double r2 = 2 * curand_uniform(&randState), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
				float3 d = cx * (((sx + .5 + dx) / 2 + x) / width - .5) +
					cy * (((sy + .5 + dy) / 2 + y) / height - .5) + cam.d;

				// create primary ray, add incoming radiance to pixelcolor
				r = r + radiance(Ray(cam.o + d * 140, normalize(d)), &randState)*(1. / samps) ;
			}
			output[i] += r * 0.25;
		}

}

//! This very important for cuda random pattern.
//! http://www.reedbeta.com/blog/2013/01/12/quick-and-easy-gpu-random-numbers-in-d3d11/
uint64_t WangHash(uint64_t a)
{
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

int TestSmallPTOnGPU(int width, int height, int samps)
{
    static float3* output_h = new float3[width * height]; // pointer to memory for image on the host (system RAM)
    float3* output_d;    // pointer to memory for image on the device (GPU VRAM)

	memset(output_h, 0, width * height * sizeof(float3));
    CUDA_CALL_CHECK( cudaSetDevice(0) );

    // allocate memory on the CUDA device (GPU VRAM)
    CUDA_CALL_CHECK( cudaMalloc(&output_d, width * height * sizeof(float3)) );
        
    dim3 block(8, 8, 1);   
    dim3 grid(width / block.x, height / block.y, 1);

    printf("\nStart rendering... %d, %d, %d\n", width, height, samps);
 
	uint64_t iterates = 0;
	int spsp = 2;

    auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < samps / 4 ; ++i, iterates++)
	{
    // Record start time                          
    auto start = std::chrono::high_resolution_clock::now();
		// Launch CUDA kernel from host
		render_kernel <<< grid, block >>>(output_d, width, height, spsp, WangHash(iterates));  
		// Check for any errors launching the kernel
		CUDA_CALL_CHECK(cudaGetLastError());
		CUDA_CALL_CHECK(cudaDeviceSynchronize());
    // Record end time
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    printf("\r- [Iterate %4llu] Kernel Done! Time=%3.5lf ms", iterates, 1000.f*elapsed.count());
	}

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    printf("\n- [Total Time] Render Done! Time=%3.5lf ms", 1000.f*elapsed.count());

    // copy results of computation from device back to host
    CUDA_CALL_CHECK(cudaMemcpy(output_h, output_d, width * height * sizeof(float3), cudaMemcpyDeviceToHost));
 
    // free CUDA memory
    CUDA_CALL_CHECK( cudaFree(output_d) );  

    SaveToPPM(output_h, width, height, samps / 4);

    printf("Saved image to 'smallptcuda.ppm'\n");
    delete[] output_h;
    return 0;
}

int main(int argc, char *argv[])
{
    int width = 1024, height = 1024, samps = 128;
    
    if (argc > 1)
    {
        if (checkCmdLineFlag(argc, (const char **)argv, "width"))
            width = getCmdLineArgumentInt(argc, (const char **)argv, "width");
        if (checkCmdLineFlag(argc, (const char **)argv, "height"))
            height = getCmdLineArgumentInt(argc, (const char **)argv, "height");
        if (checkCmdLineFlag(argc, (const char **)argv, "samples"))
            samps = getCmdLineArgumentInt(argc, (const char **)argv, "samples");
    }

    TestSmallPTOnGPU(width, height, samps);

    system("ffplay smallptcuda.ppm");
    system("PAUSE");
}

void SaveToPPM(float3* output, int w, int h, int count)
{
    // Write image to PPM file, a very simple image file format
    FILE *f = fopen("smallptcuda.ppm", "w");          
    fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
    for (int i = 0; i < w * h; i++)  // loop over pixels, write RGB values
    fprintf(f, "%d %d %d ", toInt(output[i].x / count),
                            toInt(output[i].y / count),
                            toInt(output[i].z / count));
    fclose(f);

}
