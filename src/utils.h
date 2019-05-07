#ifndef RAYH_UTILS_H
#define RAYH_UTILS_H

#include <cstdlib>
#include <limits>


#define M_PI 3.14159265359f  // pi

#define MAXFLOAT 1.7e38f

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
// SRAND48 and DRAND48 don't exist on windows, but these are the equivalent functions
	void srand48(long seed)
	{
		srand((unsigned int)seed);
	}
	double drand48()
	{
		return double(rand())/RAND_MAX;
	}
#endif
#endif 


