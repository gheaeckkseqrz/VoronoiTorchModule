#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define THREAD_COUNT 1024

__global__ void init(float *input, float *map, int w, int h)
{
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < w * h)
    {
      int x = index % w;
      int y = index / w;

      if (input[(0 * (w * h)) + (y * w) + x] != 0 || input[(1 * (w * h)) + (y * w) + x] != 0 || input[(2 * (w * h)) + (y * w) + x] != 0)
      	{
      	  map[(0 * (w * h)) + (y * w) + x] = (float)x;
      	  map[(1 * (w * h)) + (y * w) + x] = (float)y;
	  map[(2 * (w * h)) + (y * w) + x] = (float)0;
      	}
      else
      	{
      	  map[(0 * (w * h)) + (y * w) + x] = (float)-1;
      	  map[(1 * (w * h)) + (y * w) + x] = (float)-1;
      	  map[(2 * (w * h)) + (y * w) + x] = (float)-1;
      	}
    }
}

__global__ void run(float *map, float *out, int w, int h, unsigned int stride)
{
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < w * h)
    {
      unsigned int x = index % w;
      unsigned int y = index / w;

      int n[9][2] = {
	{-1, -1},
	{-1,  0},
	{-1,  1},
	{ 0, -1},
	{ 0,  0},
	{ 0,  1},
	{ 1, -1},
	{ 1,  0},
	{ 1,  1}
      };

      float bestScore = -1;
      int best = -1;
      for (unsigned int i=0 ; i < 9 ; ++i)
	{
	  int sx = x + (n[i][0] * stride);
	  int sy = y + (n[i][1] * stride);
	  if (sx >= 0 && sx < w && sy >= 0 && sy < h)
	    {
	      if (map[(2 * (w * h)) + (sy * w) + sx] >= 0)
		{
		  float score = sqrt(pow(map[(0 * (w * h)) + (sy * w) + sx] - x, 2) + pow(map[(1 * (w * h)) + (sy * w) + sx] - y, 2));
		  if (score < bestScore || bestScore < 0)
		    {
		      best = i;
		      bestScore = score;
		    }
		}
	    }
	}
      if (best >= 0)
	{
	  int sx = x + n[best][0] * stride;
	  int sy = y + n[best][1] * stride;
	  out[(0 * (w * h)) + (y * w) + x] = map[(0 * (w * h)) + (sy * w) + sx];
	  out[(1 * (w * h)) + (y * w) + x] = map[(1 * (w * h)) + (sy * w) + sx];
	  out[(2 * (w * h)) + (y * w) + x] = bestScore;
	}
    }
}

__global__ void finish(float *input, float *map, int w, int h)
{
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < w * h)
    {
      unsigned int x = index % w;
      unsigned int y = index / w;
      int mx = map[(0 * (w * h)) + (y * w) + x];
      int my = map[(1 * (w * h)) + (y * w) + x];

      if (mx >= 0 && my >= 0)
	{
	  input[(0 * (w * h)) + (y * w) + x] = input[(0 * (w * h)) + (my * w) + mx];
	  input[(1 * (w * h)) + (y * w) + x] = input[(1 * (w * h)) + (my * w) + mx];
	  input[(2 * (w * h)) + (y * w) + x] = input[(2 * (w * h)) + (my * w) + mx];
	}
    }
}

extern "C" void computeVoronoi(float *input, int w, int h)
{
  float *ping = NULL;
  float *pong = NULL;

  cudaMalloc(&ping, 3 * w * h * sizeof(float)); // 3 Channels - Closest point X / Closest Point Y / Distance
  cudaMalloc(&pong, 3 * w * h * sizeof(float)); // 3 Channels - Closest point X / Closest Point Y / Distance

  init<<<(w*h) / THREAD_COUNT + 1, THREAD_COUNT>>>(input, ping, w, h);
  init<<<(w*h) / THREAD_COUNT + 1, THREAD_COUNT>>>(input, pong, w, h);

  run<<<(w*h) / THREAD_COUNT + 1, THREAD_COUNT>>>(ping, pong, w, h, 128);
  run<<<(w*h) / THREAD_COUNT + 1, THREAD_COUNT>>>(pong, ping, w, h, 64);
  run<<<(w*h) / THREAD_COUNT + 1, THREAD_COUNT>>>(ping, pong, w, h, 32);
  run<<<(w*h) / THREAD_COUNT + 1, THREAD_COUNT>>>(pong, ping, w, h, 16);
  run<<<(w*h) / THREAD_COUNT + 1, THREAD_COUNT>>>(ping, pong, w, h, 8);
  run<<<(w*h) / THREAD_COUNT + 1, THREAD_COUNT>>>(pong, ping, w, h, 4);
  run<<<(w*h) / THREAD_COUNT + 1, THREAD_COUNT>>>(ping, pong, w, h, 2);
  run<<<(w*h) / THREAD_COUNT + 1, THREAD_COUNT>>>(pong, ping, w, h, 1);
  run<<<(w*h) / THREAD_COUNT + 1, THREAD_COUNT>>>(ping, pong, w, h, 1);

  finish<<<(w*h) / THREAD_COUNT + 1, THREAD_COUNT>>>(input, pong, w, h);

  cudaFree(ping);
  cudaFree(pong);
}

