#ifndef ATHLETE_KERNEL_H
#define ATHLETE_KERNEL_H

#include "athlete.h"
#include "cuda_runtime.h"

// CUDA kernel fonksiyonunun deklarasyonu
__global__ void updateAthletePosition(athlete* athletes, int total_athletes , float swim_dist, float cycle_dist, float run_dist);
 void runRaceStages(athlete* d_athletes, int total_athletes, int num_blocks, int block_size, float swim_dist, float cycle_dist, float run_dist);
#endif // ATHLETE_KERNEL_H