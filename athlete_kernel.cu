#include "athlete_kernel.h"  // Artık kernel başlık dosyasını include ediyoruz

__global__ void updateAthletePosition(athlete* athletes, int numAthletes , float swim_dist, float cycle_dist, float run_dist) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numAthletes) {
        // Sporcuların pozisyonunu güncelle (her turda hızlarına göre pozisyon değişmeli)
        athletes[id].updatePosition(swim_dist, cycle_dist, run_dist);

        // Hızı her döngüde değil, yalnızca aşama değişimlerinde güncelle
        if (athletes[id].getCurrentStage() != athletes[id].getPreviousStage()) {
            athletes[id].update_speed();
        }
    }
}

// Yarış aşamalarını çalıştıran fonksiyon
void runRaceStages(athlete* d_athletes, int total_athletes, int num_blocks, int block_size, float swim_dist, float cycle_dist, float run_dist) {
    // GPU üzerinde pozisyonları güncelleyen kernel'i çağır
    updateAthletePosition<<<num_blocks, block_size>>>(d_athletes, total_athletes, swim_dist, cycle_dist, run_dist);  
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "CUDA Kernel Error: " << cudaGetErrorString(error) << std::endl;
    }
}


