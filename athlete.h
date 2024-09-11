#ifndef ATHLETE_H
#define ATHLETE_H

#include <iostream>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "algorithm"
// Enum tanımını buraya taşıyın
enum stage {
    SWIMMING,
    CYCLING,
    RUNNING
};

class athlete {
private:
    stage previous_stage;  // Önceki aşamayı tutar
    stage current_stage;  // Sporcuların mevcut aşamasını tutar
    float speed;
    float position;
    float total_time;
    float transition_time = 10.0f;  // Aşama değişim süresi

public:

    athlete() 
        : speed(0), position(0), total_time(0), current_stage(SWIMMING), previous_stage(SWIMMING) {}

    athlete(float swim_dist, float cycle_dist, float run_dist) 
       : speed(0), position(0), total_time(0), current_stage(SWIMMING), previous_stage(SWIMMING) {}

    void initialize_speed() { // 1-5 m/s arası rastgele hız ataması yapar
        speed = 1 + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (5 - 1)));
    }
    // Copy constructor
    athlete(const athlete& other) 
        : speed(other.speed), position(other.position), total_time(other.total_time),
          current_stage(other.current_stage){}

    // Assignment operator
    athlete& operator=(const athlete& other) {
        if (this != &other) {
            this->speed = other.speed;
            this->position = other.position;
            this->total_time = other.total_time;
            this->current_stage = other.current_stage;
        }
        return *this;
    }
    
     __host__ __device__ void update_speed() {
        if (current_stage != previous_stage) {  // Aşama değiştiyse hızı güncelle
            if (current_stage == SWIMMING) {
                // Swimming için hız aynı kalabilir
            } else if (current_stage == CYCLING) {
                speed *= 3.0f;  // Bisiklet aşamasında hız 3 katına çıkıyor
            } else if (current_stage == RUNNING) {
                speed /= 9.0f;  // Koşu aşamasında hız ilk hızın  1/3 ' üne düşüyor
            }
        }
        previous_stage = current_stage;  // Geçişten sonra aşamayı güncelle
    }


      __host__ __device__ void updatePosition(float swim_dist, float cycle_dist, float run_dist) {
        if (position >= swim_dist + cycle_dist + run_dist) {
            return;  // Yarışı tamamladıysa pozisyonu güncelleme
        }
        position += speed;

        if (position > swim_dist + cycle_dist + run_dist) {
            position = swim_dist + cycle_dist + run_dist;  // Yarışı tamamladıysa pozisyonu sınırlandır
        }

        total_time += 1.0f; // Her adımda 1 saniye geçer
        checkStageProgress(swim_dist, cycle_dist, run_dist);  // Sporcu mesafeyi tamamladı mı kontrol edilir
    }

    __host__ __device__ void checkStageProgress(float swim_dist, float cycle_dist, float run_dist) {
        if (current_stage == SWIMMING && position >= swim_dist) {
            transitionToNextStage(CYCLING);
        } else if (current_stage == CYCLING && position >= swim_dist + cycle_dist) {
            transitionToNextStage(RUNNING);
        }
    }

    __host__ __device__ void transitionToNextStage(stage next_stage) {
        total_time += transition_time;  // Geçiş süresini ekle
        current_stage = next_stage;
    }

    __host__ __device__ float getPosition() const { return position; }
    __host__ __device__ float getSpeed() const { return speed; }
    __host__ __device__ float getTotalTime() const { return total_time; }
    __host__ __device__ stage getCurrentStage() const { return current_stage; }
    __host__ __device__ stage getPreviousStage() const { return previous_stage; }
};

#endif // ATHLETE_H