#ifndef TEAM_H
#define TEAM_H

#include <iostream>
#include "athlete.h"  // athlete sınıfını dahil ediyoruz

class team {
private:
    athlete athletes[3];  // Her takımda 3 sporcu var
    float total_team_time;  // Takımın toplam süresi

public:
    team() : total_team_time(0) {}

    // Takımdaki sporcuların hızlarını başlat
    void initialize_athletes() {
        for (int i = 0; i < 3; i++) {
            athletes[i].initialize_speed();
        }
    }
    // Takımın toplam süresini hesaplar
    float calculate_total_team_time() {
        total_team_time = 0;
        for (int i = 0; i < 3; ++i) {
            total_team_time += athletes[i].getTotalTime();
        }
        return total_team_time;
    }

    // Takımdaki belirli bir sporcuyu döndürür
    athlete& getAthlete(int i) {
        return athletes[i];
    }

    // Takımdaki belirli bir sporcuyu günceller
    void setAthlete(int i, const athlete& ath) {
        athletes[i] = ath;
    }
    
};

#endif // TEAM_H