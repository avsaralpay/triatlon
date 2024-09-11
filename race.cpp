#include <iostream>
#include <stdlib.h>
#include "team.h"         
#include "athlete_kernel.h"
#include <chrono>
#include <algorithm>
#include <thread>
#include <vector>

class race {

private:
    team teams[300];  // 300 takımdan oluşan yarış
    int first_place_id = -1;  // Birinci olan atletin id'si
    int first_place_team_id = -1;  // Birinci olan takımın id'si
    float swimming_distance;
    float cycling_distance;
    float running_distance;

public:
    race(float swim_dist, float cycle_dist, float run_dist) 
            : swimming_distance(swim_dist), cycling_distance(cycle_dist), running_distance(run_dist) {}

    team getTeam(int index) { return teams[index]; }

    void checkAthletesOnCPU(athlete* d_athletes, athlete* h_athletes, int total_athletes) {
        // GPU'dan CPU'ya kopyala
        cudaMemcpy(h_athletes, d_athletes, total_athletes * sizeof(athlete), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }

    // Yarıştaki takımları başlatır, her takımdaki atletlerin hızlarını rastgele atar
    void initialize_teams() {
        for (int i = 0; i < 300; i++) {
            teams[i].initialize_athletes();
        }
    }

    void printAthletePosition(athlete* h_athletes, int team_index, int athlete_index) {
        int athlete_id = team_index * 3 + athlete_index;

        // Süreyi dakika ve saniyeye çevir
        int minutes = static_cast<int>(h_athletes[athlete_id].getTotalTime()) / 60;
        int seconds = static_cast<int>(h_athletes[athlete_id].getTotalTime()) % 60;

        std::cout << "Takım " << team_index << " - Sporcu " << athlete_index << " Hız: " 
                << h_athletes[athlete_id].getSpeed() << " m/s, Konum: " 
                << h_athletes[athlete_id].getPosition() << " metre, Süre: "
                << minutes << " dakika " << seconds << " saniye" << std::endl;
    }

    void printAllPositions(athlete* h_athletes, int total_athletes) { 
        for (int i = 0; i < total_athletes; ++i) {
            // Süreyi dakika ve saniyeye çevir
            int minutes = static_cast<int>(h_athletes[i].getTotalTime()) / 60;
            int seconds = static_cast<int>(h_athletes[i].getTotalTime()) % 60;

            std::cout << "Athlete " << i << " Konum: " << h_athletes[i].getPosition() << " metre, "
                    << "Süre: " << minutes << " dakika " << seconds << " saniye" << std::endl;
        }
    }

    void printTeamResults(athlete* h_athletes, int total_athletes) {
        std::vector<std::pair<int, float>> team_times;  // Takım numarası ve toplam süre

        // Her takımın toplam süresini hesapla
        for (int i = 0; i < 300; ++i) {
            float total_time = 0;
            for (int j = 0; j < 3; ++j) {  // Her takımda 3 sporcu
                int athlete_id = i * 3 + j;
                total_time += h_athletes[athlete_id].getTotalTime();  // Her atletin toplam süresi
            }
            team_times.push_back(std::make_pair(i + 1, total_time));  // Takım numarasını ve süreyi kaydet
        }

        // Takımları süreye göre sıralama (küçükten büyüğe)
        std::sort(team_times.begin(), team_times.end(),
                [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                    return a.second < b.second;
                });

        // Kazanan takımın (team_times[0]) en hızlı atletini bul
        int winning_team = team_times[0].first - 1;  // Kazanan takımın indeksini al (team_times[0].first ile 1'e ekleme yapıldığı için)
        int best_athlete_index = -1;
        float best_time = std::numeric_limits<float>::max();  // Minimum zamanı bulmak için başlat

        // Kazanan takımın en hızlı atleti için süreleri kontrol et
        for (int j = 0; j < 3; ++j) {
            int athlete_id = winning_team * 3 + j;
            if (h_athletes[athlete_id].getTotalTime() < best_time) {
                best_time = h_athletes[athlete_id].getTotalTime();
                best_athlete_index = j;  // Takım içindeki en iyi atleti belirle
            }
        }

        // Birinci olan atletin bilgilerini yazdır
        std::cout << "Birinci olan atletin bilgileri:\n";
        printAthletePosition(h_athletes, winning_team, best_athlete_index);  // En hızlı atleti yazdır

        // Yarışı kazanan takımı yazdır (süreyi dakika ve saniye olarak)
        int team_minutes = static_cast<int>(team_times[0].second) / 60;
        int team_seconds = static_cast<int>(team_times[0].second) % 60;
        std::cout << "Yarışı kazanan takım: " << team_times[0].first << " - Toplam Süre: "
                << team_minutes << " dakika " << team_seconds << " saniye\n";

        // Tüm takım sıralamalarını yazdır
        std::cout << "Tüm Takım Sıralamaları:\n";
        for (const auto& team : team_times) {
            int minutes = static_cast<int>(team.second) / 60;
            int seconds = static_cast<int>(team.second) % 60;
            std::cout << "Takım " << team.first << " - Toplam Süre: " 
                    << minutes << " dakika " << seconds << " saniye\n";
        }
}

    // Yarış simülasyonunu başlatır
    void startRace(int team_index, int athlete_index) {
        int total_athletes = 300 * 3;
        athlete* d_athletes;
        athlete* h_athletes = new athlete[total_athletes];

        // Host bellekteki sporcuları doldur
        int index = 0;
        for (int i = 0; i < 300; ++i) {
            for (int j = 0; j < 3; ++j) {
                h_athletes[index] = teams[i].getAthlete(j);
                ++index;
            }
        }

        // GPU belleği için sporcuları ayır ve kopyala
        cudaMalloc((void**)&d_athletes, total_athletes * sizeof(athlete));
        cudaMemcpy(d_athletes, h_athletes, total_athletes * sizeof(athlete), cudaMemcpyHostToDevice);

        int block_size = 256;
        int num_blocks = (total_athletes + block_size - 1) / block_size;

        bool race_finished = false;
        bool first_athlete_finished = false;
        bool selected_athlete_finished = false;  // Flag for the selected athlete
        float total_distance = swimming_distance + cycling_distance + running_distance; // Toplam yarış mesafesi

        while (!race_finished) {
            // CUDA kernel'ini çağır (GPU üzerinde yarış aşamaları)
            runRaceStages(d_athletes, total_athletes, num_blocks, block_size, swimming_distance, cycling_distance, running_distance);  // Yarış aşamalarını çalıştır
            // GPU verilerini CPU'ya kopyala
            checkAthletesOnCPU(d_athletes, h_athletes, total_athletes);

            int selected_athlete_id = team_index * 3 + athlete_index;
            if (!selected_athlete_finished && h_athletes[selected_athlete_id].getPosition() < total_distance) {
                // Eğer seçilen atlet yarışı bitirmemişse bilgilerini yazdır
                printAthletePosition(h_athletes, team_index, athlete_index);
            } else if (!selected_athlete_finished && h_athletes[selected_athlete_id].getPosition() >= total_distance) {
                // Seçilen atlet yarışı bitirdiyse flag'i true yap
                selected_athlete_finished = true;
                std::cout << "Seçilen atlet yarışı bitirdi! , tüm yarışçıların bitirmesi bekleniyor..." << std::endl;
            }

            // Eğer ilk atlet yarışı bitirdiyse tüm pozisyonları bir kez yazdır ve kupa mesajını ver
            if (!first_athlete_finished) {
                for (int i = 0; i < total_athletes; ++i) {
                    if (h_athletes[i].getPosition() >= (total_distance)) {
                        first_athlete_finished = true;
                        std::cout << "İlk atlet yarışı bitirdi ve Kupa kazandı!\n";
                        std::cout << "Birinci olan atletin bilgileri:\n";
                        first_place_id = i / 3;
                        first_place_team_id = i % 3;
                        printAthletePosition(h_athletes,  first_place_id, first_place_team_id);
                        std::cout << "Tüm atletlerin anlık pozisyonları:\n";
                        printAllPositions(h_athletes, total_athletes);  // Tüm atletlerin pozisyonlarını yazdır
                        break;
                    }
                }
            }

            // Eğer tüm atletler yarışı bitirdiyse döngüyü sonlandır
            race_finished = true;
            for (int i = 0; i < total_athletes; ++i) {
                if (h_athletes[i].getPosition() < total_distance) {
                    race_finished = false;
                    break;
                }
            }

            // Her döngüde 1 saniye bekle
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        // Yarış bittiğinde takım sıralamasını yazdır ve kazanan takıma kupa ver
        printTeamResults(h_athletes, total_athletes);  // CPU'daki atlet verileriyle takımları hesapla

        // Belleği temizle
        cudaFree(d_athletes);
        delete[] h_athletes;
    }
};

int main(int argc, char* argv[]) {  // Argümanları al

    srand(time(NULL));  // Rastgele sayı üretmek için seed
    std::cout << "Lütfen performansını izlemek istediğiniz atletin takım ve atlet id'sini ( 0 - 2 ) ve parkur mesafelerini metre olarak girin " << std::endl;
    std::cout << " <team_index> <athlete_index> <swim_distance> <cycle_distance> <run_distance>" << std::endl;

    if (argc < 6) {
        std::cerr << "Lütfen bu formatta input girin : " << argv[0] << " <team_index> <athlete_index> <swim_distance> <cycle_distance> <run_distance>" << std::endl;
        return 1;
    }

    // Takım ve atlet indekslerini alıyoruz
    int team_index = std::stoi(argv[1]);
    int athlete_index = std::stoi(argv[2]);

    // Kullanıcıdan gelen mesafeleri alıyoruz
    float swim_distance = std::stof(argv[3]);
    float cycle_distance = std::stof(argv[4]);
    float run_distance = std::stof(argv[5]);

    if (team_index < 0 || team_index >= 300 || athlete_index < 0 || athlete_index >= 3) {
        std::cerr << "Geçersiz atlet veya takım." << std::endl;
        return 1;
    }

    // Yarış sınıfını başlatıyoruz ve mesafeleri gönderiyoruz
    race r(swim_distance, cycle_distance, run_distance);  // Dinamik mesafelerle race sınıfı
    r.initialize_teams();  // Takımları başlat

    // Yarışı başlat ve belirtilen atleti takip et
    r.startRace(team_index, athlete_index);

    return 0;
}
