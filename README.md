# triathlon

Açıklama:

Bu proje, CUDA kullanarak GPU üzerinde paralel bir triatlon yarış simülasyonu gerçekleştirmeyi amaçlamaktadır. Yarış, üç farklı aşamadan oluşmaktadır: yüzme, bisiklet ve koşu. Her aşama, her atletin hızına göre güncellenir ve yarışın sonunda tüm atletlerin sonuçları sıralanır. Kullanıcı, yarış parkurlarının mesafelerini dinamik olarak belirleyebilir ve belirli bir atletin performansını izleyebilir. İlk sporcu bitiş çizgisine ulaştığında, tüm sporcuların o an bulundukları konumlar ; tüm yarış bittiğinde birinci olan atlet , birinci olan takım ve takım sıralamaları yazdırılır.

C++ ve CUDA kullanılmıştır.

Derlemek için:

```
nvcc -o triathlon athlete_kernel.cu race.cpp
```

Çalıştırmak için:
```
./triathlon <team_index> <athlete_index> <swim_distance> <cycle_distance> <run_distance>
```
team_index: İzlemek istediğiniz takımın indexi (0-299). <br />
athlete_index: İzlemek istediğiniz atletin indexi (0-2).<br />
swim_distance: Yüzme parkurunun uzunluğu (metre cinsinden).<br />
cycle_distance: Bisiklet parkurunun uzunluğu (metre cinsinden).<br />
run_distance: Koşu parkurunun uzunluğu (metre cinsinden).<br />

