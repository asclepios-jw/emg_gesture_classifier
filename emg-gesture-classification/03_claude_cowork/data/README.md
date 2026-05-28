# Dataset

이 폴더에 데이터셋 CSV가 위치해야 합니다.

## 다운로드

[UCI Machine Learning Repository - EMG Data for Gestures](https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures)

압축을 풀면 피험자별로 raw 데이터가 분리돼 있는데, 모든 데이터를 합친 단일 CSV (`EMG-data.csv`) 형식으로 본 프로젝트는 입력을 받습니다. CSV 컬럼:

```
time, channel1, channel2, ..., channel8, class, label
```

- `class`: 0~7 동작 라벨
- `label`: 1~36 피험자 ID

## 인용

> Lobov, S., Krilova, N., Kastalskiy, I., Kazantsev, V., & Makarov, V. A. (2018).
> Latent factors limiting the performance of sEMG-interfaces.
> Sensors, 18(4), 1122.
