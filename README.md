# 생활쓰레기 4분류 모델

생활쓰레기 사진을 **비닐(vinyl), 유리(glass), 플라스틱(plastic), 캔(can)** 네 가지로 분류하는 모델을 학습하고 평가하는 코드입니다.

- 사전학습 가중치를 쓰지 않고 ResNet을 처음부터 학습합니다.
- 학습 데이터는 AI Hub의 공개 데이터셋 두 개에서 만듭니다.
- 최종 평가는 직접 모은 사진(`test/` 폴더)으로 합니다.

사진, 학습된 가중치, AI Hub 원본은 이 저장소에 들어 있지 않습니다. AI Hub 데이터는 이용약관상 재배포할 수 없어서, 아래 절차대로 각자 받아야 합니다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `prepare_data.py` | AI Hub 원본을 내려받고, 사진에서 물건을 잘라 4개 클래스 폴더로 정리합니다. |
| `train.py` | 모델을 학습합니다. |
| `eval.py` | 학습된 가중치를 `test/` 폴더의 사진으로 평가합니다. |

## 준비

- Python 3.11 이상, PyTorch 2.x, torchvision, Pillow
- CUDA를 지원하는 GPU (학습과 평가 코드는 GPU에서만 돌아갑니다)
- 데이터 내려받기에는 `curl`과 `tar`가 필요합니다. Windows 10 이상과 Linux에는 기본으로 들어 있습니다.

```
pip install torch torchvision pillow
```

## 1. 데이터 준비

### AI Hub 신청

[aihub.or.kr](https://aihub.or.kr)에 가입한 뒤 아래 두 데이터셋의 다운로드를 신청해 승인을 받고, "AI 허브 오픈 API" 메뉴에서 API 키를 발급받습니다.

| 이 저장소에서 부르는 이름 | AI Hub 데이터셋 | 쓰는 범위 | 촬영 환경 |
|---|---|---|---|
| 140번 | 생활 폐기물 이미지 | 비닐, 유리병, 캔류, 페트병, 플라스틱류 | 실외 바닥 |
| 71385번 | 생활폐기물 데이터 활용·환류 | "실내형 분류기" 부분 | 실내 선별기 (어두움) |

API 키는 환경변수 `AIHUB_APIKEY`로 주거나, `prepare_data.py`와 같은 폴더에 `.aihub_key` 파일로 저장합니다. 키 파일은 `.gitignore`에 들어 있어 커밋되지 않습니다.

### 내려받기와 가공

```
python prepare_data.py fetch     # 원본 zip을 _aihub_raw/ 에 내려받습니다 (전체 약 135GB)
python prepare_data.py build     # 물건을 잘라 256px로 줄이고 manifest.csv를 만듭니다
```

일부만 받으려면 `--parts`로 고릅니다. 용량은 `140-val` 약 13GB, `140-train` 약 104GB, `71385` 약 19GB입니다.

```
python prepare_data.py fetch --parts 140-val 71385
```

AI Hub 서버는 받는 도중에 연결을 끊는 일이 있고 이어받기를 지원하지 않습니다. 그래서 `fetch`는 파일마다 처음부터 통째로 받고, 받은 zip이 정상인지 검사한 뒤 완료로 표시합니다. 중간에 멈춰도 다시 실행하면 완료된 파일은 건너뜁니다.

`build`는 라벨에 적힌 박스대로 물건을 잘라(여백 5%) 짧은 변을 256px로 줄여 저장합니다. 결과는 다음과 같습니다.

```
vinyl/  glass/  plastic/  can/     물건 하나가 찍힌 사진
manifest.csv                       사진마다 경로, 클래스, 출처, group, 세부 품목
```

클래스는 이렇게 묶습니다.

| 클래스 | 140번 라벨 | 71385번 라벨 코드 |
|---|---|---|
| vinyl | 비닐류 | `c_7*` |
| glass | 유리병류 | `c_4*` |
| plastic | 페트병류, 플라스틱류 | `c_5*`, `c_6*` |
| can | 캔류 | `c_3*` |

`build`가 끝나면 `_aihub_raw/` 폴더는 지워도 됩니다. 나중에 데이터셋을 더 받아 `build`를 다시 돌리면 기존 `manifest.csv`에 새 사진이 추가됩니다.

### test 사진

평가용 사진은 클래스 이름의 폴더에 넣습니다. 이 사진들은 학습에 쓰지 않습니다.

```
test/vinyl/   test/glass/   test/plastic/   test/can/
```

## 2. 학습

```
python train.py --epochs 100 --name r18_base
```

결과는 `runs/<이름>/`에 저장됩니다.

| 파일 | 내용 |
|---|---|
| `best.pt` | 검증(val) 정확도가 가장 높았던 epoch의 가중치 |
| `last.pt` | 마지막 epoch의 상태. 이어서 학습할 때 씁니다. |
| `metrics.csv` | epoch별 loss와 정확도 |
| `config.json` | 실행에 쓴 설정 |

기본 설정은 ResNet-18, 입력 224px, Cross Entropy(클래스 가중치, label smoothing 0.1), AdamW(lr 1e-3, weight decay 0.05), warmup 5 epoch 뒤 cosine 스케줄입니다.

### 주요 옵션

| 옵션 | 설명 |
|---|---|
| `--arch` | `resnet18`(기본), `resnet34`, `resnet50` |
| `--aug` | 학습 전처리와 augmentation. 아래 표를 참고하세요. |
| `--split` | `object`(기본): 같은 물건의 사진을 한쪽에만 둡니다. `capture`: 71385번은 촬영 묶음(촬영 날짜 단위)을 통째로 val로 뺍니다. |
| `--train-source` | `all`(기본), `140`, `71385`. 한쪽 데이터셋만으로 학습합니다. val은 바뀌지 않습니다. |
| `--source-balance` | 두 데이터셋에서 반반씩 뽑아 학습합니다. |
| `--skip-140tr` | 140번의 Training 분할을 학습에서 뺍니다. |
| `--dropout` | 마지막 분류층 앞의 dropout 비율 |
| `--test-dir test` | epoch마다 test 정확도를 `metrics.csv`에 기록합니다. 가중치 선택에는 쓰지 않습니다. |
| `--subset N` | 학습 사진 N장만 써서 빠르게 돌려 봅니다. |
| `--resume runs/<이름>/last.pt` | 이어서 학습합니다. 설정은 그 실행의 `config.json`에서 가져옵니다. |

`--aug` 모드는 다음과 같습니다.

| 모드 | 내용 |
|---|---|
| `none` | 비율을 유지하고 여백을 채워 224px로 줄이기만 합니다. |
| `light` | 무작위 crop, 좌우 반전 |
| `base` (기본) | `light`에 TrivialAugmentWide와 RandomErasing을 더합니다. |
| `strong` | `base`에 무작위 배경 캔버스, 색 변화, 흑백 변환, 상하 반전, blur를 더합니다. |
| `squash` | 여백 없이 224×224로 눌러 넣기만 합니다. |
| `squash_base` | 눌러 넣은 뒤 `base`의 augmentation을 적용합니다. |

### train/val 분할

`manifest.csv`의 `group`이 같은 사진은 반드시 같은 쪽에 들어갑니다. 140번은 물건 하나를 4~5장씩 찍었기 때문에, 파일 단위로 나누면 거의 같은 사진이 train과 val 양쪽에 들어가 val 점수가 부풀기 때문입니다. 분할은 group 이름의 해시로 정하므로 매번 같은 결과가 나옵니다. 140번 Training 분할(`140tr`)은 항상 train에만 들어갑니다.

## 3. 평가

```
python eval.py runs/r18_base/best.pt
```

전체 정확도, 클래스별 정확도, 혼동행렬을 출력하고, 사진별 예측을 `runs/<이름>/test_predictions.csv`에 저장합니다.

test 사진은 물건이 작게 찍힌 경우가 많아서, 기본 전처리는 사진의 **가운데 70%만 남겨 320px로** 넣습니다. 사진 전체를 그대로 224px로 넣으려면 이렇게 합니다.

```
python eval.py runs/r18_base/best.pt --center 1 --size 224
```

## 지금까지의 결과

ResNet-18, `base` augmentation, 100 epoch 기준입니다. test는 직접 모은 사진 466장입니다.

| 측정 | 정확도 |
|---|---|
| AI Hub val (`object` 분할) | 0.98 |
| AI Hub val (`capture` 분할) | 0.96 |
| test, 가운데 70% + 320px | 0.63 ~ 0.65 |
| test, 사진 전체를 224px로 | 약 0.49 |

- AI Hub 안에서는 정확도가 높지만 직접 모은 사진에서는 크게 떨어집니다. 학습 사진과 test 사진의 촬영 환경과 물건 종류가 다르기 때문으로 보고 있습니다.
- test 성능은 대부분 140번(실외 촬영) 사진에서 나옵니다. 140번만으로 학습해도 test 0.63이 나오고, 71385번만으로 학습하면 0.49로 떨어집니다.
- augmentation을 빼면 0.50, 더 강하게 넣으면 0.57로 둘 다 `base`보다 낮았습니다.
- ResNet-34와 50, 300 epoch 학습, dropout 0.5는 test를 의미 있게 올리지 못했습니다(차이 0.03 이내).
- 가장 약한 클래스는 glass이며, 주로 plastic으로 틀립니다.
- 140번 Training 분할을 더해 학습한 결과는 아직 없습니다.

## 주의

- "가운데 70% + 320px" 전처리는 test 사진의 점수를 보고 고른 설정입니다. 처음 보는 사진에서는 위 숫자보다 낮게 나올 수 있습니다.
- 대부분의 실험은 한 번씩만 돌렸습니다. 같은 설정을 다시 돌리면 test 정확도가 0.02 정도 달라집니다.
- can 클래스에는 알루미늄 캔 외에 통조림 같은 스틸캔도 들어 있습니다.
- AI Hub 데이터와 그것으로 만든 사진은 외부에 공개하거나 재배포하지 않도록 주의해야 합니다.
