# 4D-Humans: Установка и использование

Руководство по извлечению 3D поз человека из видео с помощью 4D-Humans (HMR2.0).

## Что это

**4D-Humans** — метод реконструкции и трекинга 3D людей из видео (ICCV 2023, UC Berkeley).

```
Видео → 4D-Humans → SMPL параметры (PKL) → GMR → CSV для робота → Обучение
```

**Выход:**
- `PHALP_*.mp4` — видео с визуализацией 3D мешей
- `demo_*.pkl` — SMPL параметры для ретаргетинга на робота

## Быстрый старт (если уже установлено)

```bash
conda activate 4D-humans
cd ~/Documents/unitree/motion_capture/4D-Humans

# Обработка видео
python track.py video.source="/путь/к/видео.mp4"

# Результаты в outputs/
ls outputs/
# PHALP_*.mp4      — видео с визуализацией
# results/*.pkl   — данные SMPL
```

## Установка с нуля

### 1. Создание окружения

```bash
conda create -n 4D-humans python=3.10 -y
conda activate 4D-humans

# PyTorch с CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Понизить numpy (chumpy не работает с numpy 2.x)
pip install numpy==1.26.4
```

### 2. Установка зависимостей

```bash
cd ~/Documents/unitree/motion_capture/4D-Humans

# chumpy (проблемный пакет — устанавливать ПЕРВЫМ)
pip install git+https://github.com/mattloper/chumpy.git --no-build-isolation

# Основной пакет
pip install -e . --no-deps

# Остальные зависимости
pip install gdown pytorch-lightning "smplx==0.1.28" pyrender opencv-python yacs scikit-image einops timm webdataset dill pandas

# ninja для сборки
pip install ninja

# Detectron2 (собрать из исходников для PyTorch 2.7+)
pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
```

### 3. PHALP для видео трекинга

```bash
# PHALP без зависимостей (detectron2 уже есть)
pip install git+https://github.com/brjathu/PHALP.git --no-deps

# Доустановить зависимости PHALP
pip install joblib scikit-learn rich scenedetect av

# pytube для YouTube
pip install git+https://github.com/pytube/pytube.git
```

### 4. SMPL модель

```bash
mkdir -p ~/Documents/unitree/motion_capture/4D-Humans/data

# Скачать с https://smpl.is.tue.mpg.de/ (бесплатная регистрация)
# Файл: basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
# Положить в data/

# Также скопировать в корень (нужно для PHALP)
cp data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl .
```

### 5. Патчи для PyTorch 2.6+

В файле `hmr2/models/__init__.py` (строка 84):
```python
# Было:
model = HMR2.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)

# Стало:
import torch
torch.serialization.add_safe_globals([__builtins__['dict']])
model = HMR2.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg, weights_only=False)
```

В файле `track.py` добавить в начало после импортов:
```python
# Fix for PyTorch 2.6+ weights_only default change
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
```

В файле `track.py` (строка ~150) для обхода neural_renderer:
```python
# Было:
def setup_hmr(self):
    self.HMAR = HMR2023TextureSampler(self.cfg)

# Стало:
def setup_hmr(self):
    # Use HMR2Predictor instead of HMR2023TextureSampler to avoid neural_renderer dependency
    self.HMAR = HMR2Predictor(self.cfg)
```

### 6. Проверка установки

```bash
conda activate 4D-humans
python -c 'import torch; import detectron2; import hmr2; import phalp; print("OK")'
```

## Использование

### Обработка изображений

```bash
python demo.py \
    --img_folder example_data/images \
    --out_folder demo_out \
    --batch_size 1 \
    --side_view \
    --save_mesh
```

### Обработка видео

```bash
# Локальный файл
python track.py video.source="example_data/videos/gymnasts.mp4"

# Своё видео
python track.py video.source="/home/dias/Videos/dance.mp4"

# YouTube (если pytube работает)
python track.py video.source='"https://www.youtube.com/watch?v=VIDEO_ID"'
```

### Результаты

```
outputs/
├── PHALP_*.mp4           # Видео с визуализацией
├── results/
│   └── demo_*.pkl        # SMPL данные
├── results_tracks/       # Треки
└── _DEMO/                # Временные файлы
```

## Структура PKL файла

```python
import joblib
data = joblib.load('outputs/results/demo_gymnasts.pkl')

# Структура:
{
    'frame_path_001.jpg': {
        'tracked_ids': [1, 2, 3, ...],      # ID людей
        'smpl': [                            # Для каждого человека:
            {
                'global_orient': (1, 3, 3),  # Ориентация тела
                'body_pose': (23, 3, 3),     # Позы 23 суставов
                'betas': (10,),              # Форма тела
            },
            ...
        ],
        '3d_joints': [(45, 3), ...],         # 3D координаты суставов
        '2d_joints': [...],                  # 2D проекции
        'camera': [...],                     # Параметры камеры
        'bbox': [...],                       # Bounding boxes
    },
    'frame_path_002.jpg': {...},
    ...
}
```

## Конвертация в робота G1

### Через GMR (рекомендуется)

```bash
# Установка GMR
cd ~/Documents/unitree/motion_capture
git clone https://github.com/YanjieZe/GMR
cd GMR
conda create -n gmr python=3.10 -y
conda activate gmr
pip install -e .

# Конвертация PKL → CSV
python scripts/smpl_to_robot.py \
    --input ~/Documents/unitree/motion_capture/4D-Humans/outputs/results/demo_gymnasts.pkl \
    --robot unitree_g1 \
    --output g1_motion.csv
```

### Обучение робота

```bash
cd ~/Documents/unitree/unitree_rl_lab
conda activate env_isaaclab

# Конвертация CSV → NPZ
python scripts/mimic/csv_to_npz.py -f g1_motion.csv --input_fps 30

# Обучение
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Mimic-Custom --headless
```

## Первый запуск — что скачивается

При первом запуске автоматически скачиваются:

| Файл | Размер | Назначение |
|------|--------|------------|
| `hmr2_data.tar.gz` | ~2.5 GB | Веса HMR2 |
| `model_final_f05665.pkl` | ~2.77 GB | ViTDet детектор |
| `model_final_2d9806.pkl` | ~431 MB | Mask RCNN |
| `hmar_v2_weights.pth` | ~166 MB | HMAR модель |
| `pose_predictor.pth` | ~140 MB | Предсказание поз |

Всё кэшируется в `~/.cache/4DHumans/` и `~/.cache/phalp/`.

## Troubleshooting

### chumpy не устанавливается
```bash
pip install numpy==1.26.4
pip install git+https://github.com/mattloper/chumpy.git --no-build-isolation
```

### detectron2 не устанавливается
```bash
pip install ninja
pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
```

### WeightsUnpickler error (PyTorch 2.6+)
Добавить патч `weights_only=False` (см. раздел "Патчи для PyTorch 2.6+")

### neural_renderer не устанавливается
Если нет CUDA Toolkit (nvcc), используй HMR2Predictor вместо HMR2023TextureSampler (см. патч в track.py)

### SMPL модель не найдена
```bash
cp data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl .
cp data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl ~/.cache/phalp/3D/
```

## Системные требования

- Ubuntu 22.04
- NVIDIA GPU (RTX 3080+ рекомендуется)
- ~10 GB свободного места для моделей
- Python 3.10
- PyTorch 2.x + CUDA 11.8+

## Ссылки

- [4D-Humans GitHub](https://github.com/shubham-goel/4D-Humans)
- [PHALP GitHub](https://github.com/brjathu/PHALP)
- [SMPL модель](https://smpl.is.tue.mpg.de/)
- [GMR для ретаргетинга](https://github.com/YanjieZe/GMR)
- [Статья arXiv](https://arxiv.org/abs/2305.20091)
