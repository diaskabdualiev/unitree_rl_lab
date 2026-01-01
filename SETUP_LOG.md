# Unitree RL Lab - Журнал установки

**Дата**: 2025-12-24
**Система**: Ubuntu 22.04.5 LTS, 2x RTX 4080, Intel i9-13900K, 128GB RAM

---

## 1. Анализ системы (до установки)

### Системные требования - ОК

| Компонент | Значение | Требования | Статус |
|-----------|----------|------------|--------|
| ОС | Ubuntu 22.04.5 LTS | 22.04+ | OK |
| GLIBC | 2.35 | >= 2.35 | OK |
| GPU | 2x RTX 4080 (16GB) | RTX 3080+ | OK |
| RAM | 128 GB | 64 GB рек. | OK |
| Диск | 850 GB свободно | 100 GB | OK |

### Структура проектов

```
/home/dias/Documents/unitree/
├── unitree_rl_lab/        # RL обучение (Isaac Lab)
├── unitree_sim_isaaclab/  # Манипуляция
├── unitree_sdk2/          # C++ SDK
├── xr_teleoperate/        # VR телеуправление
├── unitree_mujoco/        # MuJoCo симулятор
└── unitree_model/         # USD модели роботов
```

---

## 2. Порядок установки unitree_rl_lab

### Шаг 1: Создание conda окружения

```bash
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab
```

### Шаг 2: Установка Isaac Sim 5.1.0

```bash
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

**Время**: ~10-15 минут (зависит от интернета)

### Шаг 3: Установка Isaac Lab 2.3.0

```bash
cd /home/dias/Documents/unitree
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.3.0
./isaaclab.sh --install
```

**Предупреждение** (можно игнорировать):
```
[WARN] Could not find Isaac Sim VSCode settings...
```

### Шаг 4: Установка unitree_rl_lab

```bash
cd /home/dias/Documents/unitree/unitree_rl_lab
conda activate env_isaaclab
./unitree_rl_lab.sh -i
```

**Ошибки** (не критичны):
```
./unitree_rl_lab.sh: line 36: .../setenv.sh: No such file or directory
activate-global-python-argcomplete: error: Permission denied
```
Это ошибки автодополнения shell, не влияют на работу.

### Шаг 5: Настройка пути к моделям

Отредактировать файл:
```
/home/dias/Documents/unitree/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py
```

Изменить строку 20:
```python
# Было:
UNITREE_MODEL_DIR = "path/to/unitree_model"

# Стало:
UNITREE_MODEL_DIR = "/home/dias/Documents/unitree/unitree_model"
```

---

## 3. Проблемы и решения

### Проблема 1: Ошибка драйвера NVIDIA

**Симптомы**:
```
[Error] The currently installed NVIDIA graphics driver is unsupported
Installed driver: 535.18
The unsupported driver range: [0.0, 535.129)
```

**Причина**: Isaac Sim неправильно определял версию драйвера (показывал 535.18 вместо реальной 535.274)

**Решение**: Обновление драйвера до 580.x
```bash
sudo apt update
sudo apt install nvidia-driver-550  # или выше
sudo reboot
```

**Результат**: Драйвер 580.95.05 - работает корректно

---

### Проблема 2: Нет директории logs для play.py

**Симптомы**:
```
FileNotFoundError: No such file or directory: '.../logs/rsl_rl/unitree_g1_29dof_velocity'
```

**Причина**: Скрипт `play.py` ищет обученную модель в `logs/`, но там пусто. Готовые ONNX модели предназначены для MuJoCo/реального робота, не для Isaac Lab.

**Решение**:
- Вариант A: Сначала обучить модель через `train.py`
- Вариант B: Использовать MuJoCo с готовыми ONNX моделями

---

### Проблема 3: Чёрный экран в симуляторе (RTX рендерер)

**Симптомы**:
```
[Error] rtx driver verification failed
[Warning] HydraEngine rtx failed creating scene renderer
```

**Причина**: Несовместимость драйвера с RTX рендерером Isaac Sim

**Решение**: Обновить драйвер NVIDIA (см. Проблему 1)

**Альтернатива**: Работать в headless режиме
```bash
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity --headless
```

---

### Проблема 4: Предупреждения PCIe

**Симптомы**:
```
[Warning] Device 0 PCIe link current width 16 and device 1 PCIe link current width 4 don't match
[Warning] PCIe link width current (4) and maximum (16) for device 1 don't match
```

**Причина**: Вторая GPU работает на PCIe x4 вместо x16 (аппаратное ограничение слота)

**Влияние**: Снижение скорости GPU-to-GPU передачи (5.8 GB/s вместо ~25 GB/s)

**Решение**: Не критично для обучения. Если важна скорость - переставить GPU в другой слот x16.

---

### Проблема 5: CPU в режиме powersave

**Симптомы**:
```
[Warning] CPU performance profile is set to powersave
```

**Решение** (опционально):
```bash
# Временно
sudo cpupower frequency-set -g performance

# Постоянно
sudo apt install cpufrequtils
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
sudo systemctl restart cpufrequtils
```

---

## 4. Готовые модели (Pre-trained)

Расположение:
```
unitree_rl_lab/deploy/robots/g1_29dof/config/policy/
├── velocity/v0/exported/policy.onnx      # Ходьба
├── mimic/gangnam_style/exported/policy.onnx  # Танец
└── mimic/dance_102/exported/policy.onnx      # Танец
```

**Использование**: Только для MuJoCo или реального робота (не для Isaac Lab play.py)

---

## 5. Основные команды

### Обучение (с визуализацией)
```bash
conda activate env_isaaclab
cd /home/dias/Documents/unitree/unitree_rl_lab
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity --num_envs 64
```

### Обучение (headless, быстрее)
```bash
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity --headless
```

### Список задач
```bash
./unitree_rl_lab.sh -l
```

### Инференс (после обучения)
```bash
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity
```

### Мониторинг через TensorBoard
```bash
tensorboard --logdir /home/dias/Documents/unitree/unitree_rl_lab/logs/rsl_rl/
# Открыть http://localhost:6006
```

---

## 6. Финальная конфигурация системы

```
OS:          Ubuntu 22.04.5 LTS
Kernel:      6.8.0-90-generic
NVIDIA:      580.95.05
GPU 0:       RTX 4080 16GB (PCIe x16)
GPU 1:       RTX 4080 16GB (PCIe x4)
CPU:         Intel i9-13900K (24 cores, 32 threads available)
RAM:         128 GB
Python:      3.11 (conda: env_isaaclab)
Isaac Sim:   5.1.0
Isaac Lab:   2.3.0
```

---

## 7. Что ещё не настроено

| Проект | Статус | Команда для установки |
|--------|--------|----------------------|
| unitree_sim_isaaclab | Не настроен | Создать отдельное окружение |
| unitree_mujoco | Не собран | `cmake .. && make` |
| unitree_sdk2 | Не собран | `cmake .. && make && sudo make install` |
| xr_teleoperate | Не настроен | Создать окружение `tv` |

---

## 8. Полезные ссылки

- [Isaac Lab Docs](https://isaac-sim.github.io/IsaacLab)
- [unitree_rl_lab GitHub](https://github.com/unitreerobotics/unitree_rl_lab)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)
- [Unitree Discord](https://discord.gg/ZwcVwxv5rq)
