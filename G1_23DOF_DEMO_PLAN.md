# G1-23dof EDU: План демонстрации и обучения

Документ для поездки к реальному роботу G1-23dof (EDU версия).

**Дата создания:** 2025-12-29

---

## Содержание

1. [Статус обучения](#статус-обучения)
2. [Что нужно взять](#что-нужно-взять)
3. [Подготовка перед поездкой](#подготовка-перед-поездкой)
4. [План демонстрации](#план-демонстрации)
5. [План обучения людей](#план-обучения-людей)
6. [Управление геймпадом](#управление-геймпадом)
7. [Troubleshooting](#troubleshooting)

---

## Статус обучения

| Задача | Task ID | Статус | Описание |
|--------|---------|--------|----------|
| **Velocity** | `Unitree-G1-23dof-Velocity` | Обучается | Ходьба по командам |
| **Bata Dias** | `Unitree-G1-23dof-Mimic-Bata-Dias` | Обучается | Танец из Rokoko MoCap |

### Проверка прогресса

```bash
# Проверить последние чекпоинты
ls -la logs/rsl_rl/unitree_g1_23dof_velocity/*/model_*.pt | tail -5
ls -la logs/rsl_rl/unitree_g1_23dof_mimic_bata_dias/*/model_*.pt | tail -5

# TensorBoard мониторинг
tensorboard --logdir logs/rsl_rl/
# Открыть http://localhost:6006
```

---

## Что нужно взять

### Оборудование

- [ ] Ноутбук с Linux (Ubuntu 22.04+)
- [ ] Xbox 360 / PS4 / DualShock 4 геймпад
- [ ] Ethernet кабель (CAT5e/CAT6)
- [ ] USB флешка 8GB+ (резервные файлы)
- [ ] Зарядка для ноутбука

### Файлы для копирования

После завершения обучения скопировать:

```
unitree_rl_lab/deploy/robots/g1_23dof/
├── config/
│   ├── config.yaml                           # Конфигурация FSM
│   └── policy/
│       ├── velocity/
│       │   └── exported/policy.onnx          # Модель ходьбы
│       └── mimic/bata_dias/
│           ├── exported/policy.onnx          # Модель танца
│           └── params/bata_dias_23dof_60hz.csv
├── build/
│   └── g1_ctrl                               # Исполняемый файл
├── src/
├── include/
└── CMakeLists.txt
```

### Документация (опционально)

- [ ] CLAUDE.md
- [ ] G1_23DOF_GUIDE.md
- [ ] ROKOKO_PIPELINE.md
- [ ] Этот файл (G1_23DOF_DEMO_PLAN.md)

---

## Подготовка перед поездкой

### 1. Дождаться завершения обучения

Рекомендуемый минимум: **10,000-15,000 iterations** для базового качества.
Оптимально: **25,000-30,000 iterations**.

### 2. Экспорт моделей в ONNX

```bash
cd /home/dias/Documents/unitree/unitree_rl_lab
conda activate env_isaaclab

# Экспорт Velocity
./unitree_rl_lab.sh -e --task Unitree-G1-23dof-Velocity

# Экспорт Bata Dias
./unitree_rl_lab.sh -e --task Unitree-G1-23dof-Mimic-Bata-Dias
```

### 3. Копирование ONNX в deploy

```bash
# Найти последний запуск и скопировать
VELOCITY_RUN=$(ls -td logs/rsl_rl/unitree_g1_23dof_velocity/*/ | head -1)
MIMIC_RUN=$(ls -td logs/rsl_rl/unitree_g1_23dof_mimic_bata_dias/*/ | head -1)

cp "${VELOCITY_RUN}exported/policy.onnx" \
   deploy/robots/g1_23dof/config/policy/velocity/exported/

cp "${MIMIC_RUN}exported/policy.onnx" \
   deploy/robots/g1_23dof/config/policy/mimic/bata_dias/exported/
```

### 4. Тест в MuJoCo (локально)

```bash
# Терминал 1: MuJoCo симулятор
cd /home/dias/Documents/unitree/unitree_mujoco/simulate/build
./unitree_mujoco

# Терминал 2: Контроллер
cd /home/dias/Documents/unitree/unitree_rl_lab/deploy/robots/g1_23dof/build
./g1_ctrl --network lo
```

### 5. Упаковка файлов

```bash
# Создать архив для переноса
cd /home/dias/Documents/unitree/unitree_rl_lab
tar -czvf g1_23dof_deploy.tar.gz deploy/robots/g1_23dof/

# Скопировать на флешку
cp g1_23dof_deploy.tar.gz /media/usb/
```

---

## План демонстрации

### Этап 1: Демонстрация в симуляции (30 мин)

**Цель:** Показать как система работает без риска повреждения робота.

```bash
# 1. Запустить MuJoCo
cd unitree_mujoco/simulate/build
./unitree_mujoco

# 2. Запустить контроллер
cd unitree_rl_lab/deploy/robots/g1_23dof/build
./g1_ctrl --network lo

# 3. Управление геймпадом:
#    LT + ↑  = Встать
#    RB + X  = Ходьба
#    LT + A  = Танец Bata Dias
#    LT + B  = Стоп
```

**Что показать:**
- Переключение между режимами (FSM)
- Ходьба в разных направлениях (левый стик)
- Вращение (правый стик)
- Танец Bata Dias

### Этап 2: Подключение к реальному роботу (20 мин)

**Подключение по Ethernet:**

```bash
# Типичный IP робота G1
ROBOT_IP="192.168.123.161"

# Проверка связи
ping $ROBOT_IP

# SSH на робот
ssh unitree@$ROBOT_IP
# Пароль обычно: 123
```

**Копирование файлов:**

```bash
# С ноутбука на робота
scp -r deploy/robots/g1_23dof unitree@$ROBOT_IP:~/
```

**Компиляция на роботе (если нужно):**

```bash
# На роботе
cd ~/g1_23dof
mkdir -p build && cd build
cmake .. && make -j4
```

### Этап 3: Запуск на реальном роботе (40 мин)

**ВАЖНО: Меры безопасности**

1. Убедиться что робот на ровной поверхности
2. Иметь возможность быстро нажать E-STOP
3. Первый запуск делать с подстраховкой (2 человека рядом)
4. Начинать с режима FixStand, не сразу с Velocity

**Запуск:**

```bash
# На роботе
cd ~/g1_23dof/build
./g1_ctrl --network eth0

# Если Wi-Fi:
./g1_ctrl --network wlan0
```

**Последовательность:**

1. `LT + ↑` — Робот встаёт (FixStand)
2. Подождать 3-5 секунд пока стабилизируется
3. `RB + X` — Переход в режим ходьбы (Velocity)
4. Аккуратно двигать левый стик — робот идёт
5. `LT + B` — Вернуться в Passive (остановка)

**Демонстрация танца:**

1. Из FixStand: `LT + A` — Танец Bata Dias
2. Робот выполняет заученное движение
3. `LT + B` — Остановка

---

## План обучения людей

### Модуль 1: Обзор архитектуры (45 мин)

**Слайд 1: Общая схема**

```
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE ОБУЧЕНИЯ                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. ДАННЫЕ              2. ОБУЧЕНИЕ           3. DEPLOY     │
│  ─────────              ──────────            ────────      │
│  Rokoko MoCap    →      Isaac Lab      →      ONNX Runtime  │
│  или Velocity           (GPU, Python)         (CPU, C++)    │
│  rewards                                                    │
│                                                             │
│  CSV файлы              .pt чекпоинты         policy.onnx   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Слайд 2: Два типа обучения**

| Тип | Velocity | Mimic |
|-----|----------|-------|
| **Цель** | Ходить по командам | Повторять движения |
| **Данные** | Не нужны | CSV с позами суставов |
| **Источник** | Reward функции | Motion Capture |
| **Применение** | Локомоция | Танцы, жесты |

**Слайд 3: G1-29dof vs G1-23dof**

```
G1-29dof (полная версия):
  Ноги: 12 суставов
  Талия: 3 сустава (yaw, roll, pitch)
  Руки: 14 суставов (включая запястья)

G1-23dof (EDU версия):
  Ноги: 12 суставов (без изменений)
  Талия: 1 сустав (только yaw)
  Руки: 10 суставов (без wrist_pitch/yaw)

Удалённые суставы:
  - waist_roll_joint
  - waist_pitch_joint
  - left_wrist_pitch_joint
  - left_wrist_yaw_joint
  - right_wrist_pitch_joint
  - right_wrist_yaw_joint
```

### Модуль 2: Практика — запуск обучения (1 час)

**Шаг 1: Настройка окружения**

```bash
# Активация conda
conda activate env_isaaclab

# Проверка GPU
nvidia-smi
```

**Шаг 2: Запуск обучения Velocity**

```bash
cd /home/dias/Documents/unitree/unitree_rl_lab

# С визуализацией (для демонстрации, медленнее)
./unitree_rl_lab.sh -t --task Unitree-G1-23dof-Velocity --num_envs 64

# Headless (для реального обучения, быстрее)
./unitree_rl_lab.sh -t --task Unitree-G1-23dof-Velocity --headless --num_envs 4096
```

**Шаг 3: Мониторинг**

```bash
# TensorBoard
tensorboard --logdir logs/rsl_rl/
# Открыть http://localhost:6006

# Проверка чекпоинтов
watch -n 60 'ls -la logs/rsl_rl/unitree_g1_23dof_velocity/*/model_*.pt | tail -5'
```

**Шаг 4: Просмотр результата**

```bash
# Визуализация обученной модели
./unitree_rl_lab.sh -p --task Unitree-G1-23dof-Velocity
```

### Модуль 3: Практика — deploy на робота (1 час)

**Шаг 1: Экспорт в ONNX**

```bash
./unitree_rl_lab.sh -e --task Unitree-G1-23dof-Velocity
```

**Шаг 2: Структура deploy**

```
deploy/robots/g1_23dof/
├── config/
│   ├── config.yaml          ← FSM конфигурация
│   └── policy/
│       └── velocity/
│           └── exported/
│               └── policy.onnx  ← Нейросеть
├── src/
│   └── State_RLBase.cpp     ← Логика RL состояния
├── include/
│   └── Types.h
├── main.cpp                 ← Точка входа
└── CMakeLists.txt
```

**Шаг 3: Компиляция**

```bash
cd deploy/robots/g1_23dof
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

**Шаг 4: Запуск**

```bash
# В симуляции
./g1_ctrl --network lo

# На реальном роботе
./g1_ctrl --network eth0
```

### Модуль 4: Создание своих движений (опционально, 2 часа)

**Pipeline создания Mimic задачи:**

```
Rokoko/Video → SMPL → CSV → NPZ → Task → Train → ONNX
```

**Шаг 1: Подготовка CSV**

CSV файл должен иметь формат:
- 30 колонок для 23dof: 7 (поза) + 23 (суставы)
- Частота: 60 FPS рекомендуется

```
# Формат каждой строки:
x, y, z, qw, qx, qy, qz, joint1, joint2, ..., joint23
```

**Шаг 2: Конвертация в NPZ**

```bash
python scripts/mimic/csv_to_npz.py \
    -f path/to/motion.csv \
    --input_fps 60 \
    --robot g1_23dof \
    --headless
```

**Шаг 3: Создание Task**

Скопировать существующий task:

```bash
cp -r source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_23dof/bata_dias \
      source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_23dof/my_motion
```

Отредактировать `__init__.py`:

```python
import gymnasium as gym

gym.register(
    id="Unitree-G1-23dof-Mimic-My-Motion",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotEnvCfg",
        ...
    },
)
```

**Шаг 4: Обучение**

```bash
./unitree_rl_lab.sh -t --task Unitree-G1-23dof-Mimic-My-Motion --headless
```

---

## Управление геймпадом

### Xbox 360 / DualShock 4 Layout

```
              [LB]                              [RB]
              [LT]                              [RT]

           ┌──────┐                          ┌──────┐
           │  ↑   │     [BACK]  [START]      │  Y   │
           │←   →│                          │X   B│
           │  ↓   │                          │  A   │
           └──────┘                          └──────┘
            D-PAD                            Buttons

         (L-STICK)                          (R-STICK)
           Move                              Rotate
```

### Команды FSM

| Комбинация | Действие | Из состояния |
|------------|----------|--------------|
| `LT + ↑` | FixStand (встать) | Passive |
| `RB + X` | Velocity (ходьба) | FixStand |
| `LT + A` | Bata Dias (танец) | FixStand |
| `LT + B` | Passive (стоп) | Любое |

### Управление движением (в режиме Velocity)

| Стик | Действие |
|------|----------|
| L-Stick ↑↓ | Вперёд/Назад |
| L-Stick ←→ | Влево/Вправо (strafe) |
| R-Stick ←→ | Поворот |

### Elastic Band (в MuJoCo симуляции)

| Клавиша | Действие |
|---------|----------|
| `9` | Вкл/Выкл виртуальную пружину |
| `7` или `↑` | Поднять робота |
| `8` или `↓` | Опустить робота |

---

## Troubleshooting

### Проблема: Робот не отвечает на команды

```bash
# Проверить DDS связь
# Domain ID должен быть 0 для реального робота, 1 для симуляции

# Проверить сетевой интерфейс
ip addr show

# Перезапустить контроллер с правильным интерфейсом
./g1_ctrl --network eth0  # или enp0s31f6, wlan0 и т.д.
```

### Проблема: Робот падает при старте

1. Убедиться что робот на ровной поверхности
2. Начинать с FixStand, не сразу с Velocity
3. Проверить что используется правильная модель (23dof, не 29dof)

### Проблема: ONNX не найден

```bash
# Проверить путь в config.yaml
cat deploy/robots/g1_23dof/config/config.yaml | grep policy_dir

# Убедиться что файл существует
ls -la deploy/robots/g1_23dof/config/policy/velocity/exported/policy.onnx
```

### Проблема: Ошибка компиляции на роботе

```bash
# Установить зависимости
sudo apt install cmake g++ libyaml-cpp-dev

# Проверить что unitree_sdk2 установлен
pkg-config --libs unitree_sdk2
```

### Проблема: Геймпад не работает

```bash
# Проверить подключение
ls /dev/input/js*

# Тест геймпада
jstest /dev/input/js0

# Изменить устройство в config.yaml MuJoCo
# joystick_device: "/dev/input/js0"
```

---

## Полезные ссылки

| Документ | Описание |
|----------|----------|
| `CLAUDE.md` | Главная документация проекта |
| `G1_23DOF_GUIDE.md` | Детальный гайд по 23dof |
| `ROKOKO_PIPELINE.md` | Создание движений из Rokoko |
| `MOTION_CAPTURE_PIPELINE.md` | MoCap pipeline |

---

## Контакты и поддержка

При возникновении вопросов обращаться к документации в репозитории или использовать Claude Code для анализа и решения проблем.

---

*Документ сгенерирован автоматически. Последнее обновление: 2025-12-29*
