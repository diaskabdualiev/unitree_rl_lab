# Roadmap: От примеров к своим проектам

## Что уже сделано

- [x] Установка unitree_mujoco + gamepad контроль
- [x] Установка unitree_rl_lab + Isaac Sim
- [x] Запуск готовых policy (velocity, gangnam_style, dance_102)
- [x] Тренировка танцев на своих GPU
- [x] Деплой в MuJoCo
- [x] Понимание архитектуры (CSV → train → ONNX → deploy)

---

## Уровень 1: Свой первый танец (1-2 дня)

**Цель**: Создать простое движение и обучить робота

### Шаг 1.1: Создать простой танец вручную

```bash
cd /home/dias/Documents/unitree/unitree_rl_lab
mkdir -p source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/my_first_dance
```

Создать Python скрипт для генерации CSV:
- Простые движения руками (махи)
- Покачивание торсом
- Приседания

**Формат CSV** (36 колонок):
```
base_x, base_y, base_z,           # позиция (3)
quat_x, quat_y, quat_z, quat_w,   # ориентация (4)
joint_0, joint_1, ... joint_28    # 29 суставов
```

### Шаг 1.2: Конвертация и тренировка

```bash
# Конвертировать CSV → NPZ
python scripts/mimic/csv_to_npz.py -f <path_to_csv> --input_fps 60

# Создать task config (скопировать и изменить dance_102)
# Тренировать
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Mimic-MyFirstDance --headless
```

### Шаг 1.3: Тест в MuJoCo

```bash
# Скопировать ONNX в deploy
cp logs/rsl_rl/.../exported/policy.onnx deploy/robots/g1_29dof/config/policy/mimic/my_first_dance/exported/

# Добавить в FSM config и протестировать
```

### Результат уровня 1:
- [ ] Понимание формата motion данных
- [ ] Умение создавать task config
- [ ] Первый свой танец работает в MuJoCo

---

## Уровень 2: Motion Capture танец (3-5 дней)

**Цель**: Использовать реальные движения человека

### Вариант A: Видео → Pose Estimation → CSV

```
Видео танца → MediaPipe/OpenPose → 3D позы → Ретаргетинг на G1 → CSV
```

**Инструменты**:
- MediaPipe (Google) - бесплатно
- OpenPose - бесплатно
- MoveNet - бесплатно

### Вариант B: Готовые BVH файлы

```
Mixamo.com → BVH файл → Ретаргетинг → CSV
```

**Ресурсы**:
- https://www.mixamo.com (бесплатные анимации)
- CMU Motion Capture Database

### Вариант C: Запись через VR (xr_teleoperate)

```bash
conda activate tv
python teleop/teleop_hand_and_arm.py --arm=g1_29 --ee=dex3 --sim --record
# Записать движения своим телом через VR
```

### Результат уровня 2:
- [ ] Пайплайн Video → CSV
- [ ] Или пайплайн BVH → CSV
- [ ] Несколько разных танцев обучены

---

## Уровень 3: Кастомная локомоция (1-2 недели)

**Цель**: Изменить как робот ходит

### Шаг 3.1: Изучить reward functions

```
source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/velocity/
├── config/g1_29dof/agent/rsl_rl_cfg.py    # Гиперпараметры
├── config/g1_29dof/env_cfg.py              # Награды, termination
└── velocity_env.py                          # Основная логика
```

### Шаг 3.2: Эксперименты с наградами

| Награда | Эффект |
|---------|--------|
| `lin_vel_z` penalty | Меньше вертикальных колебаний |
| `ang_vel_xy` penalty | Стабильнее торс |
| `action_rate` penalty | Плавнее движения |
| `feet_air_time` reward | Выше поднимает ноги |

### Шаг 3.3: Создать свой стиль ходьбы

Примеры:
- Крадущаяся походка (низкий центр тяжести)
- Маршевый шаг (высокое поднятие колен)
- Бег

### Результат уровня 3:
- [ ] Понимание reward shaping
- [ ] Своя кастомная походка
- [ ] Умение дебажить тренировку (TensorBoard)

---

## Уровень 4: Манипуляция (2-3 недели)

**Цель**: Робот берёт и перемещает объекты

### Шаг 4.1: Освоить unitree_sim_isaaclab

```bash
conda activate unitree_sim_env
cd /home/dias/Documents/unitree/unitree_sim_isaaclab

# Запустить готовый пример
python sim_main.py --device gpu --enable_cameras \
  --task Isaac-PickPlace-Cylinder-G129-Dex3-Joint \
  --enable_dex3_dds --robot_type g129
```

### Шаг 4.2: Телеоперация через xr_teleoperate

```bash
# Терминал 1: Симуляция
python sim_main.py ... --enable_dex3_dds

# Терминал 2: Телеоперация
conda activate tv
python teleop/teleop_hand_and_arm.py --arm=g1_29 --ee=dex3 --sim --record
```

### Шаг 4.3: Сбор датасета

- Записать 50-100 демонстраций pick-place
- Аугментировать (свет, камера)
- Обучить imitation learning policy

### Результат уровня 4:
- [ ] Телеоперация работает
- [ ] Собран датасет манипуляций
- [ ] Policy для pick-place

---

## Уровень 5: Реальный робот (когда будет доступ)

**Цель**: Перенести всё на физического G1

### Подготовка

1. **Страховочная система** - обязательно!
2. **Проверенные policy** - много тестов в MuJoCo
3. **Безопасное пространство** - мягкий пол, нет препятствий

### Порядок деплоя

```
1. Debug mode (только руки) → проверить SDK работает
2. Low-level + страховка → тест FixStand
3. Low-level + страховка → тест Velocity (ходьба на месте)
4. Low-level + страховка → тест танца
5. Без страховки (осторожно!)
```

### Результат уровня 5:
- [ ] Робот выполняет твой танец
- [ ] Робот ходит с твоей policy
- [ ] Полный цикл sim2real пройден

---

## Уровень 6: Продвинутые проекты

### 6.1: Wholebody Control
Манипуляция + ходьба одновременно

### 6.2: Reactive Behaviors
Робот реагирует на окружение (камеры, датчики)

### 6.3: Multi-task Learning
Одна policy для разных задач

### 6.4: Language-conditioned
Управление голосом/текстом ("танцуй", "иди вперёд")

---

## Рекомендуемый путь

```
Сейчас
   │
   ▼
┌─────────────────────────────┐
│  Уровень 1: Простой танец   │  ← НАЧНИ ЗДЕСЬ
│  (1-2 дня)                  │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Уровень 2: Motion Capture  │
│  (3-5 дней)                 │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Уровень 3: Кастом ходьба   │
│  (1-2 недели)               │
└─────────────┬───────────────┘
              │
    ┌─────────┴─────────┐
    ▼                   ▼
┌────────────┐    ┌────────────┐
│ Уровень 4  │    │ Уровень 5  │
│ Манипуляция│    │ Реал робот │
│ (2-3 нед)  │    │ (когда есть│
└────────────┘    │  доступ)   │
                  └────────────┘
```

---

## Быстрый старт: Уровень 1

Хочешь начать прямо сейчас? Вот минимальный план:

### Сегодня (2-3 часа):
1. Создать скрипт генерации простого танца (махи руками)
2. Сгенерировать CSV файл
3. Конвертировать в NPZ

### Завтра:
4. Создать task config
5. Запустить тренировку
6. Тест в MuJoCo

Готов начать с Уровня 1?
