# G1-23DOF Mimic Policy Debug Report

## Проблема

**Симптомы:**
- В Isaac Lab: velocity, mimic (bata_dias, dance_102, gangnam_style) для G1-23dof работают нормально
- В MuJoCo/реальном роботе:
  - `passive`, `fixstand`, `velocity` - работают корректно
  - `mimic` политики (bata_dias, dance_102, gangnam_style) - робот "сходит с ума", конечности разворачиваются неправильно
  - На реальном роботе: аварийное отключение с ошибкой

## Анализ

### 1. Сравнение конфигураций

#### Velocity (РАБОТАЕТ) - `UNITREE_G1_23DOF_CFG`

```python
# unitree.py:366-394
joint_sdk_names=[
    "left_hip_pitch_joint",       # SDK 0
    "left_hip_roll_joint",        # SDK 1
    "left_hip_yaw_joint",         # SDK 2
    "left_knee_joint",            # SDK 3
    "left_ankle_pitch_joint",     # SDK 4
    "left_ankle_roll_joint",      # SDK 5
    "right_hip_pitch_joint",      # SDK 6
    "right_hip_roll_joint",       # SDK 7
    "right_hip_yaw_joint",        # SDK 8
    "right_knee_joint",           # SDK 9
    "right_ankle_pitch_joint",    # SDK 10
    "right_ankle_roll_joint",     # SDK 11
    "waist_yaw_joint",            # SDK 12
    "",                           # SDK 13 (waist_roll - НЕ СУЩЕСТВУЕТ)
    "",                           # SDK 14 (waist_pitch - НЕ СУЩЕСТВУЕТ)
    "left_shoulder_pitch_joint",  # SDK 15
    "left_shoulder_roll_joint",   # SDK 16
    "left_shoulder_yaw_joint",    # SDK 17
    "left_elbow_joint",           # SDK 18
    "left_wrist_roll_joint",      # SDK 19
    "",                           # SDK 20 (left_wrist_pitch - НЕ СУЩЕСТВУЕТ)
    "",                           # SDK 21 (left_wrist_yaw - НЕ СУЩЕСТВУЕТ)
    "right_shoulder_pitch_joint", # SDK 22
    "right_shoulder_roll_joint",  # SDK 23
    "right_shoulder_yaw_joint",   # SDK 24
    "right_elbow_joint",          # SDK 25
    "right_wrist_roll_joint",     # SDK 26
]
```

**Ключевое:** 27 элементов с `""` плейсхолдерами для несуществующих SDK моторов.

#### Mimic (НЕ РАБОТАЛО) - `UNITREE_G1_23DOF_MIMIC_CFG`

```python
# unitree.py:820-844 (ДО ИСПРАВЛЕНИЯ)
joint_sdk_names=[
    "left_hip_pitch_joint",       # позиция 0  → SDK 0 ✓
    "left_hip_roll_joint",        # позиция 1  → SDK 1 ✓
    "left_hip_yaw_joint",         # позиция 2  → SDK 2 ✓
    "left_knee_joint",            # позиция 3  → SDK 3 ✓
    "left_ankle_pitch_joint",     # позиция 4  → SDK 4 ✓
    "left_ankle_roll_joint",      # позиция 5  → SDK 5 ✓
    "right_hip_pitch_joint",      # позиция 6  → SDK 6 ✓
    "right_hip_roll_joint",       # позиция 7  → SDK 7 ✓
    "right_hip_yaw_joint",        # позиция 8  → SDK 8 ✓
    "right_knee_joint",           # позиция 9  → SDK 9 ✓
    "right_ankle_pitch_joint",    # позиция 10 → SDK 10 ✓
    "right_ankle_roll_joint",     # позиция 11 → SDK 11 ✓
    "waist_yaw_joint",            # позиция 12 → SDK 12 ✓
    "left_shoulder_pitch_joint",  # позиция 13 → SDK 13 ✗ ДОЛЖЕН БЫТЬ SDK 15!
    "left_shoulder_roll_joint",   # позиция 14 → SDK 14 ✗ ДОЛЖЕН БЫТЬ SDK 16!
    "left_shoulder_yaw_joint",    # позиция 15 → SDK 15 ✗ ДОЛЖЕН БЫТЬ SDK 17!
    "left_elbow_joint",           # позиция 16 → SDK 16 ✗ ДОЛЖЕН БЫТЬ SDK 18!
    "left_wrist_roll_joint",      # позиция 17 → SDK 17 ✗ ДОЛЖЕН БЫТЬ SDK 19!
    "right_shoulder_pitch_joint", # позиция 18 → SDK 18 ✗ ДОЛЖЕН БЫТЬ SDK 22!
    "right_shoulder_roll_joint",  # позиция 19 → SDK 19 ✗ ДОЛЖЕН БЫТЬ SDK 23!
    "right_shoulder_yaw_joint",   # позиция 20 → SDK 20 ✗ ДОЛЖЕН БЫТЬ SDK 24!
    "right_elbow_joint",          # позиция 21 → SDK 21 ✗ ДОЛЖЕН БЫТЬ SDK 25!
    "right_wrist_roll_joint",     # позиция 22 → SDK 22 ✗ ДОЛЖЕН БЫТЬ SDK 26!
]
```

**Ошибка:** 23 элемента БЕЗ плейсхолдеров. Позиция в массиве = SDK индекс, что неверно!

### 2. Результирующий `joint_ids_map`

При экспорте `deploy.yaml` функция `export_deploy_cfg.py` генерирует `joint_ids_map` на основе `joint_sdk_names`:

```python
joint_ids_map, _ = resolve_matching_names(asset.data.joint_names, joint_sdk_names, preserve_order=True)
```

#### Velocity deploy.yaml (ПРАВИЛЬНО):
```yaml
joint_ids_map: [0, 6, 12, 1, 7, 15, 22, 2, 8, 16, 23, 3, 9, 17, 24, 4, 10, 18, 25, 5, 11, 19, 26]
```
- Использует SDK индексы: 0-12, **15-19, 22-26**
- Пропускает: 13, 14, 20, 21

#### Mimic deploy.yaml (НЕПРАВИЛЬНО):
```yaml
joint_ids_map: [0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22]
```
- Использует SDK индексы: 0-22
- **Включает несуществующие:** 13, 14, 20, 21

### 3. Как это влияет на робота

`joint_ids_map` используется в `State_Mimic::run()` для отправки команд на моторы:

```cpp
void State_Mimic::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
```

#### Что происходило с неправильным mapping:

| Isaac Lab Joint | Ожидаемый SDK Motor | Фактический SDK Motor | Результат |
|-----------------|---------------------|----------------------|-----------|
| left_shoulder_pitch | 15 | **13** (waist_roll) | Команда руки идёт на несуществующий мотор поясницы |
| left_shoulder_roll | 16 | **14** (waist_pitch) | Команда руки идёт на несуществующий мотор поясницы |
| right_shoulder_pitch | 22 | **18** (left_elbow) | Команда правой руки идёт на левый локоть! |
| right_elbow | 25 | **21** (left_wrist_yaw) | Команда правого локтя идёт на несуществующий мотор |

**Результат:** Робот получает команды на неправильные моторы → конечности двигаются хаотично → аварийное отключение.

### 4. Дополнительная проблема: `stiffness/damping`

В `State_Mimic::enter()` stiffness/damping применяются напрямую по индексу:

```cpp
void State_Mimic::enter()
{
    for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
    {
        lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
        lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
    }
}
```

- Mimic deploy.yaml имел 23-элементные массивы stiffness/damping
- Velocity deploy.yaml имел 27-элементные массивы с нулями на позициях 13, 14, 20, 21
- При 23 элементах: `motor_cmd[13-22]` получают неправильные значения kp/kd

## Исправления

### 1. Исправлен `UNITREE_G1_23DOF_MIMIC_CFG.joint_sdk_names`

**Файл:** `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py:820-849`

```python
# SDK order with placeholders for non-existent joints (same as UNITREE_G1_23DOF_CFG)
joint_sdk_names=[
    "left_hip_pitch_joint",       # SDK 0
    "left_hip_roll_joint",        # SDK 1
    "left_hip_yaw_joint",         # SDK 2
    "left_knee_joint",            # SDK 3
    "left_ankle_pitch_joint",     # SDK 4
    "left_ankle_roll_joint",      # SDK 5
    "right_hip_pitch_joint",      # SDK 6
    "right_hip_roll_joint",       # SDK 7
    "right_hip_yaw_joint",        # SDK 8
    "right_knee_joint",           # SDK 9
    "right_ankle_pitch_joint",    # SDK 10
    "right_ankle_roll_joint",     # SDK 11
    "waist_yaw_joint",            # SDK 12
    "",                           # SDK 13 (waist_roll - not present in 23dof)
    "",                           # SDK 14 (waist_pitch - not present in 23dof)
    "left_shoulder_pitch_joint",  # SDK 15
    "left_shoulder_roll_joint",   # SDK 16
    "left_shoulder_yaw_joint",    # SDK 17
    "left_elbow_joint",           # SDK 18
    "left_wrist_roll_joint",      # SDK 19
    "",                           # SDK 20 (left_wrist_pitch - not present in 23dof)
    "",                           # SDK 21 (left_wrist_yaw - not present in 23dof)
    "right_shoulder_pitch_joint", # SDK 22
    "right_shoulder_roll_joint",  # SDK 23
    "right_shoulder_yaw_joint",   # SDK 24
    "right_elbow_joint",          # SDK 25
    "right_wrist_roll_joint",     # SDK 26
],
```

### 2. Исправлены deploy.yaml для всех mimic политик

**Файлы:**
- `deploy/robots/g1_23dof/config/policy/mimic/bata_dias/params/deploy.yaml`
- `deploy/robots/g1_23dof/config/policy/mimic/gangnam_style/params/deploy.yaml`
- `deploy/robots/g1_23dof/config/policy/mimic/dance_102/params/deploy.yaml`

**Изменения:**

```yaml
# Правильный joint_ids_map (как у velocity)
joint_ids_map: [0, 6, 12, 1, 7, 15, 22, 2, 8, 16, 23, 3, 9, 17, 24, 4, 10, 18, 25, 5, 11, 19, 26]

# Stiffness в SDK порядке (27 элементов, нули на 13,14,20,21)
stiffness: [40.2, 99.1, 40.2, 99.1, 28.5, 28.5, 40.2, 99.1, 40.2, 99.1, 28.5, 28.5,
  40.2, 0, 0, 14.3, 14.3, 14.3, 14.3, 16.8, 0, 0, 14.3, 14.3, 14.3, 14.3, 16.8]

# Damping в SDK порядке (27 элементов, нули на 13,14,20,21)
damping: [2.56, 6.31, 2.56, 6.31, 1.81, 1.81, 2.56, 6.31, 2.56, 6.31, 1.81, 1.81,
  2.56, 0, 0, 0.907, 0.907, 0.907, 0.907, 1.07, 0, 0, 0.907, 0.907, 0.907, 0.907, 1.07]
```

## G1 SDK Motor Layout (Reference)

```
SDK Index | Joint Name              | G1-29DOF | G1-23DOF
----------|-------------------------|----------|----------
0         | left_hip_pitch_joint    | ✓        | ✓
1         | left_hip_roll_joint     | ✓        | ✓
2         | left_hip_yaw_joint      | ✓        | ✓
3         | left_knee_joint         | ✓        | ✓
4         | left_ankle_pitch_joint  | ✓        | ✓
5         | left_ankle_roll_joint   | ✓        | ✓
6         | right_hip_pitch_joint   | ✓        | ✓
7         | right_hip_roll_joint    | ✓        | ✓
8         | right_hip_yaw_joint     | ✓        | ✓
9         | right_knee_joint        | ✓        | ✓
10        | right_ankle_pitch_joint | ✓        | ✓
11        | right_ankle_roll_joint  | ✓        | ✓
12        | waist_yaw_joint         | ✓        | ✓
13        | waist_roll_joint        | ✓        | ✗
14        | waist_pitch_joint       | ✓        | ✗
15        | left_shoulder_pitch     | ✓        | ✓
16        | left_shoulder_roll      | ✓        | ✓
17        | left_shoulder_yaw       | ✓        | ✓
18        | left_elbow_joint        | ✓        | ✓
19        | left_wrist_roll_joint   | ✓        | ✓
20        | left_wrist_pitch_joint  | ✓        | ✗
21        | left_wrist_yaw_joint    | ✓        | ✗
22        | right_shoulder_pitch    | ✓        | ✓
23        | right_shoulder_roll     | ✓        | ✓
24        | right_shoulder_yaw      | ✓        | ✓
25        | right_elbow_joint       | ✓        | ✓
26        | right_wrist_roll_joint  | ✓        | ✓
27        | right_wrist_pitch_joint | ✓        | ✗
28        | right_wrist_yaw_joint   | ✓        | ✗
```

## Рекомендации

1. **После исправления deploy.yaml:** Существующие обученные модели могут работать, так как policy.onnx обучался с правильными observations в Isaac Lab.

2. **Для гарантированного результата:** Переобучить mimic политики с исправленным конфигом:
   ```bash
   ./unitree_rl_lab.sh -t --task Unitree-G1-23dof-Mimic-Bata-Dias --headless --num_envs 4096
   ```

3. **При создании новых конфигов для 23DOF:** Всегда использовать 27-элементный `joint_sdk_names` с плейсхолдерами `""` для несуществующих моторов.

## Коммит

```
9757cd5 fix(g1-23dof): correct joint_ids_map for mimic policies
```

---

*Документ создан: 2026-01-03*
*Автор анализа: Claude Code*
