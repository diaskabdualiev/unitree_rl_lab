# XR Teleoperate + Isaac Sim Integration Guide

**Дата**: 2025-12-24
**Система**: Ubuntu 22.04, 2x RTX 4080, Intel i9-13900K

---

## Оглавление

1. [Архитектура системы](#1-архитектура-системы)
2. [Требования](#2-требования)
3. [Порядок запуска](#3-порядок-запуска)
4. [Видеопоток с камер](#4-видеопоток-с-камер)
5. [Подключение VR шлема](#5-подключение-vr-шлема)
6. [Управление](#6-управление)
7. [Исправленные проблемы](#7-исправленные-проблемы)
8. [Troubleshooting](#8-troubleshooting)
9. [Полезные команды](#9-полезные-команды)

---

## 1. Архитектура системы

```
┌─────────────────────┐     DDS (domain 1)     ┌─────────────────────┐
│   xr_teleoperate    │◄─────────────────────►│ unitree_sim_isaaclab│
│   (conda: tv)       │     rt/lowstate        │ (conda: unitree_sim)│
│                     │     rt/lowcmd          │                     │
│   - VR tracking     │     rt/dex3_*          │   - Isaac Sim 5.0   │
│   - Hand retarget   │◄────────────────────────│   - Physics sim     │
│   - Arm IK          │     ZMQ (port 5555)    │   - Camera render   │
│   - TeleVuer        │     (video stream)     │   - ImageServer     │
└─────────────────────┘                        └─────────────────────┘
         ▲                                              │
         │ WebSocket (8012)                             │
         │ HTTPS + ZMQ video                            │
         ▼                                              ▼
┌─────────────────────┐                        ┌─────────────────────┐
│    VR Headset       │                        │   Simulation Window │
│ (Quest/Pico/Vision) │                        │   (Isaac Sim GUI)   │
└─────────────────────┘                        └─────────────────────┘
```

### Архитектура видеопотока

```
unitree_sim_isaaclab                      xr_teleoperate
┌──────────────────┐                     ┌──────────────────┐
│  Isaac Sim       │                     │  ImageClient     │
│  Camera Sensors  │                     │  (teleimager)    │
│        │         │                     │        │         │
│        ▼         │                     │        ▼         │
│  SharedMemory    │                     │  TeleVuer        │
│  (multi_image)   │                     │  (televuer)      │
│        │         │                     │        │         │
│        ▼         │     ZMQ:5555        │        ▼         │
│  ImageServer     │────────────────────►│  VR Display      │
│  (JPEG stream)   │   (tcp://localhost) │  (WebSocket)     │
└──────────────────┘                     └──────────────────┘
```

### DDS Topics

| Topic | Direction | Description |
|-------|-----------|-------------|
| `rt/lowstate` | Sim → Teleop | Joint states (29 joints) |
| `rt/lowcmd` | Teleop → Sim | Joint commands |
| `rt/dex3_left_cmd` | Teleop → Sim | Left hand commands |
| `rt/dex3_right_cmd` | Teleop → Sim | Right hand commands |
| `rt/dex3_left_state` | Sim → Teleop | Left hand state |
| `rt/dex3_right_state` | Sim → Teleop | Right hand state |
| `rt/sim_state` | Sim → Teleop | Simulation state |

---

## 2. Требования

### Conda окружения

| Окружение | Python | Isaac Sim | Назначение |
|-----------|--------|-----------|------------|
| `unitree_sim_env` | 3.11 | 5.0.0 | Симулятор |
| `tv` | 3.10 | - | VR телеуправление |

### Сетевые порты

| Порт | Протокол | Назначение |
|------|----------|------------|
| 8012 | HTTPS/WSS | VR интерфейс (televuer) |
| 5555 | ZMQ | Image streaming |
| 60000 | ZMQ | Camera config |

### Переменные окружения

```bash
# Обязательно для DDS связи между процессами
export CYCLONEDDS_URI=file:///home/dias/cyclonedds.xml
```

---

## 3. Порядок запуска

### Шаг 1: Запуск симулятора (Терминал 1)

```bash
export CYCLONEDDS_URI=file:///home/dias/cyclonedds.xml
conda activate unitree_sim_env
cd /home/dias/Documents/unitree/unitree_sim_isaaclab

python sim_main.py \
    --device cuda:0 \
    --enable_cameras \
    --task Isaac-PickPlace-Cylinder-G129-Dex3-Joint \
    --enable_dex3_dds \
    --robot_type g129
```

**Ожидаемый вывод:**
```
[DDSManager] register object 'g129' success
[DDSManager] register object 'dex3' success
[g1_robot] State publisher initialized (rt/lowstate)
***  Please left-click on the Sim window to activate rendering. ***
```

**Важно:** Кликни на окно симулятора для активации!

После клика:
```
controller started, start main loop...
```

### Шаг 2: Запуск телеуправления (Терминал 2)

```bash
export CYCLONEDDS_URI=file:///home/dias/cyclonedds.xml
conda activate tv
cd /home/dias/Documents/unitree/xr_teleoperate

python teleop/teleop_hand_and_arm.py \
    --arm=G1_29 \
    --ee=dex3 \
    --sim \
    --img-server-ip=127.0.0.1 \
    --display-mode=immersive
```

**Важно:** `--img-server-ip=127.0.0.1` — указывает на localhost для получения видео из симулятора (по умолчанию 192.168.123.164 — IP реального робота).

**Ожидаемый вывод:**
```
[G1_29_ArmController] Subscribe dds ok.
Initialize G1_29_ArmController OK!
[Dex3_1_Controller] Subscribe dds ok.
Initialize Dex3_1_Controller OK!
Please enter the start signal (enter 'r' to start the subsequent program)
```

### Шаг 3: Подключение VR шлема

См. раздел [5. Подключение VR шлема](#5-подключение-vr-шлема)

### Шаг 4: Начало телеуправления

В терминале телеуправления нажми `r`

---

## 4. Видеопоток с камер

### Как работает видеопоток

1. **Симулятор** рендерит 3 камеры (head, left_wrist, right_wrist)
2. Изображения записываются в **SharedMemory** (`isaac_multi_image_shm`)
3. **ImageServer** читает из SharedMemory и публикует через **ZMQ** на порт 5555
4. **ImageClient** (teleimager) подписывается на ZMQ и получает кадры
5. **TeleVuer** (televuer) отправляет изображения на VR шлем через WebSocket

### Конфигурация камер

Файл: `xr_teleoperate/teleop/teleimager/cam_config_client.yaml`

```yaml
# Конфигурация для СИМУЛЯЦИИ
head_camera:
  enable_zmq: true
  zmq_port: 5555          # Порт симулятора
  enable_webrtc: false
  type: simulation
  image_shape: [480, 1920]  # 640*3 камеры склеены
  binocular: false
  fps: 30

left_wrist_camera:
  enable_zmq: false       # Отключено для симуляции

right_wrist_camera:
  enable_zmq: false       # Отключено для симуляции
```

### Режимы отображения (display-mode)

| Режим | Описание | Видео |
|-------|----------|-------|
| `immersive` | Полный VR вид с камеры робота | Да (fullscreen) |
| `ego` | Маленькое окно + реальный мир | Да (small window) |
| `pass-through` | Только реальный мир через шлем | Нет |

Пример:
```bash
# Полный VR вид
python teleop/teleop_hand_and_arm.py --sim --display-mode=immersive --img-server-ip=127.0.0.1

# Маленькое окно в центре
python teleop/teleop_hand_and_arm.py --sim --display-mode=ego --img-server-ip=127.0.0.1

# Только управление без видео
python teleop/teleop_hand_and_arm.py --sim --display-mode=pass-through
```

### Проверка видеопотока

Тест что симулятор публикует изображения:

```bash
python -c "
import zmq
import cv2
import numpy as np

ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.connect('tcp://127.0.0.1:5555')
sock.setsockopt_string(zmq.SUBSCRIBE, '')
print('Ожидание изображений...')

while True:
    data = sock.recv()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        print(f'Получено: {img.shape}')
        cv2.imshow('Sim Camera', img)
        if cv2.waitKey(1) == ord('q'):
            break
cv2.destroyAllWindows()
"
```

### Параметры видео

| Параметр | Описание |
|----------|----------|
| `--img-server-ip` | IP сервера камер (127.0.0.1 для симуляции) |
| `--display-mode` | immersive / ego / pass-through |

---

## 5. Подключение VR шлема

### Поддерживаемые устройства

- Apple Vision Pro
- Meta Quest 3 / Quest 3S
- Pico 4 Ultra

### Настройка сети

1. Шлем должен быть в той же локальной сети (192.168.0.x)
2. IP компьютера: `192.168.0.181`

### Подключение

1. Открой браузер на шлеме (Safari / Quest Browser / Pico Browser)

2. Перейди по адресу:
   ```
   https://192.168.0.181:8012/?ws=wss://192.168.0.181:8012
   ```

3. При предупреждении о сертификате:
   - Нажми **"Advanced"** / **"Дополнительно"**
   - Нажми **"Proceed to site"** / **"Перейти на сайт"**

4. На странице нажми **"Virtual Reality"**

5. Разреши доступ к камерам/сенсорам

### Проверка локально (без шлема)

```bash
# В браузере на компьютере
https://localhost:8012

# Или через curl
curl -k https://localhost:8012
```

---

## 6. Управление

### Клавиши терминала

| Клавиша | Действие |
|---------|----------|
| `r` | Начать/остановить слежение за руками |
| `s` | Начать/остановить запись (если `--record`) |
| `q` | Выход из программы |

### Параметры запуска

| Параметр | Значения | Описание |
|----------|----------|----------|
| `--arm` | G1_29, G1_23, H1_2, H1 | Тип робота |
| `--ee` | dex3, dex1, inspire_dfx | Тип руки |
| `--sim` | flag | Режим симуляции |
| `--record` | flag | Запись данных |
| `--headless` | flag | Без GUI |
| `--motion` | flag | Режим с ходьбой |
| `--frequency` | float | Частота управления (Hz) |
| `--display-mode` | immersive, ego, pass-through | Режим VR отображения |
| `--img-server-ip` | IP address | IP сервера камер (127.0.0.1 для симуляции) |
| `--input-mode` | hand, controller | Режим ввода (трекинг рук / контроллеры) |

---

## 7. Исправленные проблемы

### 7.1 DDS не подключается (domain ID mismatch)

**Проблема:** `MotionSwitcher` инициализировал DDS с domain 0 вместо 1.

**Решение:** Добавлена проверка `args.sim` в `teleop_hand_and_arm.py:143`:
```python
elif not args.sim:  # Skip MotionSwitcher in simulation mode
    motion_switcher = MotionSwitcher()
```

### 7.2 Относительные пути не работают

**Проблема:** `../assets` не находится при запуске из другой директории.

**Решение:** Заменены относительные пути на абсолютные в:
- `robot_arm_ik.py` — добавлена переменная `_ASSETS_DIR`
- `hand_retargeting.py` — добавлена переменная `_ASSETS_DIR`

### 7.3 Несовместимость dex_retargeting

**Проблема:** pip версия 0.5.0 имеет другой API.

**Решение:** Используется субмодульная версия 0.4.7:
```bash
pip uninstall dex-retargeting
pip install -e teleop/robot_control/dex-retargeting --no-deps
pip install trimesh  # зависимость
```

### 7.4 DDS не работает между процессами

**Проблема:** CycloneDDS не настроен для локальной связи.

**Решение:** Создан конфиг `/home/dias/cyclonedds.xml`:
```xml
<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS xmlns="https://cdds.io/config">
    <Domain id="any">
        <General>
            <NetworkInterfaceAddress>lo</NetworkInterfaceAddress>
            <AllowMulticast>true</AllowMulticast>
            <EnableMulticastLoopback>true</EnableMulticastLoopback>
        </General>
    </Domain>
</CycloneDDS>
```

Переменная окружения:
```bash
export CYCLONEDDS_URI=file:///home/dias/cyclonedds.xml
```

---

## 8. Troubleshooting

### "Waiting to subscribe dds..." бесконечно

1. Проверь что симулятор запущен и активирован (кликнуть на окно)
2. Проверь что `CYCLONEDDS_URI` установлен в обоих терминалах
3. Проверь DDS тестом:
   ```bash
   python /home/dias/test_dds_subscriber.py
   ```

### Сайт не открывается на VR шлеме

1. Проверь что порт слушает:
   ```bash
   ss -tlnp | grep 8012
   ```

2. Открой firewall:
   ```bash
   sudo ufw allow 8012
   ```

3. Проверь локально:
   ```bash
   curl -k https://localhost:8012
   ```

4. Пинг до шлема:
   ```bash
   ping <IP шлема>
   ```

### "URDF dir not exists"

Запускай из корня проекта:
```bash
cd /home/dias/Documents/unitree/xr_teleoperate
python teleop/teleop_hand_and_arm.py ...
```

### Симулятор зависает / чёрный экран

1. Используй `--device cuda:0` (не `gpu`)
2. Обнови NVIDIA драйвер до 550+
3. Попробуй `--headless` режим

### numpy version conflicts

```bash
pip install numpy==1.26.0
```

### Нет видео на VR шлеме

1. Проверь что `--img-server-ip=127.0.0.1` указан (для симуляции)

2. Проверь что симулятор публикует видео:
   ```bash
   python -c "
   import zmq
   ctx = zmq.Context()
   sock = ctx.socket(zmq.SUB)
   sock.connect('tcp://127.0.0.1:5555')
   sock.setsockopt_string(zmq.SUBSCRIBE, '')
   print('Waiting...')
   data = sock.recv()
   print(f'Got {len(data)} bytes')
   "
   ```

3. Проверь `cam_config_client.yaml`:
   ```yaml
   head_camera:
     enable_zmq: true
     zmq_port: 5555
   ```

4. Убедись что `--display-mode=immersive` или `--display-mode=ego`

### Видео тормозит / низкий FPS

1. Используй `--device cuda:0` для GPU рендеринга
2. Уменьши разрешение в симуляторе
3. Проверь загрузку GPU: `nvidia-smi`

---

## 9. Полезные команды

### Проверка DDS

```bash
# Тест подписки
python /home/dias/test_dds_subscriber.py

# Тест Read() метода
python /home/dias/test_dds_read.py
```

### Добавить CYCLONEDDS_URI в bashrc

```bash
echo 'export CYCLONEDDS_URI=file:///home/dias/cyclonedds.xml' >> ~/.bashrc
source ~/.bashrc
```

### Список задач симулятора

```bash
conda activate unitree_sim_env
cd /home/dias/Documents/unitree/unitree_sim_isaaclab
grep -r "Isaac-" tasks/ | grep "class" | head -20
```

### Доступные задачи

| Задача | Робот | Рука |
|--------|-------|------|
| Isaac-PickPlace-Cylinder-G129-Dex3-Joint | G1-29dof | Dex3 |
| Isaac-PickPlace-Cylinder-G129-Dex1-Joint | G1-29dof | Dex1 |
| Isaac-PickPlace-RedBlock-G129-Dex3-Joint | G1-29dof | Dex3 |
| Isaac-Stack-RgyBlock-G129-Dex3-Joint | G1-29dof | Dex3 |

### Мониторинг GPU

```bash
watch -n 1 nvidia-smi
```

### Проверка видеопотока

```bash
# Тест ZMQ видеопотока из симулятора
python -c "
import zmq, cv2, numpy as np
ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.connect('tcp://127.0.0.1:5555')
sock.setsockopt_string(zmq.SUBSCRIBE, '')
print('Ожидание изображений (q для выхода)...')
while True:
    data = sock.recv()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        cv2.imshow('Sim Camera', img)
        if cv2.waitKey(1) == ord('q'):
            break
cv2.destroyAllWindows()
"
```

### Быстрый запуск (copy-paste)

```bash
# Терминал 1: Симулятор
export CYCLONEDDS_URI=file:///home/dias/cyclonedds.xml && \
conda activate unitree_sim_env && \
cd /home/dias/Documents/unitree/unitree_sim_isaaclab && \
python sim_main.py --device cuda:0 --enable_cameras \
  --task Isaac-PickPlace-Cylinder-G129-Dex3-Joint \
  --enable_dex3_dds --robot_type g129

# Терминал 2: Телеуправление
export CYCLONEDDS_URI=file:///home/dias/cyclonedds.xml && \
conda activate tv && \
cd /home/dias/Documents/unitree/xr_teleoperate && \
python teleop/teleop_hand_and_arm.py --arm=G1_29 --ee=dex3 --sim \
  --img-server-ip=127.0.0.1 --display-mode=immersive
```

---

## Файлы конфигурации

### /home/dias/cyclonedds.xml
CycloneDDS конфигурация для локальной DDS связи.

### ~/.config/xr_teleoperate/
SSL сертификаты для HTTPS:
- `cert.pem` — сертификат
- `key.pem` — приватный ключ

### teleop/teleimager/cam_config_client.yaml
Конфигурация камер для симуляции (ZMQ порт 5555).

### teleop/teleimager/cam_config_server.yaml
Конфигурация камер для реального робота (порты 55555-55557).

---

## Измененные файлы

| Файл | Изменение |
|------|-----------|
| `teleop/teleop_hand_and_arm.py` | Пропуск MotionSwitcher в sim mode, добавлен traceback |
| `teleop/robot_control/robot_arm_ik.py` | Абсолютные пути для URDF |
| `teleop/robot_control/hand_retargeting.py` | Абсолютные пути для assets |

---

## Версии пакетов

### tv environment
```
python: 3.10
pinocchio: 3.1.0
numpy: 1.26.4
dex_retargeting: 0.4.7 (submodule)
torch: 2.9.1+cpu
cyclonedds: 0.10.2
unitree_sdk2py: 1.0.1
```

### unitree_sim_env environment
```
python: 3.11
isaacsim: 5.0.0
isaaclab: 2.2.0
numpy: 1.26.0
torch: 2.7.0+cu128
cyclonedds: 0.10.2
unitree_sdk2py: 1.0.1
```
docker run -d \
  --name=wg-easy \
  -e WG_HOST=46.101.122.245 \
  -e PASSWORD=dias2025Vpn! \
  -e WG_DEFAULT_ADDRESS=10.0.0.x \
  -v ~/.wg-easy:/etc/wireguard \
  -p 51820:51820/udp \
  -p 51821:51821/tcp \
  --cap-add=NET_ADMIN \
  --cap-add=SYS_MODULE \
  --sysctl="net.ipv4.ip_forward=1" \
  --restart unless-stopped \
  ghcr.io/wg-easy/wg-easy