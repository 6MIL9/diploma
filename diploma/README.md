# Rising Bubble PINN на PyTorch

Этот каталог содержит перенос training pipeline из `twophasePINN` на PyTorch для задачи rising bubble из статьи Buhendwa, Adami, Adams, 2021.

Исходный TensorFlow-проект не изменяется. Новый код находится в `diploma/pinn`.

## Данные

Скачайте `rising_bubble.h5` из ссылки авторов и положите файл сюда:

```powershell
d:\diploma_jupyter\twophasePINN\cfd_data\rising_bubble.h5
```

По умолчанию PyTorch-скрипт ищет данные именно там. Можно указать другой путь через `--data-path`.

## Установка зависимостей

В текущем `.venv` уже есть PyTorch, но нет `h5py`. Минимальная установка:

```powershell
.\.venv\Scripts\python.exe -m pip install -r diploma\requirements.txt
```

## Быстрый smoke-run

Команда ниже запускает маленькую конфигурацию, чтобы проверить, что данные читаются, автодифференцирование работает и checkpoint сохраняется:

```powershell
.\.venv\Scripts\python.exe diploma\train_rising_bubble.py --preset smoke
```

## Обучение

Конфигурация `paper` повторяет основные гиперпараметры авторов: 8 скрытых слоев по 350 нейронов, те же физические параметры, веса PDE и распределение точек. На CPU это будет очень долго.

```powershell
.\.venv\Scripts\python.exe diploma\train_rising_bubble.py --preset paper
```

Практичная стартовая конфигурация для дипломных экспериментов:

```powershell
.\.venv\Scripts\python.exe diploma\train_rising_bubble.py --preset default --epochs 2000 --device cuda
```

Если CUDA недоступна, используйте `--device cpu`.

## Оценка модели

```powershell
.\.venv\Scripts\python.exe -m pinn.evaluate --checkpoint diploma\checkpoints\<run>\best.pt --data-path twophasePINN\cfd_data\rising_bubble.h5
```

## Что перенесено

- сеть `(x, y, t) -> (u, v, p, alpha)`;
- граничные условия rising bubble из репозитория авторов;
- Volume-of-Fluid transport loss для `alpha`;
- невязки incompressible two-phase Navier-Stokes с переменными `rho(alpha)`, `mu(alpha)`, поверхностным натяжением и гравитацией;
- генерация точек около интерфейса, в near-field, в домене и на границах;
- сохранение checkpoint, истории loss и конфигурации запуска.
