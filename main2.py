import os
import json
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt


# ====================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ======================

def get_neighbors_4(i, j, n):
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    return [
        (i + di, j + dj)
        for di, dj in dirs
        if 0 <= i + di < n and 0 <= j + dj < n
    ]


# ====================== БЫСТРАЯ ПРОВЕРКА СВЯЗНОСТИ ======================

def is_solid_connected(field):
    n = field.shape[0]
    solid_positions = np.argwhere(field == 1)

    if len(solid_positions) == 0:
        return False

    start = tuple(solid_positions[0])
    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        i, j = queue.popleft()

        for ni, nj in get_neighbors_4(i, j, n):
            if field[ni, nj] == 1 and (ni, nj) not in visited:
                visited.add((ni, nj))
                queue.append((ni, nj))

    return len(visited) == len(solid_positions)


# ====================== СОЗДАНИЕ НАЧАЛЬНЫХ ПОР ======================

def create_pore_seeds(field, num_pores, min_distance=5):
    n = field.shape[0]
    seeds = []

    if n < 5:
        raise ValueError("Размер матрицы n должен быть хотя бы 5")

    for _ in range(num_pores * 3):
        if len(seeds) >= num_pores:
            break

        i = random.randint(2, n - 3)
        j = random.randint(2, n - 3)

        if any(abs(i - si) + abs(j - sj) < min_distance for si, sj in seeds):
            continue

        field[i, j] = 0

        if is_solid_connected(field):
            seeds.append((i, j))
        else:
            field[i, j] = 1

    return seeds


# ====================== ОСНОВНАЯ ФУНКЦИЯ ГЕНЕРАЦИИ ======================

def generate_porous_structure(
    n=60,
    solid_percent=55,
    num_pores=12,
    max_steps=50000,
    allow_boundary_pores=False,
    max_failed=3000
):
    if not (0 < solid_percent <= 100):
        raise ValueError("solid_percent должен быть в диапазоне от 1 до 100")

    if num_pores <= 0:
        raise ValueError("num_pores должен быть больше 0")

    if n <= 0:
        raise ValueError("n должен быть больше 0")

    field = np.ones((n, n), dtype=int)

    total_cells = n * n
    target_solid = int(round(total_cells * solid_percent / 100))

    seeds = create_pore_seeds(field, num_pores, min_distance=6)

    if not seeds:
        raise RuntimeError("Не удалось создать начальные поры")

    # Это вход для нейронки: матрица только с начальными порами
    input_matrix = field.copy()

    current_solid = int(np.sum(field == 1))

    steps = 0
    failed_attempts = 0

    while current_solid > target_solid and steps < max_steps and failed_attempts < max_failed:
        steps += 1

        candidates = set()

        for i in range(n):
            for j in range(n):
                if field[i, j] == 0:
                    for ni, nj in get_neighbors_4(i, j, n):
                        if field[ni, nj] == 1:
                            if allow_boundary_pores or not (ni in (0, n - 1) or nj in (0, n - 1)):
                                candidates.add((ni, nj))

        if not candidates:
            break

        candidates = list(candidates)
        random.shuffle(candidates)

        removed = False

        for i, j in candidates[:30]:
            field[i, j] = 0

            if is_solid_connected(field):
                current_solid -= 1
                removed = True
                failed_attempts = 0
                break
            else:
                field[i, j] = 1

        if not removed:
            failed_attempts += 1

    real_solid = np.sum(field == 1) / total_cells * 100
    porosity = 100 - real_solid
    connected = is_solid_connected(field)

    info = {
        "n": n,
        "solid_percent_target": solid_percent,
        "solid_percent_real": round(real_solid, 4),
        "porosity_percent": round(porosity, 4),
        "num_pores_requested": num_pores,
        "num_pores_created": len(seeds),
        "max_steps": max_steps,
        "steps_done": steps,
        "max_failed": max_failed,
        "failed_attempts": failed_attempts,
        "allow_boundary_pores": allow_boundary_pores,
        "is_solid_connected": connected,
        "seeds": seeds
    }

    return input_matrix, field, info


# ====================== СОХРАНЕНИЕ ОДНОГО SAMPLE ======================

def save_sample(input_matrix, output_matrix, info, dataset_dir, sample_number):
    sample_dir = os.path.join(dataset_dir, f"sample{sample_number}")
    os.makedirs(sample_dir, exist_ok=True)

    input_matrix_path = os.path.join(sample_dir, "input_matrix.npy")
    output_matrix_path = os.path.join(sample_dir, "output_matrix.npy")
    info_path = os.path.join(sample_dir, "info.txt")

    # Вход для нейронки
    np.save(input_matrix_path, input_matrix.astype(np.float32))

    # Правильный ответ для нейронки
    np.save(output_matrix_path, output_matrix.astype(np.float32))

    # Описание для человека
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("SAMPLE INFO\n")
        f.write("====================\n")

        for key, value in info.items():
            f.write(f"{key}: {value}\n")

    return input_matrix_path, output_matrix_path, info_path
# ====================== СОХРАНЕНИЕ ОШИБОК ======================

def save_error(dataset_dir, vol, cnt_pore, error_text):
    os.makedirs(dataset_dir, exist_ok=True)

    error_log_path = os.path.join(dataset_dir, "errors.txt")

    with open(error_log_path, "a", encoding="utf-8") as f:
        f.write("====================\n")
        f.write(f"solid_percent: {vol}\n")
        f.write(f"num_pores: {cnt_pore}\n")
        f.write(f"error: {error_text}\n\n")


# ====================== ВИЗУАЛИЗАЦИЯ ======================

def plot_porous_structure(field):
    plt.figure(figsize=(9, 9))
    plt.imshow(field, cmap="gray_r", interpolation="nearest")
    plt.title("2D porous structure: 1 — solid, 0 — pores")
    plt.grid(True, color="black", linewidth=0.3)
    plt.show()


# ====================== ЗАПУСК ГЕНЕРАЦИИ ДАТАСЕТА ======================

if __name__ == "__main__":

    nn = 100
    dataset_dir = "dataset_pore"

    max_steps = 100000
    max_failed = 3000
    allow_boundary_pores = True

    sample_number = 0
    last_field = None

    os.makedirs(dataset_dir, exist_ok=True)

    # Лучше начинать с небольшого количества вариантов.
    # Потом можно увеличить диапазон.
    for vol in range(10, 30, 10):
        for cnt_pore in range(180, 201, 20):

            print(f"\nГенерация sample{sample_number}: vol={vol}, cnt_pore={cnt_pore}")

            try:
                input_matrix, output_matrix, info = generate_porous_structure(
                    n=nn,
                    solid_percent=vol,
                    num_pores=cnt_pore,
                    max_steps=max_steps,
                    allow_boundary_pores=allow_boundary_pores,
                    max_failed=max_failed
                )

                input_path, output_path, info_path = save_sample(
                    input_matrix=input_matrix,
                    output_matrix=output_matrix,
                    info=info,
                    dataset_dir=dataset_dir,
                    sample_number=sample_number
                )

                print(f"Сохранено:")
                print(f"  {input_path}")
                print(f"  {output_path}")

                last_field = output_matrix
                sample_number += 1

            except Exception as e:
                print(f"Ошибка. Пропуск vol={vol}, cnt_pore={cnt_pore}")
                print(f"Причина: {e}")

                save_error(
                    dataset_dir=dataset_dir,
                    vol=vol,
                    cnt_pore=cnt_pore,
                    error_text=str(e)
                )

    print("\nГенерация завершена.")
    print(f"Всего успешно сохранено samples: {sample_number}")

    if last_field is not None:
        plot_porous_structure(last_field)
    else:
        print("Ни одна матрица не была успешно сгенерирована.")