import os
import random
import json
import numpy as np
import random
import matplotlib.pyplot as plt


# ====================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ======================

def get_neighbors_4(i, j, n):
    if i > 0:
        yield i - 1, j
    if i + 1 < n:
        yield i + 1, j
    if j > 0:
        yield i, j - 1
    if j + 1 < n:
        yield i, j + 1


def is_allowed_candidate(i, j, n, allow_boundary_pores):
    return allow_boundary_pores or not (i in (0, n - 1) or j in (0, n - 1))


def add_frontier_from_pore(field, frontier, pore_i, pore_j, allow_boundary_pores):
    n = field.shape[0]

    for ni, nj in get_neighbors_4(pore_i, pore_j, n):
        if field[ni, nj] == 1 and is_allowed_candidate(ni, nj, n, allow_boundary_pores):
            frontier.add((ni, nj))


def build_frontier(field, allow_boundary_pores):
    frontier = set()

    for pore_i, pore_j in np.argwhere(field == 0):
        add_frontier_from_pore(field, frontier, int(pore_i), int(pore_j), allow_boundary_pores)

    return frontier


# ====================== БЫСТРАЯ ПРОВЕРКА СВЯЗНОСТИ ======================

def is_solid_connected(field):
    n = field.shape[0]
    solid_count = int(np.count_nonzero(field))

    if solid_count == 0:
        return False

    start_i, start_j = np.argwhere(field == 1)[0]
    visited = np.zeros(field.shape, dtype=bool)
    stack = [(int(start_i), int(start_j))]
    visited[start_i, start_j] = True
    visited_count = 0

    while stack:
        i, j = stack.pop()
        visited_count += 1

        for ni, nj in get_neighbors_4(i, j, n):
            if field[ni, nj] == 1 and not visited[ni, nj]:
                visited[ni, nj] = True
                stack.append((ni, nj))

    return visited_count == solid_count


def are_neighbors_locally_connected(field, neighbors, removed_i, removed_j):
    targets = set(neighbors[1:])
    if not targets:
        return True

    min_i = max(0, removed_i - 1)
    max_i = min(field.shape[0] - 1, removed_i + 1)
    min_j = max(0, removed_j - 1)
    max_j = min(field.shape[1] - 1, removed_j + 1)

    stack = [neighbors[0]]
    visited = {neighbors[0], (removed_i, removed_j)}

    while stack and targets:
        i, j = stack.pop()

        for ni, nj in get_neighbors_4(i, j, field.shape[0]):
            if ni < min_i or ni > max_i or nj < min_j or nj > max_j:
                continue
            if (ni, nj) in visited or field[ni, nj] == 0:
                continue

            if (ni, nj) in targets:
                targets.remove((ni, nj))

            visited.add((ni, nj))
            stack.append((ni, nj))

    return not targets


def can_remove_solid_cell(field, i, j):
    n = field.shape[0]
    solid_neighbors = [
        (ni, nj)
        for ni, nj in get_neighbors_4(i, j, n)
        if field[ni, nj] == 1
    ]

    if len(solid_neighbors) <= 1:
        return True

    if are_neighbors_locally_connected(field, solid_neighbors, i, j):
        return True

    targets = set(solid_neighbors[1:])
    stack = [solid_neighbors[0]]
    visited = np.zeros(field.shape, dtype=bool)
    visited[i, j] = True
    visited[solid_neighbors[0]] = True

    while stack and targets:
        ci, cj = stack.pop()

        for ni, nj in get_neighbors_4(ci, cj, n):
            if visited[ni, nj] or field[ni, nj] == 0:
                continue

            if (ni, nj) in targets:
                targets.remove((ni, nj))

            visited[ni, nj] = True
            stack.append((ni, nj))

    return not targets


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

        if can_remove_solid_cell(field, i, j):
            field[i, j] = 0
            seeds.append((i, j))

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

    field = np.ones((n, n), dtype=np.uint8)

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
    frontier = build_frontier(field, allow_boundary_pores)
    rejected = set()

    while current_solid > target_solid and steps < max_steps and failed_attempts < max_failed:
        steps += 1

        frontier.difference_update(rejected)
        if not frontier:
            break

        attempts = min(30, len(frontier))
        candidates = random.sample(tuple(frontier), attempts)

        removed = False

        for i, j in candidates:
            frontier.discard((i, j))

            if field[i, j] == 0:
                continue

            if can_remove_solid_cell(field, i, j):
                field[i, j] = 0
                current_solid -= 1
                removed = True
                failed_attempts = 0
                add_frontier_from_pore(field, frontier, i, j, allow_boundary_pores)
                break
            else:
                rejected.add((i, j))

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
    output_txt_path = os.path.join(sample_dir, "output_matrix.txt")
    info_path = os.path.join(sample_dir, "info.txt")

    # Вход для нейронки
    np.save(input_matrix_path, input_matrix.astype(np.float32))

    # Правильный ответ для нейронки
    np.save(output_matrix_path, output_matrix.astype(np.float32))

    np.savetxt(output_txt_path, output_matrix, fmt="%d")
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

    nn = 50
    dataset_dir = "dataset_pore"

    max_steps = 100000
    max_failed = 3000
    allow_boundary_pores = True

    sample_number = 0
    last_field = None

    os.makedirs(dataset_dir, exist_ok=True)


    ## ====================== НАСТРОЙКИ ДИАПАЗОНА ======================

    # Лучше начинать с небольшого количества вариантов.
    # Потом можно увеличить диапазон.

    vols = [x for x in range(60, 90, 10)]
    cnts_pore = [x for x in range(10, 2000, 20)]

    # Создаем все возможные комбинации
    combinations = [(vol, cnt) for vol in vols for cnt in cnts_pore]

    # Перемешиваем случайно
    random.shuffle(combinations)

    sample_number = 0  # счетчик успешных сохранений
    last_field = None

    for vol, cnt_pore in combinations:
        current_num = sample_number + 1  # номер для текущего образца
        
        print(f"\nГенерация sample{current_num}: vol={vol}, cnt_pore={cnt_pore}")

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
                sample_number=current_num
            )

            print(f"Сохранено:")
            print(f"  {input_path}")
            print(f"  {output_path}")

            last_field = output_matrix
            sample_number += 1  # увеличиваем только при успехе

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
    print(f"Всего успешно сохранено samples: {sample_number}")  # теперь правильно

    if last_field is not None:
        plot_porous_structure(last_field)
    else:
        print("Ни одна матрица не была успешно сгенерирована.")