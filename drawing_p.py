import numpy as np
import matplotlib.pyplot as plt
import random


def draw_matrix(file_path):
    matrix = np.loadtxt(file_path)

    plt.figure(figsize=(7, 7))
    plt.imshow(matrix, cmap='gray_r', interpolation='nearest')
    
    plt.title("Matrix")
    plt.xticks([])
    plt.yticks([])
    
    plt.show()


def draw_matrix1(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        matrix = np.array([
            [int(char) for char in line.strip()]
            for line in f
            if line.strip()
        ])

    print(matrix.shape)

    plt.imshow(matrix, cmap="gray_r")
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    for i in range(1, 10):
        n = random.randint(1, 1501)
        path = rf'C:\Users\userd\Desktop\Новая папка\just_pores\update_dataset_pore\sample{n}.txt'
        draw_matrix1(
            path
        )