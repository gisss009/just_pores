import numpy as np
import matplotlib.pyplot as plt


def draw_matrix(file_path):
    matrix = np.loadtxt(file_path)

    plt.figure(figsize=(7, 7))
    plt.imshow(matrix, cmap='gray_r', interpolation='nearest')
    
    plt.title("Matrix")
    plt.xticks([])
    plt.yticks([])
    
    plt.show()


if __name__ == '__main__':
    for i in range(0, 16):
        path = rf'C:\Users\userd\Desktop\Новая папка\just_pores\dataset_pore\sample{i}\output_matrix.txt'
        draw_matrix(
            path
        )