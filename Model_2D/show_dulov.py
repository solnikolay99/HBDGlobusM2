import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


if __name__ == '__main__':
    dens_dulov = np.loadtxt('data/dulov/Dulov_check.txt')
    x_data_ = np.loadtxt('data/dulov/xx_Dulov_check.txt')
    y_data_ = np.loadtxt('data/dulov/yy_Dulov_check.txt')

    fig3 = plt.figure(figsize=(10, 8))

    # Используем pcolormesh для отображения с логарифмической шкалой
    mesh = plt.pcolormesh(y_data_, x_data_, dens_dulov, norm=LogNorm(vmin=1, vmax=30000), cmap='jet', shading='auto')

    # Добавляем цветовую шкалу
    cbar = plt.colorbar(mesh)
    cbar.set_label('Pa')

    # Настройки графика
    plt.xlabel('X, m')
    plt.ylabel('Y, m')
    plt.ylim(0.005, 0.025)
    plt.title('Плотность по Дулову')

    #plt.show()
    plt.savefig(os.getcwd() + '/data/dulov/Dulov.png')
