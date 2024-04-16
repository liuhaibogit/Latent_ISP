import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import NoReturn



def plot_error(log_dir: str, csv_name: str = 'optimization.csv',
                        init_epoch: int = 0, last_epoch: int = -1) -> NoReturn:

    df = pd.read_csv(os.path.join(log_dir, csv_name))
    if last_epoch == -1:
        last_epoch = df['epoch'].max()
    df = df[df['epoch'] >= init_epoch]
    df = df[df['epoch'] <= last_epoch]


    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    # # 设置标题
    # ax1.set_title('PCA')
    # # 设置横坐标名称
    # ax1.set_xlabel('x')
    # # 设置纵坐标名称
    # ax1.set_ylabel('y')
    ax1.plot(df['epoch'], df['loss'], c='b')
    plt.savefig(os.path.join(log_dir, 'loss.png'))
    plt.close()


    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)
    # # 设置标题
    # ax1.set_title('PCA')
    # # 设置横坐标名称
    # ax1.set_xlabel('x')
    # # 设置纵坐标名称
    # ax1.set_ylabel('y')
    ax2.plot(df['epoch'], df['indicator error'], c='b')
    plt.savefig(os.path.join(log_dir, 'indicator_error.png'))
    plt.close()