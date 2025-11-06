import matplotlib.pyplot as plt

import numpy as np
import csv

def read_csv_dataset(path_to_dataset: str) -> np.array:
    """ Чтение набора данных о деградации ёмкости по указанному пути """
    capacity = np.array([])
    with open(path_to_dataset, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=";", fieldnames=["time", "capacity"])
        for row in reader:
            capacity = np.append(capacity, float(row["capacity"].replace(",", ".")))
    return capacity


def plotting(x: np.ndarray, data: dict, save: bool = False):
    """ Отображение деградации ёмкости """
    for temp in data.keys():
        plt.plot(x, data[temp], label=f"T={temp}К, U=2.9В")
    plt.grid(True)
    plt.xlim([x[0], x[-1] * 1.02])
    plt.xlabel("t, ч")
    plt.ylabel("C, нФ")
    plt.legend()
    if save:
        plt.savefig("degradation_capacity_area.png", dpi=300, bbox_inches='tight')
    plt.show()


def plotting_compare(
    time: np.ndarray,
    experiment_data: dict, model_data: dict
) -> None:
    """ Отображение графиков модели и эксперимента на одном рисунке """
    colors = ["b", "orange", "g"]
    for color, temp in zip(colors, experiment_data.keys()):
        plt.plot(time, experiment_data[temp], color=color, label=f"Эксперимент (T={temp}К, U=2.9В)")
        plt.plot(time, model_data[temp], "--", color=color, label=f"Модель (T={temp}К, U=2.9В)")
    plt.xlim(0.95 * time[0], time[-1] * 1.025)
    plt.grid(True)
    plt.xlabel("t, час")
    plt.ylabel("C, нФ")
    plt.legend()
    plt.show()

def get_model_accuracy(data_true: dict[int: np.ndarray], data_pred: dict[int: np.ndarray]):
    """ Получение точности модели """
    assert data_pred.keys() == data_true.keys(), "Разные ключи словарей"
    accuracy_data = {}
    for temp in data_true.keys():
        assert data_true[temp].shape == data_pred[temp].shape, "Разный размер массивов"

        dc = np.abs((data_true[temp] - data_pred[temp]) / data_true[temp] * 100)
        dc = np.round(dc, 2)
        accuracy_data[temp] = [np.min(dc), np.max(dc)]

    return accuracy_data
