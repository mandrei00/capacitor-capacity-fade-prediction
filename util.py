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


def plotting(x: np.ndarray, data: dict, path_to_save: str|None = None):
    """ Отображение деградации ёмкости """
    for temp in data.keys():
        plt.plot(x, data[temp], label=f"T={temp}К, U=2.9В")
    plt.grid(True)
    plt.xlim([x[0], x[-1] * 1.02])
    plt.xlabel("t, ч")
    plt.ylabel("C, нФ")
    plt.legend()
    if path_to_save is not None:
        plt.savefig(f"{path_to_save}", dpi=300, bbox_inches='tight')
    plt.show()


def plotting_compare(
    time: np.ndarray,
    experiment_data: dict, model_data: dict,
    path_to_save: str|None = None
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
    if path_to_save is not None:
        plt.savefig(f"{path_to_save}", dpi=300, bbox_inches='tight')
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


def plot_residuals_subplots(
        residual_error_data,
        model_predictions,
        path_to_save,
        figsize=(10, 15),
):
    """
    Строит графики остатков для каждой температуры в subplots одного окна (вертикально)
    """

    if set(residual_error_data.keys()) != set(model_predictions.keys()):
        raise ValueError("Ключи в residual_error_data и model_predictions не совпадают")

    n_temps = len(residual_error_data)

    # Изменено: создаем subplots по вертикали (n_temps, 1) вместо (1, n_temps)
    fig, axes = plt.subplots(n_temps, 1, figsize=figsize)

    # Если только одна температура, преобразуем axes в массив
    if n_temps == 1:
        axes = [axes]

    for i, (temp_key, ax) in enumerate(zip(residual_error_data.keys(), axes)):
        errors = residual_error_data[temp_key]
        predictions = model_predictions[temp_key]

        if len(errors) != len(predictions):
            raise ValueError(f"Длины массивов для {temp_key} не совпадают")

        # Строим график на соответствующем subplot
        ax.scatter(predictions, errors, alpha=0.7, color='red', label='Остатки')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, label='Нулевая ошибка')

        ax.set_title(f'Остатки: {temp_key} К', fontsize=12, fontweight='bold')
        ax.set_xlabel('Предсказанные значения, С')
        ax.set_ylabel('Остатки (Ошибка)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Статистическая информация
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        ax.text(0.05, 0.95, f'Среднее: {mean_error:.4f}\nСтд: {std_error:.4f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    if path_to_save is not None:
        plt.savefig(f"{path_to_save}", dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()