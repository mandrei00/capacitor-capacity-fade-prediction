import numpy as np

class CapacityDegradationModel:

    K_BOLZ = 1.38 * (10 ** -23)

    """
    Модель взята из статьи:
    Sun, Bo, et al. 
    "PoF-simulation-assisted reliability prediction for electrolytic capacitor in LED drivers." 
    IEEE Transactions on Industrial Electronics 63.11 (2016): 6726-6735.
    """

    def __init__(self, crit_ratio_c: float, c_0: float):
        self.crit_ratio_c = crit_ratio_c
        self.c_0 = c_0
        self.data_ttf: dict | None = None

        self.e_a = None

    def get_failure_line(self, time):
        failure_line_capacity = np.ones_like(time) * self.c_0 * self.crit_ratio_c
        return failure_line_capacity

    def get_failure_time(self, time, capacity) -> float:
        """ Метод расчёта времени до отказа """
        failure_line_capacity = self.get_failure_line(time)
        return time[np.argmax(capacity < np.max(failure_line_capacity))]


    def fit(self, time: np.ndarray, experiment_data: dict[float, np.ndarray]):
        """ Обучение модели и получения значения энергии активации, удельной скорости деградации """
        x_reverse_temp = np.array([])
        y_ttf = np.array([])

        self.data_ttf = {}
        for temperature, capacity in experiment_data.items():
            x_reverse_temp = np.append(x_reverse_temp, 1 / temperature)
            ttf = self.get_failure_time(time, capacity)
            y_ttf = np.append(y_ttf, ttf)
            self.data_ttf[temperature] = ttf
        y_ln_ttf = np.log(y_ttf)

        # расчёт энергии активации и фактора частоты
        k, b = np.polyfit(x_reverse_temp, y_ln_ttf, 1)
        self.e_a = k * self.K_BOLZ
        # a = np.exp(b)
        return self.e_a

    def get_degradation_rate(self, time_to_failure: float, temperature: float):
        """ Метод расчёта скорости деградации """
        d_c0 = -(1 - self.crit_ratio_c) * np.exp(self.e_a / (self.K_BOLZ * temperature)) / time_to_failure
        return d_c0

    def predict(self, time: np.ndarray, model_data: dict):
        """ Предсказание кривой ёмкости в зависимости от времени """
        for temperature in model_data.keys():
            # # расчёт удельной скорости деградации
            d_c0 = self.get_degradation_rate(self.data_ttf[temperature], temperature)
            d_c = d_c0 * np.exp(-self.e_a / (self.K_BOLZ * temperature))
            model_data[temperature] = self.c_0 * (1 + d_c * time)
        return model_data


