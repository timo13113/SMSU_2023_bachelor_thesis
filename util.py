from lib import *

font = {'size': 15}

matplotlib.rc('font', **font)


def period_begginings(period='month', data_type='24'):
    coef = 1
    if data_type == '6':
        coef = 4

    res = []
    base_date = date(year=1979, month=1, day=1)
    if period == 'month':
        for year in range(1979, 2022):
            for month in range(1, 13):
                cur_date = date(year=year, month=month, day=1)
                res.append((cur_date - base_date).days * coef)

    elif period == 'quarter':
        for year in range(1979, 2022):
            for month in range(1, 13, 3):
                cur_date = date(year=year, month=month, day=1)
                res.append((cur_date - base_date).days * coef)

    elif period == 'halfyear':
        for year in range(1979, 2022):
            for month in range(1, 13, 6):
                cur_date = date(year=year, month=month, day=1)
                res.append((cur_date - base_date).days * coef)

    elif period == 'year':
        for year in range(1979, 2022):
            cur_date = date(year=year, month=1, day=1)
            res.append((cur_date - base_date).days * coef)

    elif period == '5_years':
        for year in range(1979, 2022, 5):
            cur_date = date(year=year, month=1, day=1)
            res.append((cur_date - base_date).days * coef)

    elif period == '10_years':
        for year in range(1979, 2022, 10):
            cur_date = date(year=year, month=1, day=1)
            res.append((cur_date - base_date).days * coef)

    elif period == '11_years':
        for year in range(1979, 2022, 11):
            cur_date = date(year=year, month=1, day=1)
            res.append((cur_date - base_date).days * coef)
    return res


def get_borders(l):
    res = []
    for i in range(len(l) - 1):
        res.append((l[i], l[i + 1]))
    return res


MONTHS = [
    'Январь',
    'Февраль',
    'Март',
    'Апрель',
    'Май',
    'Июнь',
    'Июль',
    'Август',
    'Сентябрь',
    'Октябрь',
    'Ноябрь',
    'Декабрь'
]

# эти функции используются только здесь
# month_borders - список таплов (start, end) границ (индексов массива) месяца для всего 42-летнего периода
# для получения месячных данных используются индексы как раз из month_borders
one_year_borders = get_borders(period_begginings(period='year', data_type='6'))
month_borders = get_borders(period_begginings(period='month', data_type='6'))


def is_pos_def(A):
    """
    Возвращает True если матрица положительно определена, False иначе
    """
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


def display_params(arr):
    """Корректное отображение вывода минимизации."""
    s_x = arr[4] * arr[4]
    s_y = arr[6] * arr[6]
    s_xy = arr[4] * arr[5] * arr[6]
    res = [
        [arr[0], arr[1]],
        np.array([arr[2], arr[3]]),
        np.array([[s_x, s_xy],
                  [s_xy, s_y]]),
        arr[7]]
    return res


def display_params_mixture(p, forlogs=False):
    s_x_1 = p[4] * p[4]
    s_y_1 = p[6] * p[6]
    s_xy_1 = p[4] * p[5] * p[6]
    s_x_2 = p[12] * p[12]
    s_y_2 = p[14] * p[14]
    s_xy_2 = p[12] * p[13] * p[14]
    if forlogs:
        return (
            "m1 =", np.array([p[0], p[1]]),  # m1
            "th1 =", np.array([p[2], p[3]]),  # theta1
            "\nS1 =", np.array([[s_x_1, s_xy_1],  # Sigma1
                                [s_xy_1, s_y_1]]),
            "\ntau1 =", p[7],  # tau1
            "\nm2 =", np.array([p[8], p[9]]),  # m2
            "th2 =", np.array([p[10], p[11]]),  # theta2
            "\nS2 =", np.array([[s_x_2, s_xy_2],  # Sigma2
                                [s_xy_2, s_y_2]]),
            "\ntau2 =", p[15],  # tau2
            "\nw1 =", p[16],
            "w2 =", 1 - p[16]  # weights
        )
    else:
        return (
            np.array([p[0], p[1]]),  # m1
            np.array([p[2], p[3]]),  # theta1
            np.array([[s_x_1, s_xy_1],  # Sigma1
                      [s_xy_1, s_y_1]]),
            p[7],  # tau1
            np.array([p[8], p[9]]),  # m2
            np.array([p[10], p[11]]),  # theta2
            np.array([[s_x_2, s_xy_2],  # Sigma2
                      [s_xy_2, s_y_2]]),
            p[15],  # tau2
            p[16], 1 - p[16]  # weights
        )
