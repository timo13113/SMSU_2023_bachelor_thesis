from util import *


def avg_2d_sample(mixture_mu, const_theta, normal_cov_sigma, gamma_shape_tau, sample_size=None, d=2):
    """
    Генерация сэмпла по параметрам
    """
    if not is_pos_def(normal_cov_sigma):
        normal_cov_sigma[0][1] *= 0.99
        normal_cov_sigma[1][0] *= 0.99
    #         normal_cov_sigma = nearPD(normal_cov_sigma)
    if is_pos_def(normal_cov_sigma):
        N = np.random.multivariate_normal(np.zeros(d), normal_cov_sigma, size=sample_size)
        W = np.random.gamma(gamma_shape_tau, 1, size=sample_size)
        buffer = np.hstack((N, np.sqrt(W)[:, None]))
        NsW = np.apply_along_axis(lambda x: x[-1] * x[:-1], 1, buffer)
        mW = np.vstack((mixture_mu[0] * W, mixture_mu[1] * W)).T
        Y = np.array([x + const_theta for x in NsW + mW])
        return Y
    else:
        raise ValueError('Sigma not positive-semidefinite')


def avg_2d_density(x, mixture_mu, const_theta, normal_cov_sigma, gamma_shape_tau, d=2):
    """
    Функция плотности (тета считается равным нулю)
    !иксы подавать по одному
    """
    if not is_pos_def((np.array(normal_cov_sigma, dtype=np.float64))):
        normal_cov_sigma[0, 1] *= 0.99
        normal_cov_sigma[1, 0] *= 0.99
    #         normal_cov_sigma = nearPD(normal_cov_sigma)
    if is_pos_def(np.array(normal_cov_sigma, dtype=np.float64)):
        x1 = x - const_theta
        try:
            SigmaInverse = normal_cov_sigma ** (-1)
        except:
            normal_cov_sigma[0, 1] *= 0.99
            normal_cov_sigma[1, 0] *= 0.99
            SigmaInverse = normal_cov_sigma ** (-1)
        SigmaDetRoot = sympy.sqrt(normal_cov_sigma.det())
        Q = sympy.sqrt((x1.T * SigmaInverse * x1)[0])
        C = sympy.sqrt((mixture_mu.T * SigmaInverse * mixture_mu)[0] + 2)
        a = 2 * sympy.exp((mixture_mu.T * SigmaInverse * x1)[0])
        K = sympy.functions.special.bessel.besselk(
            gamma_shape_tau - d / 2,
            Q * C)  # if sum([mpmath.isnormal(i) for i in x1]) else 0
        b = sympy.Pow(2 * np.pi, d / 2) * sympy.gamma(gamma_shape_tau) * SigmaDetRoot
        g = sympy.Pow(Q / C, gamma_shape_tau - d / 2)
        return (a / b) * g * K
    else:
        raise ValueError('Sigma not positive-semidefinite')


def make_empiric_chf_2d(s):
    """
    Возвращает эмп. хар. ф-ю, построенную по выборке.
    Возвращаемая функция может работать только с
    На текущий момент, возвращенная функция - самое тяжелое место программы.
    """

    def echf(ts):
        return np.average(np.exp(1j * (s @ ts.T).T), axis=-1)

    return echf


def avg_2d_chf(t, mixture_mu, const_theta, normal_cov_sigma, gamma_shape_tau):
    """Хар. функция AVG распределения"""
    return (1 / (1 + 1 / 2 * (t @ normal_cov_sigma @ t) - 1j * (mixture_mu @ t)
                 ) ** gamma_shape_tau) * np.exp(1j * (t @ const_theta))


def avg_2d_chf_vector(ts, mixture_mu, const_theta, normal_cov_sigma, gamma_shape_tau):
    """Векторная версия для обработки набора точек за раз"""
    return (1 / (1 + 1 / 2 * (np.sum(ts * (ts @ normal_cov_sigma), axis=1)) - 1j * (mixture_mu @ ts.T)
                 ) ** gamma_shape_tau) * np.exp(1j * (ts @ const_theta.T))


def fdiff_2d_precomputed(p, ts, precomputed_echf):
    """Считает разницу на сетке с данной сеткой и уже вычисленным ранее значением эмп. хар ф-и на этой сетке"""
    s_x = p[4] * p[4]
    s_y = p[6] * p[6]
    s_xy = p[4] * p[5] * p[6]
    f = avg_2d_chf_vector(ts,
                          np.array([p[0], p[1]]),  # m
                          np.array([p[2], p[3]]),  # theta
                          # np.array([[p[4], p[5]],  # Sigma
                          #           [p[5], p[6]]]),
                          np.array([[s_x, s_xy],  # Sigma
                                    [s_xy, s_y]]),
                          p[7]  # tau
                          )
    return np.linalg.norm(f - precomputed_echf)


def avg_2d_mixture_density(x,
                           mixture_mu_1, const_theta_1, normal_cov_sigma_1, gamma_shape_tau_1,
                           mixture_mu_2, const_theta_2, normal_cov_sigma_2, gamma_shape_tau_2,
                           weight_1, weight_2,
                           d=2):
    """Функция плотности !иксы подавать по одному"""

    np.testing.assert_almost_equal(weight_1 + weight_2, 1)  # check that weights add up to one

    if not is_pos_def((np.array(normal_cov_sigma_1, dtype=np.float64))):
        #         normal_cov_sigma_1 = nearPD(normal_cov_sigma_1)
        normal_cov_sigma_1[0, 1] *= 0.99
        normal_cov_sigma_1[1, 0] *= 0.99
    if not is_pos_def((np.array(normal_cov_sigma_2, dtype=np.float64))):
        #         normal_cov_sigma_2 = nearPD(normal_cov_sigma_2)
        normal_cov_sigma_2[0, 1] *= 0.99
        normal_cov_sigma_2[1, 0] *= 0.99
    if is_pos_def(np.array(normal_cov_sigma_1, dtype=np.float64)) and is_pos_def(
            np.array(normal_cov_sigma_2, dtype=np.float64)):
        x1 = x - const_theta_1
        x2 = x - const_theta_2

        if weight_1 < 0.01:
            ret1 = 0
            weight_2 = 1
        else:
            try:
                SigmaInverse = normal_cov_sigma_1 ** (-1)
            except:
                normal_cov_sigma_1[0, 1] *= 0.99
                normal_cov_sigma_1[1, 0] *= 0.99
                SigmaInverse = normal_cov_sigma_1 ** (-1)
            SigmaDetRoot = sympy.sqrt(normal_cov_sigma_1.det())
            Q = sympy.sqrt((x1.T * SigmaInverse * x1)[0])
            C = sympy.sqrt((mixture_mu_1.T * SigmaInverse * mixture_mu_1)[0] + 2)
            a = 2 * sympy.exp((mixture_mu_1.T * SigmaInverse * x1)[0])
            K = sympy.functions.special.bessel.besselk(
                gamma_shape_tau_1 - d / 2,
                Q * C)  # if sum([mpmath.isnormal(i) for i in x1]) else 0
            b = sympy.Pow(2 * np.pi, d / 2) * sympy.gamma(gamma_shape_tau_1) * SigmaDetRoot
            g = sympy.Pow(Q / C, gamma_shape_tau_1 - d / 2)
            ret1 = (a / b) * g * K

        if weight_2 < 0.01:
            ret2 = 0
            weight_1 = 1
        else:
            try:
                SigmaInverse = normal_cov_sigma_2 ** (-1)
            except:
                normal_cov_sigma_2[0, 1] *= 0.99
                normal_cov_sigma_2[1, 0] *= 0.99
                SigmaInverse = normal_cov_sigma_2 ** (-1)
            SigmaDetRoot = sympy.sqrt(normal_cov_sigma_2.det())
            Q = sympy.sqrt((x2.T * SigmaInverse * x2)[0])
            C = sympy.sqrt((mixture_mu_2.T * SigmaInverse * mixture_mu_2)[0] + 2)
            a = 2 * sympy.exp((mixture_mu_2.T * SigmaInverse * x2)[0])
            K = sympy.functions.special.bessel.besselk(
                gamma_shape_tau_2 - d / 2,
                Q * C)  # if sum([mpmath.isnormal(i) for i in x]) else 0
            b = sympy.Pow(2 * np.pi, d / 2) * sympy.gamma(gamma_shape_tau_2) * SigmaDetRoot
            g = sympy.Pow(Q / C, gamma_shape_tau_2 - d / 2)
            ret2 = (a / b) * g * K

        return ret1 * weight_1 + ret2 * weight_2
    else:
        raise ValueError('Sigma not positive-semidefinite')


def avg_2d_mixture_sample(mixture_mu_1, const_theta_1, normal_cov_sigma_1, gamma_shape_tau_1,
                          mixture_mu_2, const_theta_2, normal_cov_sigma_2, gamma_shape_tau_2,
                          weight_1, weight_2,
                          sample_size=None, d=2):
    """Генерация сэмпла по параметрам"""
    np.testing.assert_almost_equal(weight_1 + weight_2, 1)  # check that weights add up to one
    #     print(is_pos_def(normal_cov_sigma_1), is_pos_def(normal_cov_sigma_2))
    if not is_pos_def(normal_cov_sigma_1):
        #         normal_cov_sigma_1 = nearPD(normal_cov_sigma_1)
        normal_cov_sigma_1[0, 1] *= 0.99
        normal_cov_sigma_1[1, 0] *= 0.99
    if not is_pos_def(normal_cov_sigma_2):
        #         normal_cov_sigma_2 = nearPD(normal_cov_sigma_2)
        normal_cov_sigma_2[0, 1] *= 0.99
        normal_cov_sigma_2[1, 0] *= 0.99
    if is_pos_def(normal_cov_sigma_1) and is_pos_def(normal_cov_sigma_2):
        N = np.random.multivariate_normal(np.zeros(d), normal_cov_sigma_1, size=sample_size)
        W = np.random.gamma(gamma_shape_tau_1, 1, size=sample_size)
        buffer = np.hstack((N, np.sqrt(W)[:, None]))
        NsW = np.apply_along_axis(lambda x: x[-1] * x[:-1], 1, buffer)
        mW = np.vstack((mixture_mu_1[0] * W, mixture_mu_1[1] * W)).T
        Y1 = np.array([x + const_theta_1 for x in NsW + mW])
        N = np.random.multivariate_normal(np.zeros(d), normal_cov_sigma_2, size=sample_size)
        W = np.random.gamma(gamma_shape_tau_2, 1, size=sample_size)
        buffer = np.hstack((N, np.sqrt(W)[:, None]))
        NsW = np.apply_along_axis(lambda x: x[-1] * x[:-1], 1, buffer)
        mW = np.vstack((mixture_mu_2[0] * W, mixture_mu_2[1] * W)).T
        Y2 = np.array([x + const_theta_2 for x in NsW + mW])
        if sample_size is None:
            a = np.random.uniform()
            if a < weight_1:
                return Y1
            else:
                return Y2
        else:
            mask = np.random.choice([True, False], size=sample_size, p=[weight_1, weight_2])
            return np.vstack([Y1[mask], Y2[~mask]])
    else:
        print(is_pos_def(normal_cov_sigma_1), is_pos_def(normal_cov_sigma_2))
        print(normal_cov_sigma_1)
        print(normal_cov_sigma_2)
        raise ValueError('One or more sigma is not positive-semidefinite')


def avg_2d_mixture_chf_vector(ts,
                              mixture_mu_1, const_theta_1, normal_cov_sigma_1, gamma_shape_tau_1,
                              mixture_mu_2, const_theta_2, normal_cov_sigma_2, gamma_shape_tau_2,
                              weight_1, weight_2):
    """Векторная версия хар. функции распределения смеси 2 распределений AVG для обработки набора точек за раз"""
    ts1 = weight_1 * ts
    ts2 = weight_2 * ts
    first = (1 / (1 + 1 / 2 * (np.sum(ts1 * (ts1 @ normal_cov_sigma_1), axis=1))
                  - 1j * (mixture_mu_1 @ ts1.T)) ** gamma_shape_tau_1) * np.exp(1j * (ts1 @ const_theta_1.T))
    second = (1 / (1 + 1 / 2 * (np.sum(ts2 * (ts2 @ normal_cov_sigma_2), axis=1))
                   - 1j * (mixture_mu_2 @ ts2.T)) ** gamma_shape_tau_2) * np.exp(1j * (ts2 @ const_theta_2.T))
    return first * second


def fdiff_2d_mixture_precomputed(p, ts, precomputed_echf):
    """Считает разницу на сетке с данной сеткой и уже вычисленным ранее значением эмп. хар ф-и на этой сетке"""
    s_x_1 = p[4] * p[4]
    s_y_1 = p[6] * p[6]
    s_xy_1 = p[4] * p[5] * p[6]
    s_x_2 = p[12] * p[12]
    s_y_2 = p[14] * p[14]
    s_xy_2 = p[12] * p[13] * p[14]
    f = avg_2d_mixture_chf_vector(ts,
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
    return np.linalg.norm(f - precomputed_echf)
