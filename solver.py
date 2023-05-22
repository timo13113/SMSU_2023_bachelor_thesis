from generation import *
from kstest2d import *
from nets import *
from draw import *
from em_gmm import *


def find_parameters(data, plot=False, logs=False, method='L-BFGS-B', s="", return_minimised_func=False):
    """
    Собственно функция, которая находит параметры, получая на вход данные.
    Возвращает (res, test, time)
    res - массив из 6 оптимальных параметров (или массив None, если минимизация не сошлась),
    test - результат ks-теста на однородность между данными, полученными на вход, и сэмплом, сгенерированным по параметрам
    time - время, потраченное на минимизацию.
    """
    start = time.time()
    ### построение всего нужного #########################################################
    echf = make_empiric_chf_2d(data)
    t = make_2d_net(echf, logs=logs)  #
    precomputed = echf(t)
    if plot:
        draw_net(t, s=s)
        draw_echf(echf, s=s)
    ### попытка апроксимировать параметры по статистикам #################################
    x, y = data[:, 0], data[:, 1]
    mean = np.mean(data, axis=0)  # среднее
    Sigma_ = np.cov(x / mean[0], y / mean[1])  # ковариационная матрица
    initial_guess = [
        mean[0], mean[1],
        0, 0,
        Sigma_[0, 0] ** (1 / 2), Sigma_[0, 1] / (Sigma_[0, 0] ** (1 / 2) * Sigma_[1, 1] ** (1 / 2)),
        Sigma_[1, 1] ** (1 / 2),
        1
    ]  # нулевое приближение
    if logs:
        print('Нулевое приближение:\n', display_params(initial_guess))
    ### непосредственно минимизация ######################################################
    bnds = (
        (None, None),  # 0
        (None, None),  # 1
        (None, None),  # 2
        (None, None),  # 3
        (0.1, None),  # 4
        (-0.99, 0.99),  # 5
        (0.1, None),  # 6
        # (0.1, None),  # 7
        (0.1, 10),  # 7
    )

    start2 = time.time()
    ans = minimize(fdiff_2d_precomputed, np.array(initial_guess),
                   method=method,
                   args=(t, precomputed),
                   # constraints=cons,
                   bounds=bnds,
                   options={
                       # 'ftol': 0.000001,
                       'maxiter': 500}
                   )
    end_min = -start2 + time.time()
    end_all = -start + time.time()
    if logs:
        print("Время на минимизацию:", end_min)
        print("Время на весь пайплайн:", end_all)
        print("Результат:\n", ans)
    res = display_params(ans.x)
    ### вывод графика исходного и сгенерированного сэмпла ################################
    if ans.success:
        test_res = []
        for i in range(5):
            gen = avg_2d_sample(res[0], res[1], res[2], res[3], SAMPLE_SIZE)
            test_res.append(ks2d2s(data[:, 0], data[:, 1], gen[:, 0], gen[:, 1]))
        if logs:
            print("p-значение теста:", sum(test_res) / 5)
        if plot:
            gen = avg_2d_sample(res[0], res[1], res[2], res[3], SAMPLE_SIZE)
            draw_sample_with_hyp(data, gen, s=s)
            draw_hyp_chf_1(res, s=s)
        if return_minimised_func:
            return ans.x, sum(test_res) / 5, end_all, t, precomputed, echf
        return ans.x, sum(test_res) / 5, end_all
    else:
        return [None, None, None, None, None, None, None, None], None, end_all


def find_parameters_mixture(data, plot=False, logs=False, x0=None, method='L-BFGS-B', s=""):
    """
    Собственно функция, которая находит параметры, получая на вход данные.
    Возвращает (res, test, time), где:
    res - массив из 6 оптимальных параметров (или массив None, если минимизация не сошлась),
    test - результат ks-теста на однородность между данными, полученными на вход, и сэмплом, сгенерированным по параметрам
    time - время, потраченное на минимизацию.
    """
    start = time.time()
    ### построение всего нужного #########################################################
    echf = make_empiric_chf_2d(data)
    t = make_2d_net(echf, clip=False)  #
    precomputed = echf(t)
    if plot:
        draw_net(t, s=s)
        draw_echf(echf, s=s)
    ### попытка апроксимировать параметры по статистикам #################################
    x, y = data[:, 0], data[:, 1]
    mean = np.mean(data, axis=0)  # среднее
    theta_1_1, theta_2_1 = 0, np.percentile(x, 80, axis=0)
    theta_1_2, theta_2_2 = 0, np.percentile(y, 80, axis=0)
    Sigma_ = np.cov(x, y)  # ковариационная матрица
    Sigma_[0, 1] = Sigma_[0, 1] / (Sigma_[0, 0] * Sigma_[1, 1])
    initial_guess = [
        mean[0], mean[1],
        theta_1_1, theta_1_2,
        Sigma_[0, 0], Sigma_[0, 1], Sigma_[1, 1],
        1,
        1, 1,
        theta_2_1, theta_2_2,
        1, 0, 1,
        1,
        0.7  # weights
    ]  # нулевое приближение
    if x0 is not None:
        for i, x0_ in enumerate(x0):
            if x0_ is not None:
                initial_guess[i] = x0_
    if logs:
        print('Нулевое приближение:\n', *display_params_mixture(initial_guess, logs))
    ### непосредственно минимизация ######################################################
    bnds = (
        (None, None),  # 0
        (None, None),  # 1
        (None, None),  # 2
        (None, None),  # 3
        (0.1, None),  # 4
        (-0.99, 0.99),  # 5
        (0.1, None),  # 6
        # (0.1, None),  # 7
        (0.1, 10),  # 7
        (None, None),  # 8
        (None, None),  # 9
        (None, None),  # 10
        (None, None),  # 11
        (0.1, None),  # 12
        (-0.99, 0.99),  # 13
        (0.1, None),  # 14
        # (0.1, None),  # 15
        (0.1, 10),  # 7
        (0, 1),  # 16
    )
    start2 = time.time()
    ans = minimize(fdiff_2d_mixture_precomputed, np.array(initial_guess),
                   method=method,
                   args=(t, precomputed),
                   bounds=bnds,
                   options={
                       #                        'maxfun': 25000,
                       #                        'ftol': 0.000001,
                       #                        'xtol': 0.000001,
                       'maxiter': 5000}
                   )
    end_min = -start2 + time.time()
    end_all = -start + time.time()
    res = display_params_mixture(ans.x)
    if logs:
        print(res)
        print("Результат:\n", *display_params_mixture(ans.x, logs))
        print("Время на минимизацию:", end_min)
        print("Время на весь пайплайн:", end_all)
    ### вывод графика исходного и сгенерированного сэмпла ################################
    if ans.success:
        test_res = []
        for i in range(5):
            gen = avg_2d_mixture_sample(*res, SAMPLE_SIZE)
            test_res.append(ks2d2s(data[:, 0], data[:, 1], gen[:, 0], gen[:, 1]))
        if logs:
            print("p-значение теста:", sum(test_res) / 5)
        if plot:
            gen = avg_2d_mixture_sample(*res, SAMPLE_SIZE)
            draw_sample_with_hyp(data, gen, s=s)
            draw_hyp_chf_2(res, s=s)
        return ans.x, sum(test_res) / 5, end_all
    else:
        return [None, None, None, None, None, None,
                None, None, None, None, None, None,
                None, None, None, None, None], None, end_all


def month_distribution_AVG_mixture(data, years, label, plot=False, plot_verbose=False, method=None, dirname=None,
                                   mix=True):
    """
    Визуализирует помесячное совместное распределение явного и скрытого потока (см. data с shape == (n_measurments, 2))
    для годов, заданных в списке years
    """
    xmin, xmax, ymin, ymax = -100, 400, -100, 1200  # пределы для рисования контура
    for year in years:
        start, end = one_year_borders[year - 1979]
        x = data[start:end]

        fig, axes = plt.subplots(figsize=(20, 12), nrows=3, ncols=4)
        fig.suptitle(str(year) + f", {label} mixture AVG net=({NUM}, {RAYS})", fontsize=25, y=0.93)

        for j in range(0, 12):

            s = f"_{year}_{j + 1}_{label}"

            axes[j // 4, j % 4].scatter(x[j * 120: (j + 1) * 120, 0], x[j * 120: (j + 1) * 120, 1], s=10)
            ################
            output1 = find_parameters(x[j * 120: (j + 1) * 120], logs=True, plot=plot_verbose,
                                      s=s)  # минимизация, первый шаг
            # print("промежуточные параметры:", display_params(output1[0]) if output1[1] is not None else output1[0])
            # print("p:", output1[1])
            print(output1[2])
            if output1[1] is not None and output1[1] > 0.05:
                mix = False
            else:
                mix = True

            if mix:
                output2 = find_parameters_mixture(x[j * 120: (j + 1) * 120],
                                                  method=method, logs=True, plot=plot_verbose, s=s,
                                                  x0=[output1[0][0], output1[0][1],
                                                      output1[0][2], output1[0][3],
                                                      output1[0][4],
                                                      output1[0][5],
                                                      output1[0][6],
                                                      output1[0][7],
                                                      None, None,
                                                      None, None,
                                                      None, None, None,
                                                      None,
                                                      0.5,
                                                      ]
                                                  )  # минимизация, второй шаг
                # print("финальные параметры:",
                # display_params_mixture(output2[0]) if output2[1] is not None else output2[0])
                # print("p:", output2[1])
                print(output2[2])
                if output1[1] is not None and output1[1] > 0.05 and output2[1] is not None and output2[1] > 0.05:
                    output = output1 if output1[1] > output2[1] else output2
                elif output1[1] is not None and output1[1] > 0.05:
                    output = output1
                elif output2[1] is not None and output2[1] > 0.05:
                    output = output2
                else:
                    output = output2
            else:
                output = output1
            xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]  # сетка
            positions = np.vstack([xx.ravel(), yy.ravel()])
            cntr = []
            pars = []
            if output[1] is None:
                print("Не сошлось")
            else:
                # print(int(output2[1] == output[1])+1, "компонент подошли лучше")
                if len(output[0]) == 8:
                    for i in display_params(output[0]):
                        if isinstance(i, int) or isinstance(i, float):
                            pars.append(i)
                        else:
                            pars.append(sympy.Matrix(i))
                else:
                    for i in display_params_mixture(output[0]):
                        if isinstance(i, int) or isinstance(i, float):
                            pars.append(i)
                        else:
                            pars.append(sympy.Matrix(i))
                if len(pars) == 4:
                    for i in tqdm.tqdm(positions.T):
                        f = avg_2d_density(sympy.Matrix(i), *pars)  # расчет плотности
                        cntr.append(float(f))
                else:
                    for i in tqdm.tqdm(positions.T):
                        f = avg_2d_mixture_density(sympy.Matrix(i), *pars)
                        cntr.append(float(f))
                #                     axes[j // 4, j % 4].contourf(
                axes[j // 4, j % 4].contour(
                    xx, yy, np.array(cntr).reshape(xx.shape),
                    np.logspace(-20, 0, 10),
                    norm=matplotlib.colors.LogNorm(vmin=1e-20, vmax=1),
                    #                         zorder=-1
                )
                if plot_verbose:
                    draw_hyp_density(
                        # np.linspace(xmin, xmax, 100),
                        # np.linspace(ymin, ymax, 100),
                        xx, yy,
                        np.array(cntr).reshape(100, 100),
                        s=s
                    )
            ################
            if mix:
                p = ", p={:.4f}, {} комп.".format(output[1],
                                                  int(output2[1] == output[1]) + 1
                                                  ) if output[1] is not None else " (нет сходимости)"
            else:
                p = ", p={:.4f}, 1 комп.".format(output[1]) if output[1] is not None else " (нет сходимости)"
            axes[j // 4, j % 4].set(title=MONTHS[j] + p, xlabel='sensible', ylabel='latent')
            axes[j // 4, j % 4].set_xlim((xmin, xmax))
            axes[j // 4, j % 4].set_ylim((ymin, ymax))
            axes[j // 4, j % 4].grid()
            axes[j // 4, j % 4].label_outer()
        #                 axes[j // 4, j % 4].set_facecolor('#46085c')

        if dirname is not None:
            filename = os.path.join(dirname, str(year) + f", {label} mixture AVG net=({NUM}, {RAYS})")
            plt.savefig(filename, dpi=300)
            plt.close()
            print('Figure is saved in ' + filename)
        else:
            plt.show()


def bimonth_distribution_AVG_mixture(data, years, label, method=None, dirname=None):
    """
    Визуализирует помесячное совместное распределение явного и скрытого потока (см. data с shape == (n_measurments, 2))
    для годов, заданных в списке years
    """
    if len(data.shape) == 2:
        for year in years:
            start, end = one_year_borders[year - 1979]
            x = data[start:end]
            indexes = np.arange(0, x.shape[0])

            fig, axes = plt.subplots(figsize=(20, 12), nrows=2, ncols=3)
            fig.suptitle(str(year) + f", {label} bimonth mixture AVG net=({NUM}, {RAYS})", fontsize=25, y=0.93)

            for j in range(0, 6):
                axes[j // 3, j % 3].scatter(x[j * 240: (j + 1) * 240, 0], x[j * 240: (j + 1) * 240, 1], s=10)
                ################
                xmin, xmax, ymin, ymax = -100, 400, -100, 1200  # пределы для рисования контура
                xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]  # сетка
                positions = np.vstack([xx.ravel(), yy.ravel()])
                cntr = []
                output1 = find_parameters(x[j * 240: (j + 1) * 240])  # минимизация, первый шаг
                print("промежуточные параметры:",
                      display_params(output1[0]) if output1[1] is not None else output1[0])
                print("p:", output1[1])
                print(output1[2])
                output2 = find_parameters_mixture(x[j * 240: (j + 1) * 240], method=method,  # , logs = True,
                                                  x0=[output1[0][0], output1[0][1],
                                                      output1[0][2], output1[0][3],
                                                      output1[0][4],
                                                      output1[0][5],
                                                      output1[0][6],
                                                      output1[0][7],
                                                      None, None,
                                                      None, None,
                                                      None, None, None,
                                                      None,
                                                      0.7,
                                                      ]
                                                  )  # минимизация, второй шаг
                print("финальные параметры:",
                      display_params_mixture(output2[0]) if output2[1] is not None else output2[0])
                print("p:", output2[1])
                print(output2[2])
                # if output1[1] is not None and output1[1] > 0.05 and output2[1] is not None and output2[1] > 0.05:
                # output = output1 if output1[1] > output2[1] else output2
                if output1[1] is not None and output1[1] > 0.05:
                    output = output1
                elif output2[1] is not None and output2[1] > 0.05:
                    output = output2
                else:
                    output = output2
                # output = output2
                # print("###############\n", output[0][5], "#############")
                pars = []
                if output[1] is None:
                    print("Не сошлось")
                else:
                    print(int(output2[1] == output[1]) + 1, "компонент подошли лучше")
                    if len(output[0]) == 8:
                        for i in display_params(output[0]):
                            if isinstance(i, int) or isinstance(i, float):
                                pars.append(i)
                            else:
                                pars.append(sympy.Matrix(i))
                    else:
                        for i in display_params_mixture(output[0]):
                            if isinstance(i, int) or isinstance(i, float):
                                pars.append(i)
                            else:
                                pars.append(sympy.Matrix(i))
                    if len(pars) == 4:
                        for i in tqdm.tqdm(positions.T):
                            f = avg_2d_density(sympy.Matrix(i), *pars)  # расчет плотности
                            cntr.append(f)
                    else:
                        for i in tqdm.tqdm(positions.T):
                            f = avg_2d_mixture_density(sympy.Matrix(i), *pars)
                            cntr.append(f)
                    axes[j // 3, j % 3].contour(
                        xx, yy, np.array(cntr).reshape(xx.shape),
                        np.logspace(-20, 0, 10),
                        norm=matplotlib.colors.LogNorm(vmin=1e-20, vmax=max(cntr)),
                        zorder=-1)
                ################
                p = ", p={:.4f}, {} комп.".format(output[1],
                                                  int(output2[1] == output[1]) + 1
                                                  ) if output[1] is not None else " (нет сходимости)"
                axes[j // 3, j % 3].set(title=MONTHS[2 * j] + ", " + MONTHS[2 * j + 1] + p, xlabel='sensible',
                                        ylabel='latent')
                axes[j // 3, j % 3].set_xlim((xmin, xmax))
                axes[j // 3, j % 3].set_ylim((ymin, ymax))
                axes[j // 3, j % 3].grid()
                axes[j // 3, j % 3].label_outer()
                # axes[j // 3, j % 3].set_facecolor('#46085c')
            if dirname is not None:
                filename = os.path.join(dirname, str(year) + f", {label} bimonth mixture AVG net=({NUM}, {RAYS})")
                plt.savefig(filename, dpi=300)
                plt.close()
                print('Figure is saved in ' + filename)
            else:
                plt.show()


def five_month_distribution_AVG_mixture(month_data, slices, label, method=None, dirname=None):
    lat, long = label
    for years, limits in enumerate(slices):
        fig, axes = plt.subplots(figsize=(20, 12), nrows=3, ncols=4)
        fig.suptitle('Средний климатический год: ' + str(
            (lat, long)) + f", годы с {years * 5 + 1979} по {(years + 1) * 5 + 1978} AVG",
                     fontsize=25, y=0.95)
        for j in range(0, 12):
            axes[j // 4, j % 4].scatter(month_data[j][limits[0]:limits[1], 0],
                                        month_data[j][limits[0]:limits[1], 1], s=10)

            xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]  # сетка
            positions = np.vstack([xx.ravel(), yy.ravel()])
            cntr = []
            output1 = find_parameters(month_data[j][limits[0]:limits[1]])  # минимизация, первый шаг
            # print("промежуточные параметры:", display_params(output1[0]) if output1[1] is not None else output1[0])
            # print("p:", output1[1])
            print(output1[2])
            output2 = find_parameters_mixture(month_data[j][limits[0]:limits[1]],
                                              method=method,
                                              logs=True,
                                              x0=[output1[0][0], output1[0][1],
                                                  output1[0][2], output1[0][3],
                                                  output1[0][4],
                                                  output1[0][5],
                                                  output1[0][6],
                                                  output1[0][7],
                                                  None, None,
                                                  None, None,
                                                  None, None, None,
                                                  None,
                                                  0.7,
                                                  ]
                                              )  # минимизация, второй шаг
            # print("финальные параметры:", display_params_mixture(output2[0]) if output2[1] is not None else output2[0])
            # print("p:", output2[1])
            print(output2[2])
            if output1[1] is not None and output1[1] > 0.05 and output2[1] is not None and output2[1] > 0.05:
                output = output1 if output1[1] > output2[1] else output2
            elif output1[1] is not None and output1[1] > 0.05:
                output = output1
            elif output2[1] is not None and output2[1] > 0.05:
                output = output2
            else:
                output = output2
            pars = []
            if output[1] is None:
                print("Не сошлось")
            else:
                print(int(output2[1] == output[1]) + 1, "компонент подошли лучше")
                if len(output[0]) == 8:
                    for i in display_params(output[0]):
                        if isinstance(i, int) or isinstance(i, float):
                            pars.append(i)
                        else:
                            pars.append(sympy.Matrix(i))
                else:
                    for i in display_params_mixture(output[0]):
                        if isinstance(i, int) or isinstance(i, float):
                            pars.append(i)
                        else:
                            pars.append(sympy.Matrix(i))
                if len(pars) == 4:
                    for i in tqdm.tqdm(positions.T):
                        f = avg_2d_density(sympy.Matrix(i), *pars)  # расчет плотности
                        cntr.append(f)
                else:
                    for i in tqdm.tqdm(positions.T):
                        f = avg_2d_mixture_density(sympy.Matrix(i), *pars)
                        cntr.append(f)
                axes[j // 4, j % 4].contour(
                    xx, yy, np.array(cntr).reshape(xx.shape),
                    np.logspace(-20, 0, 10),
                    norm=matplotlib.colors.LogNorm(vmin=1e-20, vmax=1),
                    #                 zorder=-1
                )

            p = ", p={:.4f}, {} комп.".format(output[1],
                                              int(output2[1] == output[1]) + 1
                                              ) if output[1] is not None else " (нет сходимости)"
            axes[j // 4, j % 4].set(title=MONTHS[j] + p, xlabel='sensible', ylabel='latent')
            axes[j // 4, j % 4].set_xlim((xmin, xmax))
            axes[j // 4, j % 4].set_ylim((ymin, ymax))
            axes[j // 4, j % 4].grid()
            axes[j // 4, j % 4].label_outer()
            # axes[j // 4, j % 4].set_facecolor('#46085c')

        if dirname is not None:
            filename = os.path.join(dirname, str(years * 5 + 1979) + f", {(lat, long)} среднеклиматич 5 лет AVG")
            plt.savefig(filename, dpi=300)
            plt.close()
            print('Figure is saved in ' + filename)
        else:
            plt.show()


def month_distribution_nodraw(data, years, method=None):
    """
    считает помесячное совместное распределение явного и скрытого потока (см. data с shape == (n_measurments, 2))
    для годов, заданных в списке years
    """
    out_avg = []
    out_gmm = []
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]  # сетка
    pos = np.dstack((xx, yy))
    for year in years:
        start, end = one_year_borders[year - 1979]
        x = data[start:end]
        for j in range(0, 12):
            ########################################################
            output1 = find_parameters(x[j * 120: (j + 1) * 120])  # минимизация, первый шаг
            #  print(output1[2])
            if output1[1] is None or output1[1] <= 0.05:
                output2 = find_parameters_mixture(x[j * 120: (j + 1) * 120],
                                                  # logs = True,
                                                  x0=[output1[0][0], output1[0][1],
                                                      output1[0][2], output1[0][3],
                                                      output1[0][4],
                                                      output1[0][5],
                                                      output1[0][6],
                                                      output1[0][7],
                                                      None, None,
                                                      None, None,
                                                      None, None, None,
                                                      None,
                                                      0.5,
                                                      ]
                                                  )  # минимизация, второй шаг
                #                 print(output2[2])
                if output1[1] is not None and output1[1] > 0.05:
                    output = output1
                elif output2[1] is not None and output2[1] > 0.05:
                    output = output2
                else:
                    output = output2
            else:
                output = output1
            if output[1] is not None and output[1] > 0.05:
                out_avg.append(True)
            else:
                out_avg.append(False)
            ########################################################
            gm = GaussianMixture(n_components=2)
            gm.fit(x[j * 120: (j + 1) * 120])

            Z = None
            components = []
            for i in range(2):
                zi = multivariate_normal(gm.means_[i], gm.covariances_[i])
                if Z is not None:
                    Z += zi.pdf(pos) * gm.weights_[i]
                else:
                    Z = zi.pdf(pos) * gm.weights_[i]
                components.append(zi)

            test_res = []
            for i in range(5):
                # генератор семплов
                gen = []
                for _ in range(SAMPLE_SIZE):
                    c = np.random.uniform(0, 1)
                    n = np.searchsorted(np.r_[0, gm.weights_].cumsum(), c) - 1
                    gen.append(components[n].rvs(1))
                # gen = np.vstack([z1.rvs(c), z2.rvs(SAMPLE_SIZE - c)])
                # print(len(gen), len(gen[0]))
                test_res.append(ks2d2s(
                    x[j * 120: (j + 1) * 120, 0],
                    x[j * 120: (j + 1) * 120, 1],
                    np.array(gen)[:, 0],
                    np.array(gen)[:, 1]))
            test = sum(test_res) / 5
            if test > 0.05:
                out_gmm.append(True)
            else:
                out_gmm.append(False)
        ########################################################

        # print(year, "просчитан")
    return out_avg, out_gmm


def bimonth_distribution_nodraw(data, years):
    """
    считает помесячное совместное распределение явного и скрытого потока (см. data с shape == (n_measurments, 2))
    для годов, заданных в списке years
    """
    out_avg = []
    out_gmm = []
    for year in years:
        start, end = one_year_borders[year - 1979]
        x = data[start:end]
        for j in range(0, 6):
            ########################################################
            output1 = find_parameters(x[j * 240: (j + 1) * 240])  # минимизация, первый шаг
            #             print(output1[2])
            if output1[1] is None or output1[1] <= 0.05:
                output2 = find_parameters_mixture(x[j * 240: (j + 1) * 240],
                                                  # , logs = True,
                                                  x0=[output1[0][0], output1[0][1],
                                                      output1[0][2], output1[0][3],
                                                      output1[0][4],
                                                      output1[0][5],
                                                      output1[0][6],
                                                      output1[0][7],
                                                      None, None,
                                                      None, None,
                                                      None, None, None,
                                                      None,
                                                      0.99,
                                                      ]
                                                  )  # минимизация, второй шаг
                #                 print(output2[2])
                if output1[1] is not None and output1[1] > 0.05:
                    output = output1
                elif output2[1] is not None and output2[1] > 0.05:
                    output = output2
                else:
                    output = output2
            else:
                output = output1
            if output[1] is not None and output[1] > 0.05:
                out_avg.append(True)
            else:
                out_avg.append(False)
            ########################################################
            gm = GaussianMixture(n_components=2)
            gm.fit(x[j * 240: (j + 1) * 240])

            components = []
            for i in range(2):
                zi = multivariate_normal(gm.means_[i], gm.covariances_[i])
                components.append(zi)

            test_res = []
            for i in range(5):
                # генератор семплов
                gen = []
                for _ in range(SAMPLE_SIZE):
                    c = np.random.uniform(0, 1)
                    n = np.searchsorted(np.r_[0, gm.weights_].cumsum(), c) - 1
                    gen.append(components[n].rvs(1))
                #                     gen = np.vstack([z1.rvs(c), z2.rvs(SAMPLE_SIZE - c)])
                #                 print(len(gen), len(gen[0]))
                test_res.append(ks2d2s(
                    x[j * 240: (j + 1) * 240, 0],
                    x[j * 240: (j + 1) * 240, 1],
                    np.array(gen)[:, 0],
                    np.array(gen)[:, 1]))
            test = sum(test_res) / 5
            if test > 0.05:
                out_gmm.append(True)
            else:
                out_gmm.append(False)
        ########################################################

        # print(year, "просчитан")
    return out_avg, out_gmm


def fiveyear_month_distribution_nodraw(month_data):
    """
    считает помесячное совместное распределение явного и скрытого потока (см. data с shape == (n_measurments, 2))
    для годов, заданных в списке years
    """
    slices = [
        (0, 620),
        (620, 1240),
        (1240, 1860),
        (1860, 2480),
        (2480, 3100),
        (3100, 3720),
        (3720, 4340),
        (4340, 4960)
    ]
    out_avg = []
    out_gmm = []
    for years, limits in enumerate(slices):
        for j in range(0, 12):
            ########################################################
            output1 = find_parameters(month_data[j][limits[0]:limits[1]])  # минимизация, первый шаг
            #             print(output1[2])
            if output1[1] is None or output1[1] <= 0.05:
                output2 = find_parameters_mixture(month_data[j][limits[0]:limits[1]],
                                                  # logs = True,
                                                  x0=[output1[0][0], output1[0][1],
                                                      output1[0][2], output1[0][3],
                                                      output1[0][4],
                                                      output1[0][5],
                                                      output1[0][6],
                                                      output1[0][7],
                                                      None, None,
                                                      None, None,
                                                      None, None, None,
                                                      None,
                                                      0.5,
                                                      ]
                                                  )  # минимизация, второй шаг
                #                 print(output2[2])
                if output1[1] is not None and output1[1] > 0.05:
                    output = output1
                elif output2[1] is not None and output2[1] > 0.05:
                    output = output2
                else:
                    output = output2
            else:
                output = output1
            if output[1] is not None and output[1] > 0.05:
                out_avg.append(True)
            else:
                out_avg.append(False)
            ########################################################
            gm = GaussianMixture(n_components=2)
            output = gm.fit(month_data[j][limits[0]:limits[1]])
            #             print(gm.covariances_, gm.means_, gm.weights_)
            ################
            components = []
            for i in range(2):
                zi = multivariate_normal(gm.means_[i], gm.covariances_[i])
                components.append(zi)

            test_res = []
            for i in range(5):
                # генератор семплов
                gen = []
                for _ in range(SAMPLE_SIZE):
                    c = np.random.uniform(0, 1)
                    n = np.searchsorted(np.r_[0, gm.weights_].cumsum(), c) - 1
                    gen.append(components[n].rvs(1))
                #                     gen = np.vstack([z1.rvs(c), z2.rvs(SAMPLE_SIZE - c)])
                #                 print(len(gen), len(gen[0]))
                test_res.append(ks2d2s(
                    month_data[j][limits[0]:limits[1], 0],
                    month_data[j][limits[0]:limits[1], 1],
                    np.array(gen)[:, 0],
                    np.array(gen)[:, 1]))
            test = sum(test_res) / 5
            if test > 0.05:
                out_gmm.append(True)
            else:
                out_gmm.append(False)

        # print(years, "просчитан")
    return out_avg, out_gmm
