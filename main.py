from lib import *
from solver import *
from em_gmm import *


def main():
    lat_long_list = [
        (30, -75),
        (30, -45),
        (45, -45),
        (60, -30)
    ]

    data_dict = dict()

    for lat, long in lat_long_list:
        data_dict[(lat, long)] = np.load(f'time_series/sens_lat_{(lat, long)}.npy')

    month_data = dict([(i, []) for i in range(0, 12)])

    for i, borders in enumerate(month_borders):
        start, end = borders
        month_data[i % 12].append(data_dict[(60, -30)][start:end])

    for i in range(0, 12):
        month_data[i] = np.vstack(month_data[i])

    print(month_data[0].shape)

    # print(data_dict[(30, -75)].shape)

    # month_distribution_AVG_mixture(
    #     data_dict[(30, -75)],
    #     # [1984],
    #     [1979, 1989, 1999, 2009, 2019],
    #     # [1979, 1984, 1989, 1994, 1999, 2004, 2009, 2014, 2019],
    #     (30, -75),
    #     dirname="imgnew",
    #     mix=True,
    #     plot=True
    # )
    #
    # bimonth_distribution_AVG_mixture(
    #     data_dict[(30, -75)],
    #     # [1984],
    #     [1979, 1989, 1999, 2009, 2019],
    #     # [1979, 1984, 1989, 1994, 1999, 2004, 2009, 2014, 2019],
    #     (30, -75),
    #     dirname="imgnew",
    #     # mix=True,
    #     # plot=True
    # )
    # gaussian_em(
    #         data_dict[(30, -75)],
    #         # [1984],
    #         [1979, 1989, 1999, 2009, 2019],
    #         # [1979, 1984, 1989, 1994, 1999, 2004, 2009, 2014, 2019],
    #         (30, -75),
    #         dirname="imgnew",
    #         plot=True
    #     )
    # gaussian_em_bimonth(
    #     data_dict[(30, -75)],
    #     # [1984],
    #     [1979, 1989, 1999, 2009, 2019],
    #     # [1979, 1984, 1989, 1994, 1999, 2004, 2009, 2014, 2019],
    #     (30, -75),
    #     dirname="imgnew",
    #     # plot=True
    # )

    slices = [(0, 620), (620, 1240), (1240, 1860), (1860, 2480)]  # слайсы по году

    # five_month_distribution_AVG_mixture(
    #     month_data,
    #     # [1984],
    #     # [1979, 1989, 1999, 2009, 2019],
    #     slices,
    #     (30, -75),
    #     dirname="imgnew",
    #     # plot=True
    # )
    # gaussian_em_fivemonth(
    #     month_data,
    #     # [1984],
    #     # [1979, 1989, 1999, 2009, 2019],
    #     slices,
    #     (30, -75),
    #     dirname="imgnew",
    #     # plot=True
    # )

    # прогнать на январе 1979
    # start, end = one_year_borders[1979 - 1979]
    # x = data_dict[(30, -75)][start:end]
    #

    # for j in range(12):
    #     # j = 0  # январь
    #     s = f"_{1979}_{j+1}_{(30, -75)}"
    #     pars, _, _, net, netvals, echf = find_parameters(
    #         x[j * 120: (j + 1) * 120], logs=True,
    #         plot=False, s=s, return_minimised_func=True)  # минимизация, первый шаг
    # print(fdiff_2d_precomputed(pars, net, netvals))
    # for j in range(6):
    #     # j = 0  # январь
    #     s = f"_{1979}_{j+1}_{(30, -75)}_2m"
    #     pars, _, _, net, netvals = find_parameters(
    #         x[j * 240: (j + 1) * 240], logs=True,
    #         plot=False, s=s, return_minimised_func=True)  # минимизация, первый шаг
    #     print(fdiff_2d_precomputed(pars, net, netvals))
    # for years, limits in enumerate(slices):
    #     for j in range(12):
    #         # j = 0  # январь
    #         s = f"_{1979}_{j+1}_{(30, -75)}_5m"
    #         pars, _, _, net, netvals = find_parameters(
    #             month_data[j][limits[0]:limits[1]], logs=True,
    #             plot=False, s=s, return_minimised_func=True)  # минимизация, первый шаг
    #         print(fdiff_2d_precomputed(pars, net, netvals))
    # нарисовать фазовое пространство
    # xx, yy = np.mgrid[-0.1:0.1:101j, -0.1:0.1:101j]  # сетка
    # # xx, yy = np.mgrid[-1:1:101j, -1:1:101j]  # сетка
    # # xx, yy = np.mgrid[-10:10:101j, -10:10:101j]  # сетка
    # positions = np.vstack([xx.ravel(), yy.ravel()])
    # cntr = []
    # for point in tqdm.tqdm(positions.T):
    #     prec = echf(net+point)
    #     f = fdiff_2d_precomputed(
    #         (
    #             pars[0], pars[1],  # m
    #             # pars[0]+point[0], pars[1]+point[1],  # m
    #             pars[2], pars[3],  # theta
    #             # pars[2]+point[0], pars[3]+point[1],  # theta
    #             pars[4], pars[5], pars[6],  # Sigma
    #             # pars[4]+point[0], pars[5], pars[6]+point[1],  # Sigma
    #             pars[7]  # tau
    #         ),
    #         net+point, prec)
    #     cntr.append(float(f))

    # plt.contour(
    #     xx, yy, np.array(cntr).reshape(xx.shape),
    #     # np.logspace(-20, 0, 10),
    #     # norm=matplotlib.colors.LogNorm(vmin=1e-20, vmax=1),
    #     #                         zorder=-1
    # )
    # plt.show()

    # fig = go.Figure(data=[go.Surface(z=np.array(cntr).reshape(xx.shape), x=xx, y=yy)])
    # fig.show()

    # pars = [
    #         np.array([20, 30]),
    #         np.array([0, 0]),
    #         np.array([[50, 30], [30, 40]]),
    #         3
    # ]
    #
    # for s in [100, 200, 500, 1000, 2000]:
    #     sample = avg_2d_sample(
    #         np.array([20, 30]),
    #         np.array([0, 0]),
    #         np.array([[50, 30], [30, 40]]),
    #         3,
    #         s
    #     )
    #     echf = make_empiric_chf_2d(sample)
    #     draw_echf(echf, s=f'TESTING_{s}')
    # draw_hyp_chf_1(pars, s='TESTING_target')

    for j in lat_long_list:
        ans = month_distribution_nodraw(
            data_dict[j],
            [i for i in range(1979, 2021, 1)],
        )
        print(j, "по 1 мес")
        for x in ans:
            print(sum(x))
            print(100 * sum(x) / len(ans[0]))

        ans = bimonth_distribution_nodraw(
            data_dict[j],
            [i for i in range(1979, 2021, 1)],
        )
        print(j, "по 2 мес")
        for x in ans:
            print(sum(x))
            print(100 * sum(x) / len(ans[0]))

    # ans = fiveyear_month_distribution_nodraw(
    #     month_data,
    # )


    pass


if __name__ == "__main__":
    main()
