from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from util import *
from kstest2d import *
from draw import *


def gaussian_em(data, years, label, dirname=None, n_components=2, plot=False):
    for year in years:
        start, end = one_year_borders[year - 1979]
        x = data[start:end]
        indexes = np.arange(0, x.shape[0])

        fig, axes = plt.subplots(figsize=(20, 12), nrows=3, ncols=4)
        fig.suptitle(str(year) + f", {label} gaussian mix {n_components}", fontsize=25, y=0.93)

        for j in range(0, 12):
            axes[j // 4, j % 4].scatter(x[j * 120: (j + 1) * 120, 0], x[j * 120: (j + 1) * 120, 1], s=10)
            xmin, xmax, ymin, ymax = -100, 400, -100, 1200  # пределы для рисования контура
            xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]  # сетка
            ################
            start = time.time()
            gm = GaussianMixture(n_components=n_components)
            output = gm.fit(x[j * 120: (j + 1) * 120])
            print(time.time() - start)
            #             print(gm.covariances_, gm.means_, gm.weights_)
            ################
            pos = np.dstack((xx, yy))
            Z = None
            components = []
            for i in range(n_components):
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
                #                     gen = np.vstack([z1.rvs(c), z2.rvs(SAMPLE_SIZE - c)])
                #                 print(len(gen), len(gen[0]))
                test_res.append(ks2d2s(
                    x[j * 120: (j + 1) * 120, 0],
                    x[j * 120: (j + 1) * 120, 1],
                    np.array(gen)[:, 0],
                    np.array(gen)[:, 1]))
                if plot:
                    s = f"_{year}_{j+1}_{label}_gmm"
                    draw_sample_with_hyp(x[j * 120: (j + 1) * 120], np.array(gen), s=s)
            test = sum(test_res) / 5
            print("p-значение теста:", test)

            axes[j // 4, j % 4].contour(
                xx, yy, Z, norm=matplotlib.colors.LogNorm(vmin=1e-20, vmax=1), levels=np.logspace(-20, 0, 10),
            )
            p = ", p={:.4f}, {} комп.".format(test, n_components)
            axes[j // 4, j % 4].set(title=MONTHS[j] + p, xlabel='sensible', ylabel='latent')
            axes[j // 4, j % 4].set_xlim((xmin, xmax))
            axes[j // 4, j % 4].set_ylim((ymin, ymax))
            axes[j // 4, j % 4].grid()
            axes[j // 4, j % 4].label_outer()
        if dirname is not None:
            filename = os.path.join(dirname, str(year) + f", {label} gaussian mix {n_components}")
            plt.savefig(filename, dpi=300)
            plt.close()
            print('Figure is saved in ' + filename)
        else:
            plt.show()


def gaussian_em_bimonth(data, years, label, method=None, dirname=None, n_components=2):
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
            fig.suptitle(str(year) + f", {label} bimonth mixture gaussian", fontsize=25, y=0.93)

            for j in range(0, 6):
                axes[j // 3, j % 3].scatter(x[j * 240: (j + 1) * 240, 0], x[j * 240: (j + 1) * 240, 1], s=10)
                ################
                xmin, xmax, ymin, ymax = -100, 400, -100, 1200  # пределы для рисования контура
                xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]  # сетка
                ################
                gm = GaussianMixture(n_components=n_components)
                output = gm.fit(x[j * 240: (j + 1) * 240])
                print(gm.covariances_, gm.means_, gm.weights_)
                ################
                pos = np.dstack((xx, yy))
                Z = None
                components = []
                for i in range(n_components):
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
                    #                     gen = np.vstack([z1.rvs(c), z2.rvs(SAMPLE_SIZE - c)])
                    #                 print(len(gen), len(gen[0]))
                    test_res.append(ks2d2s(
                        x[j * 240: (j + 1) * 240, 0],
                        x[j * 240: (j + 1) * 240, 1],
                        np.array(gen)[:, 0],
                        np.array(gen)[:, 1]))
                test = sum(test_res) / 5
                print("p-значение теста:", test)

                axes[j // 3, j % 3].contour(
                    xx, yy, Z, norm=matplotlib.colors.LogNorm(vmin=1e-20, vmax=1), levels=np.logspace(-20, 0, 10),
                )
                p = ", p={:.4f}, {} комп.".format(test, n_components)
                axes[j // 3, j % 3].set(title=MONTHS[2 * j] + ", " + MONTHS[2 * j + 1] + p, xlabel='sensible',
                                        ylabel='latent')
                axes[j // 3, j % 3].set_xlim((xmin, xmax))
                axes[j // 3, j % 3].set_ylim((ymin, ymax))
                axes[j // 3, j % 3].grid()
                axes[j // 3, j % 3].label_outer()
            #                 axes[j // 3, j % 3].set_facecolor('#46085c')

            if dirname is not None:
                filename = os.path.join(dirname, str(year) + f", {label} bimonth mixture gaussian")
                plt.savefig(filename, dpi=300)
                plt.close()
                print('Figure is saved in ' + filename)
            else:
                plt.show()


def gaussian_em_fivemonth(month_data, slices, label, dirname=None, n_components=2):
    lat, long = label
    for years, limits in enumerate(slices):
        fig, axes = plt.subplots(figsize=(20, 12), nrows=3, ncols=4)
        fig.suptitle(
            'Средний климатический год: ' +
            str((lat, long)) + f", годы с {years * 5 + 1979} по {(years + 1) * 5 + 1978}",
            fontsize=25, y=0.95)
        for j in range(0, 12):
            axes[j // 4, j % 4].scatter(
                month_data[j][limits[0]:limits[1], 0], month_data[j][limits[0]:limits[1], 1], s=10)

            xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]  # сетка
            positions = np.vstack([xx.ravel(), yy.ravel()])
            cntr = []

            ################
            gm = GaussianMixture(n_components=n_components)
            output = gm.fit(month_data[j][limits[0]:limits[1]])
            print(gm.covariances_, gm.means_, gm.weights_)
            ################
            pos = np.dstack((xx, yy))
            Z = None
            components = []
            for i in range(n_components):
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
                #                     gen = np.vstack([z1.rvs(c), z2.rvs(SAMPLE_SIZE - c)])
                #                 print(len(gen), len(gen[0]))
                test_res.append(ks2d2s(
                    month_data[j][limits[0]:limits[1], 0],
                    month_data[j][limits[0]:limits[1], 1],
                    np.array(gen)[:, 0],
                    np.array(gen)[:, 1]))
            test = sum(test_res) / 5
            print("p-значение теста:", test)
            ####################################################################
            axes[j // 4, j % 4].contour(
                xx, yy, Z, norm=matplotlib.colors.LogNorm(vmin=1e-20, vmax=1), levels=np.logspace(-20, 0, 10),
            )
            p = ", p={:.4f}, {} комп.".format(test, n_components)
            axes[j // 4, j % 4].set(title=MONTHS[j] + p, xlabel='sensible', ylabel='latent')
            axes[j // 4, j % 4].set_xlim((xmin, xmax))
            axes[j // 4, j % 4].set_ylim((ymin, ymax))
            axes[j // 4, j % 4].grid()
            axes[j // 4, j % 4].label_outer()
        #         axes[j // 4, j % 4].set_facecolor('#46085c')

        if dirname is not None:
            filename = os.path.join(dirname,
                                    str(years * 5 + 1979) +
                                    f", {(lat, long)} среднеклиматич 5 лет gaussian mix {n_components}")
            plt.savefig(filename, dpi=300)
            plt.close()
            print('Figure is saved in ' + filename)
        else:
            plt.show()