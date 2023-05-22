from lib import *


def find_value_robust(f, x0, h, amp, eps=EPS):
    # x0 = np.array(x0)
    # h = np.array(h)
    if (abs(f(x0 + h)) > amp
            or abs(f(x0 + 2 * h)) > amp
            or abs(f(x0 + 3 * h)) > amp
            # or abs(f(x0 + 4*h)) > amp
            # or abs(f(x0 + 5*h)) > amp
    ):
        return find_value_robust(f, x0 + h, h, amp, eps)
    else:
        return find_value_robust(f, x0, h / 2, amp, eps) if sum(abs(h)) > eps else x0


def make_2d_net(echf, rays=RAYS, num=NUM, clip=False, logs=False):
    """
    Строит сетку по известной эмп. хар. ф-и для минимизации на ней.
    """
    #     a = np.linspace(-100, 100, SIZE) #
    #     x1, y1 = np.meshgrid(a, [100])
    #     x2, y2 = np.meshgrid(a, [-100])
    #     x3, y3 = np.meshgrid([100], a)
    #     x4, y4 = np.meshgrid([-100], a)
    #     xx, yy = (np.concatenate((x1.flatten(),x2.flatten(),x3.flatten(),x4.flatten())),
    #               np.concatenate((y1.flatten(),y2.flatten(),y3.flatten(),y4.flatten())))
    #     arr = np.vstack((xx, yy)).T
    x = np.linspace(-101, -100, SIZE)  #
    y = np.linspace(-101, -100, SIZE)  #
    xx, yy = np.meshgrid(x, y)
    arr = np.vstack((xx.flatten(), yy.flatten())).T
    out = echf(arr)
    amp = max(max(out.real), max(out.imag), abs(min(out.real)), abs(min(out.imag)))
    if logs:
        print("Амплитуда функции вдалеке:", amp)
    # angles = np.linspace(0, 2 * np.pi, rays+1)
    angles = np.linspace(0, np.pi, rays+1)
    out = np.array([[0, 0]])
    if clip:
        for i in range(rays):
            h = np.array([np.sin(angles[i]) * 0.01, np.cos(angles[i]) * 0.01])
            val = np.clip(find_value_robust(echf, np.array([0, 0]), h, amp), -1, 1)
            #             val = find_value_robust(echf, [0, 0], h, amp)
            # out = np.concatenate((out, np.logspace(val / num, val, num)))
            if i % 2 == 0:
                out = np.concatenate((out, np.linspace(val / num / 2, val - val / num / 2, num)))
            else:
                out = np.concatenate((out, np.linspace(val / num, val, num)))
    else:
        for i in range(rays):
            h = np.array([np.sin(angles[i]) * 0.01, np.cos(angles[i]) * 0.01])
            #             val = np.clip(find_value_robust(echf, [0, 0], h, amp), -1, 1)
            val = find_value_robust(echf, np.array([0, 0]), h, amp)
            # out = np.concatenate((out, np.logspace(val / num, val, num)))
            if i % 2 == 0:
                out = np.concatenate((out, np.linspace(val / num / 2, val - val / num / 2, num)))
            else:
                out = np.concatenate((out, np.linspace(val / num, val, num)))
    return out[1:]
