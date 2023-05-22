from generation import *

CHF_DRAW_LIMIT = 1


def draw_echf(echf, s=""):
    x, y = np.linspace(-CHF_DRAW_LIMIT, CHF_DRAW_LIMIT, 401), np.linspace(-CHF_DRAW_LIMIT, CHF_DRAW_LIMIT, 401)
    xx, yy = np.meshgrid(x, y)
    buffer = echf(np.vstack((xx.flatten(), yy.flatten())).T)
    fig = go.Figure(data=[go.Surface(z=np.array(buffer.real).reshape(401, 401), x=x, y=y)])
    plotly.offline.plot(fig, filename=f"E:/дипломная/imgnew/echf_real{s}.html", auto_open=False)
    fig = go.Figure(data=[go.Surface(z=np.array(buffer.imag).reshape(401, 401), x=x, y=y)])
    plotly.offline.plot(fig, filename=f"E:/дипломная/imgnew/echf_imag{s}.html", auto_open=False)


def draw_hyp_chf_1(pars, s=""):
    x, y = np.linspace(-CHF_DRAW_LIMIT, CHF_DRAW_LIMIT, 401), np.linspace(-CHF_DRAW_LIMIT, CHF_DRAW_LIMIT, 401)
    xx, yy = np.meshgrid(x, y)
    buffer = avg_2d_chf_vector(np.vstack((xx.flatten(), yy.flatten())).T, *pars)
    fig = go.Figure(data=[go.Surface(z=np.array(buffer.real).reshape(401, 401), x=x, y=y)])
    plotly.offline.plot(fig, filename=f"E:/дипломная/imgnew/hyp_chf_real{s}.html", auto_open=False)
    fig = go.Figure(data=[go.Surface(z=np.array(buffer.imag).reshape(401, 401), x=x, y=y)])
    plotly.offline.plot(fig, filename=f"E:/дипломная/imgnew/hyp_chf_imag{s}.html", auto_open=False)


def draw_hyp_chf_2(pars, s=""):
    x, y = np.linspace(-CHF_DRAW_LIMIT, CHF_DRAW_LIMIT, 401), np.linspace(-CHF_DRAW_LIMIT, CHF_DRAW_LIMIT, 401)
    xx, yy = np.meshgrid(x, y)
    buffer = avg_2d_mixture_chf_vector(np.vstack((xx.flatten(), yy.flatten())).T, *pars)
    fig = go.Figure(data=[go.Surface(z=np.array(buffer.real).reshape(401, 401), x=x, y=y)])
    plotly.offline.plot(fig, filename=f"E:/дипломная/imgnew/hyp_chf_mix_real{s}.html", auto_open=False)
    fig = go.Figure(data=[go.Surface(z=np.array(buffer.imag).reshape(401, 401), x=x, y=y)])
    plotly.offline.plot(fig, filename=f"E:/дипломная/imgnew/hyp_chf_mix_imag{s}.html", auto_open=False)


def draw_net(t, s=""):
    fig = px.scatter(x=t[:, 0], y=t[:, 1])
    plotly.offline.plot(fig, filename=f"E:/дипломная/imgnew/net{s}.html", auto_open=False)


def draw_sample_with_hyp(sample, gen, s=""):
    fig = px.area()
    fig.update_traces(marker=dict(color='green'))
    fig.add_scatter(x=sample[:, 0], y=sample[:, 1], mode='markers', opacity=0.8)
    fig.update_traces(marker=dict(color='violet'))
    fig.add_scatter(x=gen[:, 0], y=gen[:, 1], mode='markers', opacity=0.5)
    fig.update_layout(yaxis=dict(scaleanchor="x"))
    plotly.offline.plot(fig, filename=f"E:/дипломная/imgnew/2sample{s}.html", auto_open=False)


def draw_hyp_density(x, y, zz, s=""):
    fig = go.Figure(data=[go.Surface(z=zz, x=x, y=y)])
    plotly.offline.plot(fig, filename=f"E:/дипломная/imgnew/hyp_density{s}.html", auto_open=False)
    pass
