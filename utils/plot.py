import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def level_plot(data, **kwargs):
    n_levels = len(data)

    for i in range(n_levels):
        plt.plot(data[i], **{param: values[i] for param, values in kwargs.items()})
