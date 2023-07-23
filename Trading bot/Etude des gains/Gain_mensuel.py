import matplotlib.pyplot as plt
from numpy import linspace


def gain_brut(u0, A, n):
    x_list = [0]
    un_list = [u0]
    for i in range(1,n):
        x_list.append(i)
        un = (A ** i) * u0
        un_list.append(un)
    return x_list, un_list


def gain_net(u0, A, n):
    x_list = [0]
    un_list = [0.7*u0]
    for i in range(n):
        x_list.append(i)
        un = u0 * 1.07**i
        un = un * 0.7
        un_list.append(un)
    return x_list, un_list


def differencielle_gain(u0, A, n):
    x_list = [0]
    un_list = [0]
    for i in range(n):
        x_list.append(i)
        un = gain_brut(u0, A, n)[1][i] - gain_net(u0, A, n)[1][i]
        un_list.append(un)
    return x_list, un_list


print(gain_brut(1000, 1.1, 120))


plt.clf()
plt.plot(gain_brut(10000, 1.1, 80)[0], gain_brut(10000, 1.1, 80)[1], '.')
plt.plot(gain_net(10000, 1.1, 80)[0], gain_net(10000, 1.1, 80)[1], '--')
plt.grid()
plt.savefig('gain_graphe.png')



