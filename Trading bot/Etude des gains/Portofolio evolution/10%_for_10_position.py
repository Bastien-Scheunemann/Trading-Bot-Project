import matplotlib.pyplot as plt
import random
from numpy import linspace


def the_best_ratio(r1, r2):

    """     r1 et r2 s'exprime en % ;
            exemple : r1 = 8, r2 = 5
    """

    print("Position value evolution compared by number of loss position")
    position_value = 1.0

    for p in range(0,11):

        position_value = position_value * (1 + 0.01*r1)**(10-p) * (1 - 0.01*r2)**p

        if position_value < 1:

            print(str(p) + " loss and " + str(10-p) + " gain : " + str(round(-position_value * 100,1)) + "%")

        else:

            print(str(p) + " loss and " + str(10 - p) + " gain : +" + str(round(position_value * 100, 1)) + "%")

        position_value = 1.0
    return None


print(the_best_ratio(8, 3))


def one_years_evlolution(portofolio_size):

    L = [portofolio_size]
    G = []
    ini = portofolio_size

    for j in range(1000000):

        portofolio_size = ini

        for i in range(253):

            alea = random.random()

            if alea < 0.3:

                portofolio_size *= 1.08

            else:

                portofolio_size *= 0.97

        L.append(portofolio_size)

    for g in range(199, len(L)):

        sum = 0

        for h in range(200):

            sum += L[g - h]

        G.append(sum / 200)

    plt.clf()
    #plt.plot([x for x in range(len(L))], L)
    plt.plot([x for x in range(200, len(G) + 200)], G)
    plt.savefig("proba_0.3.png")

    return portofolio_size


one_years_evlolution(1000)











