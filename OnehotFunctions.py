import numpy as np


def table_generator(dim):
    table = np.ones([dim**2], dtype=np.float32)
    for i in range(dim):
        table[int((i+2)*(i+1)/2-1)] = 0.
    return table


def to_one_hot(alps, dim, one_hot_sections, table):
    one_hot_alps = np.zeros([dim**2*(one_hot_sections+1)], dtype=np.float32)
    for i in range(dim**2):
        temp = (alps[i]+table[i])/(1.+table[i])*one_hot_sections
        ind = np.floor(temp)
        one_hot_alps[int(i*(one_hot_sections+1)+ind)] = 1-(temp-ind)
        one_hot_alps[int(i*(one_hot_sections+1)+ind+1)] = temp - ind
    return one_hot_alps


def to_one_hot_fix(alps, dim, one_hot_sections, table):
    length_of_alps = len(alps)
    one_hot_alps = np.zeros([length_of_alps * (one_hot_sections + 1)], dtype=np.float32)
    for i in range(length_of_alps):
        temp = (alps[i]+table[i])/(1.+table[i])*one_hot_sections
        ind = np.floor(temp)
        one_hot_alps[int(i*(one_hot_sections+1)+ind)] = 1-(temp-ind)
        one_hot_alps[int(i*(one_hot_sections+1)+ind+1)] = temp - ind
    return one_hot_alps


def to_alps(one_hot_alps, dim, one_hot_sections, table):

    alps = np.zeros([dim**2])
    for i in range(dim**2):
        temp = one_hot_alps[int(i*(one_hot_sections+1)):int((i+1)*(one_hot_sections+1))]
        maxgra = np.max(temp)
        maxind = np.argmax(temp)
        if maxind == 0:
            secgra = temp[1]
            secind = 1
        elif maxind == one_hot_sections:
            secgra = temp[one_hot_sections - 1]
            secind = one_hot_sections - 1
        else:
            if temp[maxind - 1] > temp[maxind + 1]:
                secgra = temp[maxind - 1]
                secind = maxind - 1
            else:
                secgra = temp[maxind + 1]
                secind = maxind + 1
        gra1 = maxind / one_hot_sections * (1 + table[i]) - table[i]
        gra2 = secind / one_hot_sections * (1 + table[i]) - table[i]
        alps[i] = gra1 * maxgra / (maxgra + secgra) + gra2 * secgra / (maxgra + secgra)
    return alps


def to_alps_fix(one_hot_alps, dim, one_hot_sections, table):
    length_of_alps = int(len(one_hot_alps) / (one_hot_sections + 1))
    alps = np.zeros([length_of_alps])
    for i in range(length_of_alps):
        temp = one_hot_alps[int(i*(one_hot_sections+1)):int((i+1)*(one_hot_sections+1))]
        maxgra = np.max(temp)
        maxind = np.argmax(temp)
        if maxind == 0:
            secgra = temp[1]
            secind = 1
        elif maxind == one_hot_sections:
            secgra = temp[one_hot_sections - 1]
            secind = one_hot_sections - 1
        else:
            if temp[maxind - 1] > temp[maxind + 1]:
                secgra = temp[maxind - 1]
                secind = maxind - 1
            else:
                secgra = temp[maxind + 1]
                secind = maxind + 1
        gra1 = maxind / one_hot_sections * (1 + table[i]) - table[i]
        gra2 = secind / one_hot_sections * (1 + table[i]) - table[i]
        alps[i] = gra1 * maxgra / (maxgra + secgra) + gra2 * secgra / (maxgra + secgra)
    return alps