with open('liste', 'r') as f :
    _ = f.readline()
    T = f.readlines()
    L = []
    for i in range(len(T)):
        T[i] = T[i].strip().split('\t')
        L.append(T[i][0])
    print(L)