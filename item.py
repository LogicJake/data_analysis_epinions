from numpy import mean

R = {}  #项目用户评分矩阵
U = {}  #项目被评分的用户
I = set()  #项目列表

def load_data():
    with open("trainingData.txt","r") as f:
        for line in f.readlines()[1:]:          #跳过首行标题行
            line = line.strip()
            ss = line.split(",")
            userID = int(ss[0])
            itemID = int(ss[1])
            rating = int(ss[2])
            tmp1 = R.setdefault(itemID,{})
            tmp1[userID] = rating

            tmp2 = U.setdefault(itemID,[])
            tmp2.append(userID)

            I.add(itemID)

def MSD(u,v):
    Uu = U.get(u, [])
    Uv = U.get(v, [])
    Uand = set(Uu) & set(Uv)

    if Uand.__len__() == 0:
        return 0

    sum = 0
    parameter = [0,0.25,0.5,0.75,1]         #标准化参数

    for i in Uand:
        sum += pow((parameter(R[u][i]-1) - parameter(R[v][i]-1)), 2)

    return 1 - sum / Uand.__len__()

def Jac(u,v):
    Uu = U.get(u, [])
    Uv = U.get(v, [])
    Uand = set(Uu) & set(Uv)
    Uor = set(Uu) | set(Uv)
    return Uand.__len__()/Uor.__len__()

def ESim(u,v):
    return MSD(u,v) * Jac(u,v)

def average(u):
    Iu = U.get(u, [])
    ranks = []
    for i in Iu:
        ranks.append(R[u][i])
    return mean(ranks)

def rank(u,i,ElismDict,N = 20):
    Necf = {}

    for io in I:
        if io != i:     #防止过拟合
            if ElismDict[i][io] != 0:
                Necf[io] = ElismDict[i][io]

    if len(Necf) == 0:
        return -2
    else:
        Necf = sorted(Necf.items(), key=lambda item: item[1], reverse=True)
        if len(Necf) > N:
            Necf = Necf[:N]

    sum1 = 0
    sum2 = 0

    canPre = False
    for n in Necf:
        if R[n[0]].get(u,-1) != -1:
            canPre = True
            itemid = n[0]
            ave = average(itemid)
            sum1 += n[1]*(R[itemid][u]-ave)
            sum2 += n[1]

    if not canPre:
        return -2
    return sum1/sum2+average(i)

def train(load = True):
    load_data()
    ElismDict = {}
    if(load):
        try:
            with open("ElismItemDict.txt", "r") as f:
                for line in f.readlines():
                    line = line.strip()
                    ss = line.split(",")
                    i = int(ss[0])
                    j = int(ss[1])
                    esim = float(ss[2])
                    tmp1 = ElismDict.setdefault(i, {})
                    tmp1[j] = esim
                return ElismDict
        except FileNotFoundError as e:
            print("打开ElismItemDict失败，重新训练")

    with open("ElismItemDict.txt","w") as f:
        for i in I:
            for j in I:
                esim = ESim(i, j)
                f.write(str(i)+','+str(j)+','+str(esim)+"\n")
                tmp = ElismDict.setdefault(i, {})
                tmp[j] = esim

    return ElismDict

if __name__ == '__main__':
    train(False)