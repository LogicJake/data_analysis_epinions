from numpy import *

R = {}  #用户项目评分矩阵
I = {}  #用户已评分的项目
U = set()  #用户列表
D = {}       #信任距离

MPDist = 3  #最大信任传播距离
threshold = 0.01
neighbor = {}

def load_data():
    # CreatTrustList()
    with open("trainingData.txt","r") as f:
        for line in f.readlines()[1:]:          #跳过首行标题行
            line = line.strip()
            ss = line.split(",")
            userID = int(ss[0])
            itemID = int(ss[1])
            rating = int(ss[2])
            tmp1 = R.setdefault(userID,{})
            tmp1[itemID] = rating

            tmp2 = I.setdefault(userID,[])
            tmp2.append(itemID)

            U.add(userID)

def MSD(u,v):
    '''
    :param u:userID
    :param v:userID
    :return:u，v均方差值
    '''
    Iu = I.get(u,[])
    Iv = I.get(v,[])
    Iand = set(Iu) & set(Iv)                          #求交集

    if len(Iand) == 0:
        return 0

    sum = 0
    parameter = [0,0.25,0.5,0.75,1]         #标准化参数
    for i in Iand:
        sum += pow((parameter[R[u][i]-1]-parameter[R[v][i]-1]),2)

    return 1-sum/len(Iand)

def Jac(u,v):
    '''
    :param u:
    :param v:
    :return: Jaccard相似度
    '''
    Iu = I.get(u, [])
    Iv = I.get(v, [])
    Iand = set(Iu) & set(Iv)  # 求交集

    Ior = set(Iu) | set(Iv)
    return len(Iand)/len(Ior)

def JMSD(u,v):
    '''
    如果两个用户之间的共同评分项非空, 二者之间的JMSD非零, 否则二者之间的JMSD为零. 
    :param u:
    :param v:
    :return:
    '''
    return Jac(u,v) * MSD(u,v)

def average(u):
    Iu = I.get(u, [])
    ranks = []
    for i in Iu:
        ranks.append(R[u][i])
    return mean(ranks)

def variance(u):
    Iu = I.get(u, [])
    ranks = []
    for i in Iu:
        ranks.append(R[u][i])
    return var(ranks)

def Pre(u,v):
    '''
    :param u:
    :param v:
    :return: 偏好相似度
    '''
    averageU = average(u)
    averageV = average(v)

    varU = variance(u)
    varV = variance(v)

    return 1 - ( 1 / ( 1 + exp(-abs(averageU-averageV)*abs(varU-varV)) ) )

def ESim(u,v):
    '''
    最小值为0，最大值为0.5
    :param u:
    :param v:
    :return:
    '''
    return JMSD(u,v)*Pre(u,v)

def Int(u,v):
    Iu = I.get(u, [])
    Iv = I.get(v, [])
    Iand = set(Iu) & set(Iv)

    Is = []  # 交互成功的项目
    If = []  # 交互失败的项目

    averageU = mean(Iu)
    averageV = mean(Iv)

    for i in Iand:
        if (R[u][i] - averageU) * (R[v][i] - averageV) >= 0:  # 同为消极评分或积极评分时
            Is.append(i)
        else:
            If.append(i)

    Sums = 0
    Sumf = 0

    for i in Is:
        Sums += abs(R[u][i] - R[v][i])

    for i in If:
        Sums += abs(R[u][i] - R[v][i])

    if Sums + Sumf == 0:
        if len(Iand) != 0:  # 假如对评价完全一致则完全信任
            return 0.5
        else:
            return 0
    return (Sums/Sums+Sumf)

def IDTrust(u,v):
    '''
    直接信任，基于两个用户曾经交互的项目
    '''
    return JMSD(u,v)*Int(u,v)

def PTrust(IDTrustDict):
    ITrustDict = IDTrustDict
    sum1 = 0
    sum2 = 0
    for i in U:
        for j in U:
            if ITrustDict[i][j] == 0:  # 还没有关系
                for nn in neighbor[i]:  # 遍历i的邻居节点
                    if ITrustDict[nn][j] != 0:  # 能够联通
                        sum1 += ITrustDict[nn][j] * ITrustDict[i][nn]
                        sum2 += ITrustDict[i][nn]
                if sum2 != 0:
                    ITrustDict[i][j] = sum1/sum2
                    sum1 = 0
                    sum2 = 0
    return ITrustDict

def ITrust(u,v):
    return IDTrust(u,v)

def train(load = True):
    load_data()
    ElismDict = {}
    IDTrustDict = {}
    ITrustDict = {}
    if(load):
        try:
            with open("ElismDict.txt", "r") as f1, open("ITrustDict.txt", "r") as f2:
                for line in f1.readlines():  # 跳过首行标题行
                    line = line.strip()
                    ss = line.split(",")
                    i = int(ss[0])
                    j = int(ss[1])
                    esim = float(ss[2])
                    tmp1 = ElismDict.setdefault(i, {})
                    tmp1[j] = esim

                for line in f2.readlines():
                    line = line.strip()
                    ss = line.split(",")
                    i = int(ss[0])
                    j = int(ss[1])
                    itru = float(ss[2])
                    tmp1 = ITrustDict.setdefault(i, {})
                    tmp1[j] = itru
                return ElismDict, ITrustDict
        except FileNotFoundError as e:
            print("打开失败，重新训练")

    with open("ElismDict.txt","w") as f1:
        for i in U:
            for j in U:
                esim = ESim(i,j)
                f1.write(str(i)+','+str(j)+','+str(esim)+"\n")
                tmp = ElismDict.setdefault(i,{})
                tmp[j] = esim

                itru = ITrust(i, j)         #建立直接信任矩阵
                tmp2 = IDTrustDict.setdefault(i, {})
                tmp2[j] = itru

                if(itru > threshold):          #如果大于阈值，可以加入i的传播路径上的邻居节点
                    tmp4 = neighbor.setdefault(i,[])
                    tmp4.append(j)

    #继续建立间接信任矩阵
    ITrustDict = PTrust(IDTrustDict)
    with open("ITrustDict.txt","w") as f2:
        for i in U:
            for j in U:
                f2.write(str(i)+','+str(j)+','+str(ITrustDict[i][j])+"\n")

    return ElismDict,ITrustDict

def rank(u,i,ElismDict,ITrustDict,N=20):
    Necf = {}
    Ntrust = {}

    for uo in U:
        if uo != u:     #防止过拟合
            if ElismDict[u][uo] != 0:
                Necf[uo] = ElismDict[u][uo]
            if ITrustDict[u][uo] != 0:
                Ntrust[uo] = ITrustDict[u][uo]

    if len(Necf)+len(Ntrust) == 0:
        return -1           #-1代表没有足够信息对该用户预测
    # else:
    #     Necf = sorted(Necf.items(), key=lambda item: item[1], reverse=True)
    #     Ntrust = sorted(Ntrust.items(), key=lambda item: item[1], reverse=True)
    #     if len(Necf) > N:
    #         Necf = Necf[:N]
    #     if len(Ntrust) > N:
    #         Ntrust = Ntrust[:N]

    sum1 = 0
    sum2 = 0
    sum3 = 0

    canItemPre = False
    for n in Necf.items():
        if R[n[0]].get(i,-1) != -1:
            canItemPre = True
            uid = n[0]
            ave = average(uid)
            sum1 += n[1]*(R[uid][i]-ave)
            sum3 += n[1]

    for n in Ntrust.items():
        if R[n[0]].get(i,-1) != -1:
            canItemPre = True
            uid = n[0]
            ave = average(uid)
            sum2 += n[1]*(R[uid][i]-ave)
            sum3 += n[1]

    if not canItemPre:
        return -1 #代表无法对该item进行预测

    return (sum1+sum2)/sum3+average(u)

if __name__ == '__main__':
    train(False)