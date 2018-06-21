from numpy import *
import item
import user

def rank(userID,itemID,ElismUserDict,ITrustDict,ElismItemDict,N1=20,N2=20):
    rankByuser = user.rank(userID,itemID,ElismUserDict,ITrustDict,N1)
    rankByitem = item.rank(userID,itemID,ElismItemDict,N2)
    if rankByuser == -1 and rankByitem == -1:
        return -1   #无法预测
    elif rankByitem != -1 and rankByuser == -1:
        return rankByitem
    elif rankByuser != -1 and rankByitem == -1:
        return rankByuser
    elif rankByuser != -1 and rankByitem != -1:
        return (rankByuser+rankByitem)/2
    else:
        # return (user.average(userID)+item.average(itemID))/2
        return -1


def test(file, ElismUserDict, ITrustDict, ElismItemDict, N1, N2):
    sum = 0
    n = 0
    count = 0
    user = set()
    PreUser = set()
    with open(file,"r") as f:
        for line in f.readlines()[1:]:          #跳过首行标题行
            n+=1
            line = line.strip()
            ss = line.split(",")
            userID = int(ss[0])
            user.add(userID)
            itemID = int(ss[1])
            real_rating = int(ss[2])
            pre_rating = rank(userID,itemID,ElismUserDict,ITrustDict,ElismItemDict,N1,N2)
            if pre_rating != -1:
                count += 1
                PreUser.add(userID)
                sum += abs(real_rating-pre_rating)
    return sum/n,len(PreUser)/len(user),count/n

def train(load):
    '''
    :param load:是否优先从本地加载训练好的字典
    :return:
    '''
    ElismUserDict,ITrustDict = user.train(load)
    ElismItemDict = item.train(load)
    return ElismUserDict, ITrustDict,ElismItemDict

if __name__ == '__main__':
    ElismUserDict,ITrustDict,ElismItemDict = train(True)
    print("训练完毕")
    mae = {}
    up = {}
    ip = {}
    for i in range(5):
        MAE, UP, IP = test("testData{}.txt".format(i + 1), ElismUserDict, ITrustDict, ElismItemDict, 70, 70)
        print(MAE,UP, IP)
        mae.setdefault(i, []).append(MAE)
        up.setdefault(i, []).append(UP)
        ip.setdefault(i, []).append(IP)
