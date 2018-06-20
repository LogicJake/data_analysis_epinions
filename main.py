from numpy import *
import item
import user

def rank(userID,itemID,ElismUserDict,ITrustDict,ElismItemDict,N1=20,N2=20):
    rankByuser = user.rank(userID,itemID,ElismUserDict,ITrustDict,N1)
    rankByitem = item.rank(userID,itemID,ElismItemDict,N2)
    if rankByuser == -1 and rankByitem == -2:
        return -1   #对user无法预测
    elif rankByitem != -2 and rankByitem != -1 and (rankByuser == -1 or rankByuser == -2):
        return rankByitem
    elif rankByuser != -2 and rankByuser != -1 and (rankByitem == -1 or rankByitem == -2):
        return rankByuser
    elif rankByuser != -2 and rankByuser != -1 and rankByitem != -1 and rankByitem != -2:
        return (rankByuser+rankByitem)/2
    else:
        # return (user.average(userID)+item.average(itemID))/2
        return -2


def test(file, ElismUserDict, ITrustDict, ElismItemDict, N1, N2):
    sum = 0
    n = 0
    count = 0
    user = set()
    notPreUser = set()
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
            if pre_rating == -1:
                notPreUser.add(userID)
                count+=1
            elif pre_rating == -2:
                count+=1
            else:
                sum += abs(real_rating-pre_rating)
    print(sum/n," ",(len(user)-len(notPreUser))/len(user)," ",(n-count)/n)

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
    for i in range(5):
        test("testData{}.txt".format(i + 1), ElismUserDict, ITrustDict, ElismItemDict, 10, 35)
