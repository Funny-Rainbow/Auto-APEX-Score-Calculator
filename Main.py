import cv2 as cv
import numpy as np
import csv

# 用于显示图片的函数（主要用于调试）
def cv_show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

# 根据排名计算排名分的函数
def rank_score(rank):
    if rank>15:
        return 0
    elif 10<rank<16:
        return 1
    elif 7<rank<11:
        return 2
    elif 5<rank<8:
        return 3
    elif rank==5:
        return 4
    elif rank==4:
        return 5
    elif rank==3:
        return 7
    elif rank==2:
        return 9
    elif rank==1:
        return 12
    else:
        return None

# 对于一些数字（0、6）常被识别成8，所以多添加了几个素材来消除这个BUG
def calculate(recog):
    if recog == 11:
        return 0
    elif recog == 12:
        return 6
    else:
        return recog

# 将数据写入csv文件
def data_write(data):
    with open("result.csv", 'a', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)
        csvfile.flush()
        csvfile.close()

kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

kernal_open = np.ones((5,5),np.uint8)

# 添加所有参考图片，可用同名文件替换 "./Reference Picture/hello.png"
model_list = []
for i in range(0, 10):
    path = './Reference Picture/'+str(i)+'.png'
    model_list.append(path)

model_list.append("./Reference Picture/hello.png")# 井号
model_list.append("./Reference Picture/00.png")# 第二个0
model_list.append("./Reference Picture/66.png")# 第二个6
print(model_list)

digits = {}
a = 0
csv_data = []
# 给即将写入csv的数据加上第一排的说明
csv_data.append("real rank")
csv_data.append("recog rank")
csv_data.append("rank score")
csv_data.append("kill")
csv_data.append("total score")
data_write(csv_data)

# 识别处理模板
for i in model_list:
    gray = cv.imread(i, cv.IMREAD_GRAYSCALE)
    ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    #thresh = 255 - thresh
    contours, hiderarchy = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cnt = contours[-1]
    # 轮廓
    draw_img = thresh.copy()
    img = cv.cvtColor(draw_img, cv.COLOR_GRAY2BGR)
    res = cv.drawContours(img, contours, -1, (0, 0, 255), 3)
    # 方框

    (x, y, w, h) = cv.boundingRect(cnt)
    eee = cv.rectangle(thresh, (x, y), (x + w, y + h), (0, 255, 0), 3)
    roi = draw_img[y-2:y+h+2, x-2:x+w+2]
    roi = cv.resize(roi, (57,88))
    digits[a]=roi
    a = a+1
    #cv_show('roi',roi)

# 处理分数截图
rectKernel = cv.getStructuringElement(cv.MORPH_RECT, (10,8))
#sqkKernel = cv.getStructuringElement(cv.MORPH_RECT, (5,4))
test = cv.imread("ScreenShot.png")
gray = cv.imread("ScreenShot.png", cv.IMREAD_GRAYSCALE)
#gray = cv.morphologyEx(gray,cv.MORPH_TOPHAT, rectKernel)
#cv_show('tophat',gray)
#ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

#给定20个队伍的位置，已经找出来了，对于1920x1080分辨率全屏截图就无需修改，其他分辨率需要后续完善
location = [(347, 942, 33, 20),(347, 867, 34, 20),(347, 792, 31, 20),(347, 717, 33, 20),
        (347, 642, 34, 20),(347, 567, 34, 20),(347, 492, 33, 20),(347, 399, 33, 20),(347, 287, 26, 20),
        (1123, 943, 51, 20),(1123, 868, 43, 20),(1123, 793, 44, 20),(1123, 718, 41, 20),(1123, 643, 43, 20),
        (1123, 568, 43, 20),(1123, 493, 43, 20),(1123, 418, 43, 20),(1123, 343, 43, 20),(1123, 268, 36, 20),(1123, 193, 44, 20)]

# 排名(识别)顺序如下
my_Rank = [9,8,7,6,5,4,3,2,1,20,19,18,17,16,15,14,13,12,11,10]
rank_Num = 0

# 开始顺次识别每队的击杀、排名情况
for each in location:
    csv_data = []
    #print(each)
    gray = gray.copy()
    #cv_show("gray",gray)
    threshold = cv.threshold(gray, 120, 255, cv.THRESH_BINARY)[1]
    #cv_show('th',threshold)
    each = list(each)

    #print(each)
    a = each[0]
    b = each[1]
    c = each[2]
    d = each[3]
    # 初步处理图像
    team = cv.resize(threshold[b - 10:b + d + 5, a - 5 +68:a + c + 5 +140],(200,100))
    rank = cv.resize(threshold[b - 5:b + d + 5, a - 5+21:a + c + 5],(200,100))
    kill = cv.resize(threshold[b-5:b + d+5, a-5+695:a + c+695],(200,100))
    #cv_show('kill',kill)
    kill = cv.morphologyEx(kill, cv.MORPH_CLOSE,kernal_open)
    kill = cv.morphologyEx(kill, cv.MORPH_CLOSE, kernal_open)
    #cv_show('kill',kill)

    image = np.hstack((rank, kill))
    #cv_show("image",image)
    #image = np.vstack((image, kill))
    digitCnts, hierarchy = cv.findContours(rank.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    digitCnts = list(digitCnts)
    #digitCnts = contours.sort_contours(list(digitCnts),method = "left-to-right")

    # 计算分数
    total_rank = 0
    total_kill = 0
    count = 0
    for c in digitCnts:
        (x,y,w,h) = cv.boundingRect(c)
        roi = rank[y-2:y+h+2,x-2:x+w+2]
        roi = cv.resize(roi,(57,88))

        # 模板匹配得分
        scores = []
        for digit, digitROI in digits.items():
            result = cv.matchTemplate(roi, digitROI, cv.TM_CCORR_NORMED )
            (_,score,_,_) = cv.minMaxLoc(result)
            scores.append(score)
        # 选取最高得分的数字作为识别的数字
        recog = int(np.argmax(scores))
        recog = calculate(recog)

        #cv_show("roi", roi)
        # 最终得出排名
        total_rank = recog*pow(10,count) + total_rank
        count=count+1

    count = 0

    # 人头分
    digitCnts,hierarchy= cv.findContours(kill.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    digitCnts = list(digitCnts)

    for c in digitCnts:
        (x,y,w,h) = cv.boundingRect(c)
        roi = kill[y-2:y+h+2,x-2:x+w+2]
        roi = cv.resize(roi,(57,88))

        # 模板匹配得分
        scores = []
        for digit, digitROI in digits.items():
            result = cv.matchTemplate(roi, digitROI, cv.TM_CCORR_NORMED )
            (_,score,_,_) = cv.minMaxLoc(result)
            scores.append(score)
        # 选取最高得分的数字作为识别的数字
        recog = int(np.argmax(scores))
        recog = calculate(recog)


        #cv_show("roi", roi)
        # 总击杀数
        total_kill = recog*pow(10,count) + total_kill
        count=count+1


    total_rank_score = rank_score(total_rank)
    print('排名：', total_rank, '排名分：', total_rank_score, '击杀：',total_kill, '总分：', total_rank_score+total_kill)

    csv_data.append(my_Rank[rank_Num])
    csv_data.append(total_rank)
    csv_data.append(total_rank_score)
    csv_data.append(total_kill)
    csv_data.append(total_rank_score+total_kill)
    data_write(csv_data)
    #cv_show('image',image)
    #cv_show('img', team)
    rank_Num = rank_Num + 1
input('Press any key to end me')


'''
下方代码用于查找井号位置，暂时弃用
gradX = cv.Sobel(gray, ddepth = cv.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal,maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal)/(maxVal - minVal)))
gradX = gradX.astype("uint8")
cv_show('t',gradX)
gradX = cv.morphologyEx(gradX,cv.MORPH_CLOSE, rectKernel)
thresh = cv.threshold(gradX, 0, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)[1]
cv_show("l",thresh)
contours, hiderarchy = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
cnt = contours[-1]
cnts = contours
# 轮廓
draw_img = thresh.copy()
img = cv.cvtColor(draw_img, cv.COLOR_GRAY2BGR)
res = cv.drawContours(img, contours, -1, (0, 0, 255), 3)
cv_show('res',res)
locs =[]
for (i,c) in enumerate(cnts):
    (x, y, w, h) = cv.boundingRect(c)

    img = cv.cvtColor(draw_img, cv.COLOR_GRAY2BGR)


    ar = w/ float(h)
    if ar >1.2 and ar <3:
        if(w >10 and w<80) and (h>10 and h <40):
            locs.append((x, y, w, h))
            eee = cv.rectangle(test, (x, y), (x + w, y + h), (0, 255, 0), 3)
cv_show('eee', eee)
locs = sorted(locs, key = lambda x:x[0])

output = []
groupOutput = []
i = 1
for (i, (gX,gY,gW,gH)) in enumerate(locs):
    gray = gray.copy()
    group = gray[gY:gY + gH,gX:gX + gW]
    group = cv.threshold(group, 0, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)[1]
    #cv_show('group',group)
    #print(gX,gY,gW,gH)
    digitCnts,hierarchy= cv.findContours(group.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    digitCnts = list(digitCnts)


    #digitCnts = contours.sort_contours(list(digitCnts),method = "left-to-right")
    for c in digitCnts:
        (x,y,w,h) = cv.boundingRect(c)
        roi = group[y:y+h,x:x+w]
        roi = cv.resize(roi,(57,88))
        #cv_show("roi",roi)
        # 模板匹配得分
        scores = []
        for digit, digitROI in digits.items():
            result = cv.matchTemplate(roi, digitROI, cv.TM_CCOEFF)
            (_,score,_,_) = cv.minMaxLoc(result)
            scores.append(score)

        # recog = str(np.argmax(scores))
        # if recog == "0":
        #     print(i)
        #     i=i+1
        #     #cv_show("roi",roi)
        #     groupOutput.append((gX,gY,gW,gH))
        #     print((gX,gY,gW,gH))
        #     hello = gray[gY:gY + gH, gX:gX + gW]
        #     #cv_show("roi", roi)
        #     cv_show('gray',hello)

print("recog", groupOutput)




#abondoned



# for (i,c) in enumerate(cnts):
#     (x,y,w,h) = cv.boundingRect(c)
#     ar = w/ float(h)
#     if ar >1 and ar <3:
#         if(w >40 and w<55) and (h>10 and h <20)
'''