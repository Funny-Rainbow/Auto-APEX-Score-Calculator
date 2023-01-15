import cv2 as cv
#import matplotlib.pyplot as plt
import numpy as np

#用于显示的函数
def cv_show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

model_list = ["./Reference Picture/hello.png"]
for i in range(0,10):
    path = './Reference Picture/'+str(i)+'.png'
    model_list.append(path)
print(model_list)

digits = {}
a=0
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
    roi = draw_img[y:y+h, x:x+w]
    roi = cv.resize(roi, (57,88))
    digits[a]=roi
    a = a+1
#
rectKernel = cv.getStructuringElement(cv.MORPH_RECT, (10,8))
#sqkKernel = cv.getStructuringElement(cv.MORPH_RECT, (5,4))
test = cv.imread("test.png")
gray = cv.imread("test.png", cv.IMREAD_GRAYSCALE)
#gray = cv.morphologyEx(gray,cv.MORPH_TOPHAT, rectKernel)
#cv_show('tophat',gray)
#ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

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
print(contours[0])
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
print(locs)
output = []

print(gray)
for (i, (gX,gY,gW,gH)) in enumerate(locs):
    groupOutput = []
    gray = gray.copy()
    group = gray[gY:gY + gH,gX:gX + gW]
    cv_show('group',group)
    print(gX,gY,gW,gH)

locs = [[9,74,221,23,19]]

# for (i,c) in enumerate(cnts):
#     (x,y,w,h) = cv.boundingRect(c)
#     ar = w/ float(h)
#     if ar >1 and ar <3:
#         if(w >40 and w<55) and (h>10 and h <20)

#
#
#
#

# scores = []
# groupOutput = []
# for digit, digitROI in digits.items():
#     result = cv.matchTemplate(roi, digitROI, cv.TM_CCOEFF)
#     (_,score,_,_) = cv.minMaxLoc(result)
#     scores.append(int(score))
# groupOutput.append(str(np.argmax(scores)))
# print(scores)