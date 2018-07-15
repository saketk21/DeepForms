import cv2
import numpy as np
from preprocess import preProcess,checkContours
from contours import predict

def find_threshold(filePath):

    img = cv2.imread("contours/"+filePath+'.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 0, 255,cv2.THRESH_OTSU)
    img = 255-img
    kernel = np.ones((3,3),np.uint8)
    img = cv2.dilate(img,kernel,iterations = 1)

    #print("image shape", img.shape)
    x = img.shape[1]
    y = img.shape[0]
    v_count = np.zeros((x,))

    for i in range(x):
        for j in range(y):
            if(img[j][i] >200):
                v_count[i]+= 1
    max_left = 0
    max_right = 0
#    print(v_count)
    prefix_sum = np.zeros((x,))
    prefix_sum[0] = v_count[0]
    for i in range(1, x):
        prefix_sum[i] = v_count[i] + prefix_sum[i-1]
    l = int(prefix_sum[x-1]*0.2)
    r = int(prefix_sum[x-1]*0.8)
    i = 0
    #print("min sum required", prefix_sum[x-1]*0.2)
    #print("max sum required", prefix_sum[x-1]*0.8)
    #print("tot px: ", prefix_sum[x-1])
    while( l > prefix_sum[i]):
        i = i+1
    l_r = i
    while( r > prefix_sum[i]):
        i = i+1
    r_r = i
    #print("left: ", l_r,"right: ",  r_r)
    min_val = v_count[l_r];
    min_indx = l_r
    for i in range(l_r-1, r_r+1):
        # print(i, min_val, v_count[i])
        if(min_val > v_count[i]):
            min_val = v_count[i]
            min_indx = i
        #print("\n",min_indx)
    #print("iimage array: ",v_count)
    #print('File '+str(l_r)+' '+str(r_r)+' '+str(min_indx))
    img = 255-img
    max_1 = 0
    max_2 = 0
    max_idx_1 = 0
    max_idx_2 = 0
    #print("min _ indx: ", min_indx)
    for i in range(0, min_indx):
        if max_1<v_count[i]:
            max_1 = v_count[i]
            max_idx_1 = i
    for i in range(min_indx, x):
        if max_2<v_count[i]:
            max_2 = v_count[i]
            max_idx_2 = i
 #   print(max_1,max_idx_1,  max_2,max_idx_2,  min_val, min_indx)
    ans = 0
    if (min(max_1, max_2)>min_val+20):

        msb = img[:, :min_indx+2]
        lsb = img[: , min_indx:]
        cv2.imwrite('contours/'+filePath+'A.jpg',msb)
        cv2.imwrite('contours/'+filePath+'B.jpg',lsb)

        if(checkVoidImage('contours/'+filePath+'A')==False):
            msbDig = predict.predict('contours/'+filePath+'A.jpg')[0]
            ans = 10*ans+msbDig
        if(checkVoidImage('contours/'+filePath+'B')==False):
            lsbDig = predict.predict('contours/'+filePath+'B.jpg')[0]
            ans = 10*ans+lsbDig
        print(ans)
    else:
        if(checkVoidImage('contours/'+filePath+'A')==False):
            ans = predict.predict('contours/'+filePath+'.jpg')[0]
            print(ans)
    return ans

#def findThreshold(filePath):

def getStudentCourse():
    course = ""
    for i in range(1,7):
        course += str(getImage('course'+str(i)))
    return course

def getUserID():
    userId = ""
    for i in range(1,10):
        userId += str(getImage('PRN'+str(i)))
    return userId

def getRes(filePath):
    #Main function for numbers

    if(checkVoidImage('contours/'+filePath)):
        return 0
    result = containsTwo(filePath)
    if result[0]==True:
        return find_threshold(filePath)
    return predict.predict('contours/'+filePath+'.jpg')[0]
def erosion(img):
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.morphologyEx(img,cv2.MORPH_CLOSE, kernel)
    #erosion = cv2.dilate(img,kernel,iterations = 1)
    return erosion

def dilate(img):
    kernel = np.ones((5,5),np.uint8)
    dialted_img = cv2.dilate(img,kernel,iterations = 1)
    return dialted_img

def seperate(filePath):
    img = img = cv2.imread('contours/'+filePath+'.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, imgf = cv2.threshold(img, 0, 255,cv2.THRESH_OTSU)
    img = 255-imgf
    res = containsTwo(filePath)
    if res[0] == False:
        return False,0
    else:
        c1 = res[1]
        c2 = res[2]
        print("centroid: ", c1, c2)
        x = img.shape[1]
        y = img.shape[0]

        v_count = np.zeros((x,))

        for i in range(x):
            for j in range(y):
                if(img[j][i] >200):
                    v_count[i]+= 1
        min_count = v_count[c1[0]]
        min_idx = c1[0]
        print(v_count[c1[0]: c2[0]])
        for idx in range(c1[0], c2[0]+1):
            if min_count >= v_count[idx]:
                min_count = v_count[idx]
                min_idx = idx
        print(int (min_idx))
        img1 = img[:, :int(min_idx)]
        img2 = img[:,int(min_idx):]
        print(min_idx)
        ans = 0
        cv2.imwrite("contours/"+filePath+"A.jpg",img1)
        cv2.imwrite("contours/"+filePath+"B.jpg",img2)
        if(checkVoidImage('contours/'+filePath+'A')==False):
                msbDig = predict.predict('contours/'+filePath+'A.jpg')[0]
                ans = 10*ans+msbDig
        if(checkVoidImage('contours/'+filePath+'B')==False):
                lsbDig = predict.predict('contours/'+filePath+'B.jpg')[0]
                ans = 10*ans+lsbDig
        else:
            if(checkVoidImage('contours/'+filePath+'A')==False):
                ans = predict.predict('contours/'+filePath+'A.jpg')[0]
        return True,ans

def getCentroid(contour):
    M = cv2.moments(contour)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return cx,cy

def containsTwo(filePath):
    img = cv2.imread('contours/'+filePath+'.jpg')
    #img = cv2.imread(filePath+'.jpg')
    if np.any(img) == None:
        return False
    if(img.shape[0]==0 or img.shape[1]==0):
        return False
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    areaImg = img_gray.size
    ret, imgf = cv2.threshold(img_gray, 0, 255,cv2.THRESH_OTSU)
    imgf = erosion(255-imgf)
    #cv2.imshow("contours", imgf)
    #cv2.waitKey()
    contours, hierarchy = cv2.findContours(imgf,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    prev = -1
    cnt = 0
    thresh = 3*areaImg
    thresh = thresh/4
    len1 = len(contours)
    c1 = 0
    c2 = 0
    for i in range(len1):
        area = cv2.contourArea(contours[i])
        if(area>=50 and area<=thresh):
            if(prev == -1):
                prev = i;
                c1 = contours[prev]
            else:
                res = checkContours(contours[prev],contours[i])
                if(res == False):
                    prev = i
                    c2 = contours[i]
                else:
                    continue
            cnt = cnt+1
            cv2.drawContours(img, contours , i, (0,255,0), 1)

    #print(cnt)
    if cnt==2:
        return True,getCentroid(c1),getCentroid(c2)
    return False,0
def getTextRes(filePath):
    #Main function for text
    '''
    if(checkVoidImage(filePath)):
        return "Image is NULL"
    '''
    arr = filePath.split(".")
    return arr[0]

def getCourse(courseid):
    if(int(courseid)==2 or int(courseid) ==3):
        return getTextRes(courseid)
    return getRes(courseid)

def getPRN(PRN):
    return getRes(str(int(PRN)+7))

def getQues(questionString):
    #print(questionString)
    return getSubQues(questionString[0],questionString[1:])

def getSubQues(queID,subQues):
    prefix = 37
    #print(str(queID)+" "+str(subQues))
    prefix = prefix+((int(queID)-1)*15)
    if(subQues=='total'):
        prefix = prefix +7
    else:
        prefix = prefix+int(subQues)
    return getRes(str(prefix))

def getTotal(text):
    if text == 'First':
        return getRes('157')
    elif text == 'Revised':
         return getRes('158')
    else:
         return getRes('156')+getRes('157')

def getImage(text):
    if text[:6] == 'course':
        res = getCourse(text[6:])
    elif text[:3]=='PRN':
        res = getPRN(text[3:])
    elif text[0]=='Q':
        res = getQues(text[1:])
    elif text[:5] == 'total':
        res = getTotal(text[5:])
    return res

def calcVoidPixels(img):
    cnt = 0
    width,height= img.shape
    for row in img:
        for element in row:
            #print(element)
            if(element<200):
               cnt += 1
            if(cnt>40):
                return 1000
    return cnt

def getMarksString():
    finalString = ''
    prefix = 'Q'
    for i in range(1,9):
        for j in range(1,7):
            finalString = finalString+prefix+str(i)+chr(ord('A')+(j-1))+'$'+str(getImage('Q'+str(i)+str(j)))+" "
        finalString = finalString + prefix + str(i) + 'total$' + str(getImage('Q'+str(i)+'total'))+' '
    finalString = finalString + 'total$' + str(getImage('totalFirst'))+' '
    return finalString

def checkVoidImage(filePath):
    img = cv2.imread(filePath+'.jpg')
    if np.any(img) == None:
        return True
    if(img.shape[0]==0 or img.shape[1]==0):
        return True
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = calcVoidPixels(img_gray)
    #print('Res '+str(res))
    if(res>=20):
        return False
    return True

def printData():
    print('Course ID: '+getStudentCourse())
    print('PRN no: '+getUserID())
    print(getMarksString())

def mainFunction(fileName):
    preProcess(fileName)
    return getMarksString()
def decrypt(string):
    arr = string.split(" ")
    len1 = len(arr)
    lst = []
    for i in range(len1-1):
        tmplist = arr[i].split('$')
        lst.append(tmplist[1])
    return lst
def splitString(string):
    lst = decrypt(string)
    finalList = []
    tmpList = []
    for i in range(56):
        if i%7==0:
            tmpList = [lst[i]]
        else:
            tmpList.append(lst[i])
        if i%7==6:
            finalList.append(tmpList)
    return finalList,lst[56]
def errorHandling(lst):
    boolArray = []
    for i in range(8):
        tot = 0
        for j in range(6):
            tot += int(lst[i][j])
        res = False
        if tot == int(lst[i][6]):
            res = True
        boolArray.append(res)
    return boolArray

# print(find_threshold('numbers/five'))
# print(mainFunction('boxes_refined.jpg'))
# print(seperate('74'))
# print(containsTwo('contours/74.jpg'))
# string = (mainFunction('boxes_refined.jpg'))
# lst,number = splitString(string)
# for i in lst:
#     for j in i:
#         print(j,end = " ")
#     print("")
# print(number)
# print(errorHandling(lst))
# #print(containsTwo('55'))
#
