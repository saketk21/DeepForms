import numpy as np
import os
import cv2
print("vvt\n")

#inputing the image
'''

def info_image(image_name):
    image = Image.open(image_name)
    #print file format of the source file
    print("image format: ",image.format)
    print("mode: ",image.mode)
    image.show()
    print("image size: ",image.size)
    image.thumbnail((400,400))
    print("image size: ",image.size)
    print(image.palette)
    image.show()


def blur(image_name):
    image = cv2.imread(image_name)
   
    cv2.blur(image,(15,15))
   


def save_as_gray(image_name):
    image = cv2.imread(image_name)
    cv2.imshow("vvt",image)
    image= cv2.cvtColor( image, cv2.COLOR_RGB2GRAY )
    cv2.imshow("vvt1",image)
    cv2.imwrite("gray_sample2.jpg",image)


image = Image.open("sample2.jpg")
image.thumbnail((400,400))
image.save("sample2.jpg")

#image saved as gray scale image 
#save_as_gray("sample2.jpg")

def min_max_filter(image_name):
    image = Image.open(image_name)
    im1 = image.filter(ImageFilter.MinFilter(3))
    im2 = image.filter(ImageFilter.MaxFilter(3))
    im3 = image.filter(ImageFilter.MedianFilter(3))
    im1.save("minfilter.jpg")
    im3.save("medianFilter.jpg")
    im2.save("maxfilter.jpg")

def increase_contrast(image_name):
    image = Image.open(image_name)
    image = ImageOps.autocontrast(image,cutoff = 0, ignore=None)
    image.show()
# demo of min_max filter
#min_max_filter("gray_sample2.jpg")

image = Image.open("gray_sample2.jpg")
image = image.filter(ImageFilter.SHARPEN)
image.show()
from PIL import ImageChops
im0 = Image.open("gray_sample2.jpg")
im1 = Image.open("minfilter.jpg") 
im2 = Image.open("maxfilter.jpg")

diff = ImageChops.subtract(im2, im1)
newim2 = ImageChops.add(im2, diff)
newim2.show()
image = Image.open("gray_sample2.jpg")
image = ImageOps.equalize(image, mask=None)
image.show()
'''





# take the image

def read_image(image_name):
    img = cv2.imread(image_name)
    
    return img

def remove_point_noise(img):
    img = cv2.medianBlur(img, 3)
    return img;


def blurring_gaussian(img):
    img = cv2.GaussianBlur(img,(3,3),5)
    return img


def blurring_average(img):
     return cv2.boxFilter(img , -1, (3,3))

    
def sharpen_image(img):
    new_img = img
    img = cv2.Sobel(img,cv2.CV_8U,3,3,ksize = 5)
    img = new_img - img
    return img

def edges(img):
    return cv2.Canny(img, 70,100)

def matching_pattern(img, x, y, filters):
    flg = True
    for position in positions:
        if filters[position[0]+1][position[1]+1] != img[x + position[0]][y + position[1]]  :
                         flg = False
    return flg

                         
def thining_edges(img, filters):
    for i in range(1,len(img)-1):
        if i%100 == 0:
            print(i)
        for j in range(1,len(img[i])-1):
                
                if matching_pattern(img, i, j, filters):
                         img[i][j] = 0
    return img
    
def thining_img(img):
        i = 0
        img=  np.asarray(img)
        sess = tf.Session()
        while( i < 10):
            shape = img.shape
            img = np.reshape(img, ( 1,shape[0], shape[1],-1))
            img_t = tf.convert_to_tensor(img)
            print(img_t.shape)
            print(thining_filter_top.shape)
            img_t = tf.nn.convolution(img_t, thining_filter_top,padding = "SAME" ,strides=None,dilation_rate=None,   name=None)
            img_t = sess.run(img_t)

            shape = img_t.shape
            print(img_t.shape)
            print(thining_filter_bottom.shape)
            img = np.reshape(img_t, ( 1,shape[0], shape[1],-1))
            img_t = tf.convert_to_tensor(img_t)
            
            img_t = tf.nn.convolution(img_t, thining_filter_bottom,padding = "SAME" ,strides=None,dilation_rate=None,   name=None)
            img_t = sess.run(img_t)

            shape = img.shape
            img = np.reshape(img_t, ( 1,shape[0], shape[1],-1))
            img_t = tf.convert_to_tensor(img_t)
            
            img_t = tf.nn.convolution(img_t, thining_filter_right,padding = "SAME" ,strides=None,dilation_rate=None,   name=None)
            img_t = sess.run(img_t)

            shape = img.shape
            img = np.reshape(img_t, ( 1,shape[0], shape[1],-1))
            img_t = tf.convert_to_tensor(img_t)
            
            img_t = tf.nn.convolution(img_t, thining_filter_left,padding = "SAME" ,strides=None,dilation_rate=None,   name=None)
            img_t = sess.run(img_t)
            
            print("good")
            print(img_t*255)
            #cv2.imshow("imtermediate " , img_t*255)
            i = i+1


size = 0
def erosion(img):
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.morphologyEx(img,cv2.MORPH_CLOSE, kernel)
    #erosion = cv2.dilate(img,kernel,iterations = 1)
    return erosion

def contourThresholding(cnt):
    return areaThresholding(cnt) and dimensionThresholding(cnt)

def areaThresholding(cnt):
        area = cv2.contourArea(cnt)
        #print(size)
        if(area>2000 and area<30000):
            return True

def dimensionThresholding(cnt):
    rect = cv2.minAreaRect(cnt)
    width,height = rect[1]
    print(str(height)+" "+str(width))
    return height>30 and height<400 and width>30 and width<400


def skeletonization(img):
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
    size = np.size(img)
    while(not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy();
        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True
    return skel




def remove_contours_with_less_area(contours, image):
    new_contours = []
    size = image.shape[0]*image.shape[1];
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if(area>8000 and area<12000 ):
            new_contours.append(cnt)
    return new_contours

def checkContours(cnt1,cnt2):
    rect = cv2.minAreaRect(cnt1)
    box = cv2.boxPoints(rect)
    maxX = 0
    maxY = 0
    minX = 10000000
    minY = 10000000
    for i in box:
        minX = min(minX,i[0])
        minY = min(minY,i[1])
        maxX = max(maxX,i[0])
        maxY = max(maxY,i[1])
    minX = int(minX)
    maxX = int(maxX)
    minY = int(minY)
    maxY = int(maxY)
    M = cv2.moments(cnt2)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    if((cx>minX) and (cx<maxX) and (cy>minY) and (cy<maxY) ):
        return True
    return False


def getSize(img):
    width,height = img.shape
    return width*height

def cropContour(cnt,img):
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    maxX = 0
    maxY = 0
    minX = 10000000
    minY = 10000000
    for i in box:
        minX = min(minX,i[0])
        minY = min(minY,i[1])
        maxX = max(maxX,i[0])
        maxY = max(maxY,i[1])
    minX = int(minX)
    maxX = int(maxX)
    minY = int(minY)
    maxY = int(maxY)

    print(str(minX)+" "+str(maxX))
    print(str(minY)+" "+str(maxY))
    return img[minY+2:maxY, minX:maxX]

def preProcess(filePath):
    img = read_image(filePath)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #img_blur = cv2.GaussianBlur(img_gray,(5,5),0)
    ret, imgf = cv2.threshold(img_gray, 0, 255,cv2.THRESH_OTSU)
    #cv2.imshow("binarized image",imgf)

    #cv2.imwrite("thresholding.jpg", 255- imgf)
    dilated_image = erosion(255 - imgf)
    #eroded_image = erosion(255 - eroded_image)
    #cv2.imwrite("closing image.jpg", dilated_image)
    print("written")

    '''
        TODO: This is later part but need to check accuracy by thining method
    '''
    contours, hierarchy = cv2.findContours(dilated_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    new_contours = []
    for cnt in contours:
        epsilon = 0.1*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        if contourThresholding(cnt) and len(approx)==4:
            new_contours.append(approx)
    length = len(new_contours)
    base_contour = new_contours[0]  
    final_contours = [base_contour]
    for i in range(1,length):
        if checkContours(base_contour,new_contours[i])==True:
                 continue
        else:
                 final_contours.append(new_contours[i])
                 base_contour = new_contours[i]
    final_contours.reverse()
    for i in range(len(final_contours)):
        #cv2.drawContours(img, final_contours, i, (0,255,0), 1)
        #cv2.imwrite('contours/'+str(i)+'.jpg',cropContour(final_contours[i],img))
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'contours\\' + str(i + 1) + '.jpg')
        print(path)
        cv2.imwrite(path, cropContour(final_contours[i], img))
    cv2.drawContours(img, final_contours, -1, (0, 255, 0), 1)
    cv2.imshow("contours", img)
