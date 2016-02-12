__author__ = 'Administrator'
import sys
import numpy as np
import cv2 as cv
import math
from argparse import ArgumentParser
FILENAME = "1.150000001.png";

HOUGH_VOTE = 100
GRAY_THRESH  = 150

#srcImgOrg : the source image
#srcImgGray : the source image with gray scale
def calcRotAngle(srcImgOrg,srcImgGray):
    angleD = 0
    opWidth = cv.getOptimalDFTSize(srcImgGray.shape[1])
    opHeight = cv.getOptimalDFTSize(srcImgGray.shape[0])

    padded = cv.copyMakeBorder(srcImgGray, 0, opWidth - srcImgGray.shape[1] , 0, opHeight - srcImgGray.shape[0], cv.BORDER_CONSTANT);
    plane = np.zeros(padded.shape,dtype=np.float32)
    planes = [padded,plane]
    #Merge into a double-channel image
    comImg = cv.merge(planes)
    cv.dft(comImg,comImg)
    cv.split(comImg, planes)

    planes[0] = cv.magnitude(planes[0], planes[1]);
    magMat = planes[0]
    magMat += np.ones(magMat.shape)
    cv.log(magMat,magMat);

    cx = magMat.shape[1] / 2;
    cy = magMat.shape[0] / 2
    q0 = magMat[0:cx,0: cy ]
    q1 = magMat[cx:,0: cy]
    q2 = magMat[0:cx, cy:]
    q3 = magMat[cx:,cy:]
    c1 = np.vstack((q3,q2))
    c2 = np.vstack((q1,q0))
    magMat2 = np.hstack((c1,c2))

    cv.normalize(magMat2, magMat, 0, 1,cv.NORM_MINMAX);
    magMat = cv.resize(magMat,(magMat.shape[0] / 2,magMat.shape[1]/2))
    magMat = magMat * 255
    magMat = cv.threshold(magMat,GRAY_THRESH,255,cv.THRESH_BINARY)[1].astype(np.uint8)
    lines = cv.HoughLines(magMat,1,np.pi/180, HOUGH_VOTE);
    #cv.imshow("mag_binary", magMat);
    #lineImg = np.ones(magMat.shape,dtype=np.uint8)
    angle = 0
    if lines != None and len(lines) != 0:
        for line in lines[0]:
            #print line
            rho = line[0]
            theta = line[1]
            if  (theta < (np.pi/4. )) or (theta > (3.*np.pi/4.0)):
                print 'Vertical line , rho : %f , theta : %f'%(rho,theta)
                pt1 = (int(rho/np.cos(theta)),0)
                pt2 = (int((rho-magMat.shape[0]*np.sin(theta))/np.cos(theta)),magMat.shape[0])
                #cv.line( lineImg, pt1, pt2, (255))
                angle = theta
            else:
                print 'Horiz line , rho : %f , theta : %f'%(rho,theta)
                pt1 = (0,int(rho/np.sin(theta)))
                pt2 = (magMat.shape[1], int((rho-magMat.shape[1]*np.cos(theta))/np.sin(theta)))
                #cv.line(lineImg, pt1, pt2, (255), 1)
                angle = theta + np.pi / 2
        #cv.imshow('lineImg',lineImg)
        #Find the proper angel
        if angle > (np.pi / 2):
            angle = angle - np.pi

        #Calculate the rotation angel
        #The image has to be square,
        #so that the rotation angel can be calculate right
        print 'angle : %f' % angle

        #print srcImgOrg.shape
        alpha = float(srcImgOrg.shape[1]) / float(srcImgOrg.shape[0])
        print 'alpha : %f' % alpha
        if alpha > 1:
            angleT = srcImgOrg.shape[1] * np.tan(angle) / srcImgOrg.shape[0];
            angleD = np.arctan(angleT) * 180 / np.pi;
        else:
            angleD = angle * 180 / np.pi
        print 'angleD : %f' % angleD
    return angleD

def rotImage(srcImgOrg,angleD):
    size = srcImgOrg.shape
    centerPnt = (srcImgOrg.shape[1] / 2,srcImgOrg.shape[0] / 2)
    rotMat = cv.getRotationMatrix2D(centerPnt,angleD,scale=1.);
    resultImg = cv.warpAffine(srcImgOrg,rotMat,(size[1],size[0]));

    #cv.imshow('srcImgOrg',srcImgOrg);
    #resultImg = cv.resize(resultImg,(resultImg.shape[0] / 2,resultImg.shape[1]/2))
    #cv.imshow("resultImg",resultImg);
    fileParts = fileName.split('.')
    fileParts[-2] = fileParts[-2] + '-r'
    file = '.'.join(fileParts)
    print "file name : %s" % file

    ret = cv.imwrite(file,resultImg)

def handleImage(fileName):
    srcImgOrg = cv.imread(fileName)
    srcImgGray = cv.imread(fileName,cv.IMREAD_GRAYSCALE).astype(np.float32);
    angle = calcRotAngle(srcImgOrg,srcImgGray)
    if angle > 0:
        rotImage(srcImgOrg,angle)

def main():
    p = ArgumentParser(usage='it is usage tip', description='this is a usage tip')
    p.add_argument('--file', default="./", help='input file name')
    args = p.parse_args()
    #print args.file
    handleImage(args.file)

if __name__ == '__main__':
    main()
    #rotImage(FILENAME)
    cv.waitKey(0)
