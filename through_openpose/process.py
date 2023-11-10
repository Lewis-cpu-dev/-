# -*- coding: gbk -*-


import numpy as np
import cv2



class imgprocession(object):
    

    def __init__(self,filename):
        self.img = cv2.imread(filename, 1)
        


    def gaussian_blur(self,img):
        self.gaussian_img = cv2.GaussianBlur(img, (5, 5), 0.8)
        cv2.imshow('gaussian_img',self.gaussian_img)
        return self.gaussian_img


    def gray_procession(self,img):
        self.gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray_img',self.gray_img)
        return self.gray_img


    def threshold_procession(self, img, threshold):
        '''self.threshold_img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)'''
        self.threshold_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        #cv2.imshow('threshold', self.threshold_img)
        return self.threshold_img


    def draw_shape(self, img1, img2):
        self.contours, self.hierarchy = cv2.findContours(img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        self.img2 = cv2.drawContours(img2, self.contours, -1, (0, 0, 255), 3)
        cv2.namedWindow('final_img', 1)
        cv2.imshow('final_img', self.img2)
        cv2.waitKey()


    def process(self):
        '''file_path = 'C:/Users/Rzy/Desktop/safety_dataset/train/1 (2).MOV'
        file_n = '00000010.jpg'
        filename = file_path + '/' +  file_n '''
        
        gaussian_img = self.gaussian_blur(self.img)
        gray_img = self.gray_procession(gaussian_img)
        threshold_img = self.threshold_procession(gray_img, 100)
        self.draw_shape(threshold_img, self.img)
    




