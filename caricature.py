# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 23:19:21 2021

@author: Dell
"""

import dlib
import glob
import numpy as np
from skimage import io
import cv2
from PIL import Image,ImageFilter
from subprocess import Popen, PIPE
import math
import os

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 36))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                    RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

points=[]
m_index=0
class Landmarks:
    def __init__(self):
        self.points_list=None
        self.aligned_img=[]

    def SparsePoints(self,images):
            detect = dlib.get_frontal_face_detector()
            predict = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
            #avg = np.zeros((68,2))
            #pt_avg = np.zeros((76,2))
            #images=doCropping(theImage1,theImage2)
            
            #images=self.resize_img(image)
            len_img = len(images)
            
            list_pt=[]
            for l in range(len_img):
                list_pt.append([])
            index=0
            for img in images:
                details = detect(img,1)
                
                det=[]
                if len(details)==0:
                    print("Face not found")
                    pass
                elif len(details)==1:
                    det.append(details[0])
                elif len(details)>1:
                    details = self.prompt_user_to_choose_face(img,details)
                    det.append(details)

                s0=img.shape[0]
                s1=img.shape[1]
                for d in det:
                    l=d.left()
                    r=d.right()
                    t=d.top()
                    b=d.bottom()
                    shapes = predict(img,d)
                    for i in range(0,68):
                        x = shapes.part(i).x
                        y = shapes.part(i).y
                        list_pt[index].append([x,y])    
                index+=1
            self.points_list = list_pt
            return self.points_list
    
    def prompt_user_to_choose_face(self,im, rects):
        im = im.copy()
        h, w = im.shape[:2]
        for i in range(len(rects)):
            d = rects[i]
            x1, y1, x2, y2 = d.left(), d.top(), d.right()+1, d.bottom()+1
            cv2.rectangle(im, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=5)
            cv2.putText(im, str(i), (d.center().x, d.center().y),
                        fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        fontScale=1.5,
                        color=(255, 255, 255),
                        thickness=5)
    
        DISPLAY_HEIGHT = 650
        resized = cv2.resize(im, (int(w * DISPLAY_HEIGHT / float(h)), DISPLAY_HEIGHT))
        cv2.imshow("Multiple faces", resized); cv2.waitKey(1)
        target_index = int(input("Please choose the index of the target face: "))
        cv2.destroyAllWindows(); cv2.waitKey(1)
        return rects[target_index]

    def transformation_from_points(self,points1, points2):
        points1 = points1.astype(np.float64)
        points2 = points2.astype(np.float64)
        
        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2
    
        s1 = np.std(points1)
        s2 = np.std(points2)
        points1 /= s1
        points2 /= s2
    
        U, S, Vt = np.linalg.svd(np.dot(points1.T,points2))
        R = (U * Vt).T
        
        return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)),
                          np.matrix([0., 0., 1.])])
    
    def align_images(self,index, index1, im1, im2, border, prev=None):
        
        landmarks1 = self.points_list[index1]
        landmarks2 = self.points_list[index]
        
        landmarks1 = np.matrix(landmarks1)
        landmarks2 = np.matrix(landmarks2)
    
        T = self.transformation_from_points(landmarks1[ALIGN_POINTS],landmarks2[ALIGN_POINTS])
        
        M = cv2.invertAffineTransform(T[:2])
        
        if border is not None:
            im2 = cv2.copyMakeBorder(im2, border, border, border, border, 
                borderType=cv2.BORDER_CONSTANT, value=(255,255,255))
        warped_im2 = self.warp_im(im2, M, im1, im1.shape, prev)
    
        self.aligned_img.append(warped_im2)
        #cv2.imwrite(os.path.join(r'F:\ML\Image_Color\Morphing\Tank' +r'\bas_'+str(index)+'.jpg'),warped_im2)
     
        #cv2.imshow("align",warped_im2)
        #cv2.waitKey(0)
        return warped_im2

    def warp_im(self,im, M, im1, dshape, prev):
        output_im = cv2.warpAffine(
            im, M, (dshape[1], dshape[0]),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,#cv2.BORDER_REFLECT_101 if prev is not None else cv2.BORDER_CONSTANT,
        )
        
        if prev is not None:
            
            # overlay the image on the previous image  
            mask = cv2.warpAffine(
            np.ones_like(im, dtype='float32'), M, (dshape[1], dshape[0]), flags=cv2.INTER_AREA,)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = mask.astype(np.uint8)
            masked_image=cv2.bitwise_and(prev, prev, mask=(1-mask))
            
            masked_image1=cv2.bitwise_and(output_im, output_im, mask=mask)
            output_im = masked_image1+masked_image
          
        #cv2.imshow("output2",output_im)
        #cv2.waitKey(0)
        return output_im
    
    def get_boundary_points(self,shape):
        h, w = shape[:2]
        boundary_pts = [
            (1,1), (w-1,1), (1, h-1), (w-1,h-1), 
            ((w-1)//2,1), (1,(h-1)//2), ((w-1)//2,h-1), ((w-1)//2,(h-1)//2)
        ]
        return np.array(boundary_pts)
    
    
class Delaunay:
    def __init__(self,images,lists,avg):
        self.images=images
        self.length=len(images)
        #self.s0=s0
        #self.s1=s1
        self.array=avg
        self.points_l=lists
        #self.rect=[0,0,s1,s0]

    def makeDel(self,i_one,i_two):
        im1 = self.images[i_one]
        im2 = self.images[i_two]
        pt1 = self.points_l[i_one]
        pt2 = self.points_l[i_two]
        
        s1 = im2.shape[1]
        s0 = im2.shape[0]
        rect = [0,0,s1,s0]
        
        subdiv = cv2.Subdiv2D(rect)
        pts=[]
        
        #array = (pt1+pt2)/2
        array=self.array.tolist()
        
        for x in array:
            """
            if int(x[0]>s1) and int(x[1]>s0):
                pts.append((s1-1,s0-1))
            elif int(x[0]>s1) and int(x[1]<s0):
                pts.append((s1-1,int(x[1])))
            elif int(x[0]<s1) and int(x[1]>s0):
                pts.append((int(x[0]),s0-1))
            else:
                """
            pts.append((int(x[0]),int(x[1])))
                
        pts_list={pt[0]:pt[1] for pt in list(zip(pts,range(76)))}
        
        
        for pt in pts:
            subdiv.insert(pt)
            
        tri_pts = self.draw_triangles(subdiv,pts_list,self.images[i_two],rect)
        return tri_pts
        
    def check_Point(self,point,rect):
        if point[0]<rect[0]:
            return False
        elif point[0]>rect[2]:
            return False
        elif point[1]<rect[1]:
            return False
        elif point[1]>rect[3]:
            return False
        
        return True
    
    def draw_triangles(self,subdiv,points_l,image,rect):
        tri=[]
        triangles=subdiv.getTriangleList()
        
        for t in triangles:
            point1 = (int(t[0]),int(t[1]))
            point2 = (int(t[2]),int(t[3]))
            point3 = (int(t[4]),int(t[5]))
            if self.check_Point(point1,rect) and self.check_Point(point2,rect) and self.check_Point(point3,rect):
                tri.append((points_l[point1],points_l[point2],points_l[point3]))
                #cv2.line(image, point1, point2, (0, 255, 0), thickness=2)
                #cv2.line(image, point2, point3, (0, 255, 0), thickness=2)    
                #cv2.line(image, point3, point1, (0, 255, 0), thickness=2)    
        #point={}
        #cv2.imshow("tri",image)
        #cv2.waitKey(0)
        return tri


class Morph:
    def __init__(self,s0,s1,images,img_pt,tri_pt):
        self.s0=s0
        self.s1=s1
        self.images=images
        self.img_pt=img_pt
        self.tri_pt=tri_pt
        self.alpha=0.5

    def generate_morph(self,duration,frame_rate,transition_rate,video_path):
        total=int(duration*frame_rate)
        #p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-r', str(frame_rate),'-s',str(self.s1)+'x'+str(self.s0), '-i', '-', '-c:v', 'libx264', '-crf', '25','-vf','scale=trunc(iw/2)*2:trunc(ih/2)*2','-pix_fmt','yuv420p', video_path], stdin=PIPE)
        first_im = cv2.cvtColor(self.images[0], cv2.COLOR_BGR2RGB)
        command = ['ffmpeg', 
        '-y', 
        '-f', 'image2pipe', 
        '-r', str(frame_rate), 
        '-s', str(self.s1) + 'x' + str(self.s0), 
        '-i', '-',
        '-c:v', 'libx264', 
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', 
        '-pix_fmt', 'yuv420p', 
        video_path
        ]         

        #p = Popen(command, stdin=PIPE)
        #self.fill_frames(Image.fromarray(first_im), total, p)
        for i in range(len(self.images)-1):
            last = self.morphing(i,i+1,transition_rate, frame_rate, video_path)
            #self.fill_frames(last, total, p)
        #p.stdin.close()
        #p.wait()
        
    def morphing(self,ind1,ind2,duration,frame_rate,video_path):
        total = int(duration*frame_rate)
        for i in range(total):
            img_convert=[]
            
            img1 = self.images[ind1]
            img2 = self.images[ind2]
            
            img1 = np.float32(img1)
            img2 = np.float32(img2)
                        
            #points = [];
            self.alpha=i/(total-1)
            # weighted average point coordinates
            for c in range(0, len(self.img_pt[ind1])):
                x = ( 1 - self.alpha ) * self.img_pt[ind1][c][0] + self.alpha * self.img_pt[ind2][c][0]
                y = ( 1 - self.alpha ) * self.img_pt[ind1][c][1] + self.alpha * self.img_pt[ind2][c][1]
                points.append((x,y))
    
            # Allocate space for final output
            area = np.zeros(img1.shape, dtype = img1.dtype)
            # delaunay triangle points
            for i in range(len(self.tri_pt[ind1])):    
                x = int(self.tri_pt[ind1][i][0])
                y = int(self.tri_pt[ind1][i][1])
                z = int(self.tri_pt[ind1][i][2])
                t1 = [self.img_pt[ind1][x], self.img_pt[ind1][y], self.img_pt[ind1][z]]
                t2 = [self.img_pt[ind2][x], self.img_pt[ind2][y], self.img_pt[ind2][z]]
                t = [ points[x], points[y], points[z] ]
                # Morph one triangle at a time.
                self.triangle_morphing(img1,img2,area, t1, t2, t)
                        
            temp_res=cv2.cvtColor(np.uint8(area),cv2.COLOR_BGR2RGB)
            res=Image.fromarray(temp_res)
            #res.save(p.stdin,'JPEG')
            cv2.imwrite(os.path.join(r'/Users/nimishamittal/Documents/OldProjects/ML/Image_Color/Morphing/result' +r'/caricature5.jpg'),cv2.cvtColor(temp_res,cv2.COLOR_RGB2BGR))
        return Image.fromarray(cv2.cvtColor(self.images[ind2], cv2.COLOR_BGR2RGB))
    
    def fill_frames(self,img,frames,p):
        for _ in range(frames):
            img.save(p.stdin, 'JPEG')

    def triangle_morphing(self,im1,im2,area,t1,t2,t):
        # bounding rectangle
        a1 = cv2.boundingRect(np.float32([t1]))
        a2 = cv2.boundingRect(np.float32([t2]))
        a = cv2.boundingRect(np.float32([t]))
    
        t1Area = []
        t2Area = []
        tArea = []
        
        for i in range(0, 3):
            tArea.append(((t[i][0] - a[0]),(t[i][1] - a[1])))
            t1Area.append(((t1[i][0] - a1[0]),(t1[i][1] - a1[1])))
            t2Area.append(((t2[i][0] - a2[0]),(t2[i][1] - a2[1])))
    
    
        # Masking to fetch triangles
        mask = np.zeros((a[3], a[2], 3), dtype = np.float32)
        
        cv2.fillConvexPoly(mask, np.int32(tArea), (1.0, 1.0, 1.0), 16, 0);
        
        # Apply warping technique to small areas
        img1 = im1[a1[1]:a1[1] + a1[3], a1[0]:a1[0] + a1[2]]
        img2 = im2[a2[1]:a2[1] + a2[3], a2[0]:a2[0] + a2[2]]
        size = (a[2], a[3])
        #Applying affine transformation to the images
        warpImage1 = self.AffineTransform(img1, t1Area, tArea, size)
        warpImage2 = self.AffineTransform(img2, t2Area, tArea, size)
        
        # Alpha blend rectangular patches
        imgRect = (1.0 - self.alpha) * warpImage1 + self.alpha * warpImage2
        area[a[1]:a[1]+a[3], a[0]:a[0]+a[2]] = area[a[1]:a[1]+a[3], a[0]:a[0]+a[2]] * ( 1 - mask ) + imgRect * mask
        
    def AffineTransform(self,img,t1A,tA,size):
        warp = cv2.getAffineTransform( np.float32(t1A), np.float32(tA) )
        
        # Apply the Affine Transform
        res = cv2.warpAffine( img, warp, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
    
        return res


read_img = [cv2.imread(r'/Users/nimishamittal/Documents/OldProjects/ML/Image_Color/Morphing/bean.jpg',cv2.IMREAD_COLOR),cv2.imread(r'/Users/nimishamittal/Documents/OldProjects/ML/Image_Color/Morphing/mona_lisa.jpg',cv2.IMREAD_COLOR)]

prev=None
c = Landmarks()
points_sp = c.SparsePoints(read_img)
index=0
t_index = len(read_img)-1
target = read_img[t_index]
#cv2.imread(r'F:\ML\Image_Color\Morphing\Tank\tank_10.jpg')#read_img[len(read_img)-2]
aligned_im=[]
for img in read_img:
    prev=c.align_images(index,t_index,target,img, 5,None)
    index+=1
  
aligned = c.aligned_img
pts = c.SparsePoints(aligned)
index=0
pt_avg = np.zeros((76,2))
for img in aligned:
    #cv2.imshow("ori_a",img)
    #cv2.waitKey(0)
    #a=c.align_images(index,aligned[len(aligned)-1],img, 5,None)
    pts[index] = np.append(pts[index], c.get_boundary_points(img.shape), axis=0)
    pt_avg += pts[index]
    index+=1
pt_avg = np.divide(pt_avg,len(aligned))

delau = Delaunay(aligned,pts,pt_avg)
del_pt=[]
for i in range(len(aligned)-1):
    del_pt.append(delau.makeDel(i, i+1))


morph = Morph(aligned[0].shape[0],aligned[0].shape[1],aligned,pts,del_pt)
morph.generate_morph(1.4,25,3,'kun.mp4')

