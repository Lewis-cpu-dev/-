# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 20:20:43 2023

@author: Rzy
"""

import string
import cv2
import time
import numpy as np
from random import randint

# Inside function
def getKeypoints(probMap, threshold=0.1):
    
    mapSmooth = cv2.GaussianBlur(probMap, (3,3), 0, 0)
    mapMask = np.uint8(mapSmooth > threshold)
    keypoints = []

    #find the blobs
    contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   #find profile

    #for each blob find the maxima
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)      #draw
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))  #坐标
        keypoints_temp = keypoints
        
        
    print('===========keypoints========{} \n\n\n'.format(keypoints))
    
    return keypoints



# Find valid connections between the different joints of a all persons present
def getValidPairs(output, mapIdx, frameWidth, frameHeight, POSE_PAIRS, detected_keypoints):

    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    
    # loop for every POSE_PAIR
    for k in range(len(mapIdx)):
        # A->B constitute a limb
        pafA = output[0, mapIdx[k][0], :, :]
        pafB = output[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))

        # Find the keypoints for the first and second limb
        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)


        # If keypoints for the joint-pair is detected
        # check every joint in candA with every joint in candB
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between the joints
        # Use the above formula to compute a score to mark the connection valid


        if( nA != 0 and nB != 0):
            valid_pair = np.zeros((0,3))
            for i in range(nA):
                max_j=-1
                maxScore = -1
                found = 0
                for j in range(nB):
                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])  #minus of matrix
                    norm = np.linalg.norm(d_ij)   #default 2 norm
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                        
                    # Find p(u)
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),   
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples))) 

                    # Find L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ])

                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)

                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                    if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th : 
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1

                # Append the connection to the list
                if found:
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)



            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        else: # If no keypoints are detected            
            print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
            
    print('=============valied-pairs======={}'.format(valid_pairs))
    print('==============invalid-pairs========={}  \n\n\n'.format(invalid_pairs))
    return valid_pairs, invalid_pairs



# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
def getPersonwiseKeypoints(valid_pairs, invalid_pairs, mapIdx, POSE_PAIRS, keypoints_list):
    threshold=5
    # the last number in each row is the overall score
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]
                    print('found in i:{} j:{} k:{} indexA:{} B:{}'.format(str(i),str(j),str(k),indexA, indexB))
                    print(partAs,partBs)
                    print(personwiseKeypoints)

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score

                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
                    print('not found')
                    print(personwiseKeypoints)

    x=[]  
    for n in range(len(personwiseKeypoints)):
        if personwiseKeypoints[n][-1] < threshold:
            x.append(n)
    personwiseKeypoints = np.delete(personwiseKeypoints,x,0)
    print('===========personwisekeypoints=========={} \n\n\n'.format(personwiseKeypoints))
    return personwiseKeypoints


# Outside function
def humanPoseDetector(img):
    """
    input: one image(contain just one person) to detect the human pose
    output: the image whose size is changed and pose is drawed and the location of keypoints that are detected 
            and the valied pairs
    """
    # read model
    protoFile = "./weights/pose_deploy_linevec.prototxt"
    weightsFile = "./weights/pose_iter_440000.caffemodel"

    nPoints = 18

    # COCO Output Format
    keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 
                    'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']
    # pair
    POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
                  [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
                  [1,0], [0,14], [14,16], [0,15], [15,17],
                  [2,17], [5,16] ]


    # index of pafs correspoding to the POSE_PAIRS
    # e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
    mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
              [19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
              [47,48], [49,50], [53,54], [51,52], [55,56],
              [37,38], [45,46]]


    colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
             [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
             [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]
    
    
    frameWidth = img.shape[1]
    frameHeight = img.shape[0]

    t = time.time()
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    # change height
    inHeight = 368
    inWidth = int((inHeight/frameHeight)*frameWidth)

    inpBlob = cv2.dnn.blobFromImage(img, 1.0 / 255, (inWidth, inHeight),
                          (0, 0, 0), swapRB=False, crop=False)

    # forward
    net.setInput(inpBlob)
    output = net.forward()

    print(len(output))
    print("Time Taken in forward pass = {}".format(time.time() - t))

    detected_keypoints = []
    keypoints_list = np.zeros((0,3))
    keypoint_id = 0
    threshold = 0.1
    keypoints_location = []


    for part in range(nPoints):
        probMap = output[0,part,:,:]
        probMap = cv2.resize(probMap, (img.shape[1], img.shape[0]))
        keypoints = getKeypoints(probMap, threshold)      
        #keypoints_temp = list(keypoints[0])
        if keypoints != []:
            keypoints_temp = list(keypoints[0])
            keypoints_temp[2] = part
            keypoints_location.append(keypoints_temp)                  
        else:
            keypoints_location.append(keypoints)           
        
        print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
        keypoints_with_id = []
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
            keypoint_id += 1
            
        detected_keypoints.append(keypoints_with_id)

    keypointsImg = img.copy()

    valid_pairs, invalid_pairs = getValidPairs(output, mapIdx, frameWidth, frameHeight, POSE_PAIRS, detected_keypoints)
    personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, mapIdx, POSE_PAIRS, keypoints_list)
    del_personwiseKeypoints = np.delete(personwiseKeypoints,-1,1)
    
    
    personlocation=[]
    
    for n in range(len(del_personwiseKeypoints)):
        personlocation.append([])
        index_i = 0
        for i in del_personwiseKeypoints[n]:
            if i != -1 :
                for k in range(len(detected_keypoints)):
                    for m in range(len(detected_keypoints[k])):
                        info_points = detected_keypoints[k][m]
                        if i == info_points[-1]:
                            cv2.circle(keypointsImg, (info_points[0],info_points[1]), 5, colors[index_i], -1, cv2.LINE_AA)
                            personlocation[n].append([info_points[0],info_points[1],index_i])
            index_i += 1
    print(personlocation)
    
    lineImg = keypointsImg.copy()
                            
    for i in range(17):
        for n in range(len(personwiseKeypoints)):
            index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
            if -1 in index:
                continue

            B = np.int32(keypoints_list[index.astype(int), 0])
            A = np.int32(keypoints_list[index.astype(int), 1])
            #cv2.circle(keypointsImg, (keypoints_location[i][0],keypoints_location[i][1]), 5, colors[i], -1, cv2.LINE_AA)
            cv2.line(lineImg, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)

    # 添加计时
    t, _ = net.getPerfProfile()
    freq = cv2.getTickFrequency() / 1000
    # cv2.putText(lineImg, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    
    return keypointsImg, lineImg, keypoints_location, valid_pairs,invalid_pairs,personwiseKeypoints, keypoints_list,detected_keypoints
    # return keypointsImg, lineImg, keypoints_location, valid_pairs

    #cv2.imshow("Detected Pose" , frameClone)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()



PATH1 = 'C:/Users/Rzy/Desktop/safety_dataset/train/1 (2).MOV'+'/00000010.jpg'
PATH2 = 'C:/Users/Rzy/Desktop/safety_dataset/train/2 (2).MOV'+'/00000010.jpg'
PATH3 = 'C:/Users/Rzy/Desktop/a.jpg'
# used to fine the picture or use the inside camera
img = cv2.imread(PATH3)

keypointsImg, lineImg, keypoints_location, valid_pairs,invalid_pairs,personwiseKeypoints,keypoints_list,detected_keypoints = humanPoseDetector(img)

print(keypoints_location,'\n')
print(valid_pairs)

cv2.namedWindow('1', cv2.WINDOW_NORMAL)
cv2.namedWindow('2', cv2.WINDOW_NORMAL)
cv2.imshow('1', keypointsImg)
cv2.imshow('2', lineImg)
cv2.waitKey(0)
cv2.destroyAllWindows()


"""
cv2.imwrite('output/test_keypoints.jpg', imgClone)
cv2.imwrite('output/test_out.jpg', imgClone_new)
"""
