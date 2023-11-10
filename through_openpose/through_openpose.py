    # -*- coding: gbk -*-
# �򵥵Ľ��м��һ��ͼ����������˵���̬


from HumanPoseDetecte import humanPoseDetector

import cv2


PATH1 = 'C:/Users/Rzy/Desktop/safety_dataset/train/1 (2).MOV'+'/00000010.jpg'
PATH2 = 'C:/Users/Rzy/Desktop/safety_dataset/train/2 (2).MOV'+'/00000010.jpg'
img = cv2.imread(PATH1)

keypointsImg, lineImg, keypoints_location, valid_pairs,personwiseKeypoints,keypoints_list,detected_keypoints = humanPoseDetector(img)

print(keypoints_location,'\n')
print(valid_pairs)


cv2.imshow('1', keypointsImg)
cv2.imshow('2', lineImg)
cv2.waitKey(0)
cv2.destroyAllWindows()


"""
cv2.imwrite('output/test_keypoints.jpg', imgClone)
cv2.imwrite('output/test_out.jpg', imgClone_new)
"""
