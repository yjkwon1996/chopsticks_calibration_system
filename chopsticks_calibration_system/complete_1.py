# MPII를 사용한 신체부위 검출
import cv2
import os
import ast
import math
import numpy as np
import json
import sys

# MPII에서 각 파트 번호, 선으로 연결될 POSE_PAIRS
"""
BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                "Background": 15 }

POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]
    
"""

HAND_PARTS = { "A": 0, "B": 1, "C": 2, "D": 3, "E": 4,
                "F": 5, "G": 6, "H": 7, "I": 8, "J": 9,
                "K": 10, "L": 11, "N": 12, "M": 13, "O": 14,
                "P": 15, "Q": 16, "R": 17, "S": 18, "T": 19, "U": 20 }

HAND_PAIRS = [ ["A", "B"], ["B", "C"], ["C", "D"], ["D", "E"],
                ["F", "G"], ["G", "H"], ["H", "I"],
                ["J", "K"], ["K", "L"], ["L", "N"],
                ["M", "O"], ["O", "P"], ["P", "Q"],
                ["R", "S"], ["S", "T"], ["T", "U"],
                ["A", "F"], ["A", "J"], ["A", "M"], ["A", "R"] ]

# 각 파일 path
protoFile = "C:\\Users\\admin\\Desktop\\openpose\\pose_deploy.prototxt"
weightsFile = "C:\\Users\\admin\\Desktop\\openpose\\pose_iter_102000.caffemodel"
 
# 위의 path에 있는 network 불러오기
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# 이미지 읽어오기
FILE_PATH = "C:\\Users\\admin\\Desktop\\openpose\\complete1\\wrong\\007.jpg"
image = cv2.imread(FILE_PATH)
src = cv2.imread(FILE_PATH, cv2.IMREAD_GRAYSCALE)

# frame.shape = 불러온 이미지에서 height, width, color 받아옴
imageHeight, imageWidth, _ = image.shape

# network에 넣기위해 전처리
inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)

# network에 넣어주기
net.setInput(inpBlob)

# 결과 받아오기
output = net.forward()

# output.shape[0] = 이미지 ID, [1] =  출력 맵의 높이, [2] = 너비
H = output.shape[2]
W = output.shape[3]
# print("이미지 ID : ", len(output[0]), ", H : ", output.shape[2], ", W : ",output.shape[3]) # 이미지 ID

# 키포인트 검출시 이미지에 그려줌
# 1~4 엄지, 5~8 검지, 9~12 중지, 13~16 소지, 17~20 약지
points = []

for i in range(0,21):
    # 해당 신체부위 신뢰도 얻음.
    probMap = output[0, i, :, :]
 
    # global 최대값 찾기
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    # 원래 이미지에 맞게 점 위치 변경
    x = (imageWidth * point[0]) / W
    y = (imageHeight * point[1]) / H 
    
    # 키포인트 x좌표, y좌표 검출하기
    # print(i, "좌표 - x : ", x, " y : ", y)
    

    # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로    
    if prob > 0.1 :    
        points.append((int(x), int(y)))
    else :
        # x좌표, y좌표값이 검출되지 않으면(이미지가 올바른 이미지가 아닌 경우)
        print("이미지가 잘못되었습니다. 다시 촬영해 주세요")
        sys.exit();

# x좌표, y좌표 분리
x = [cox[0] for cox in points]
y = [coy[1] for coy in points]

# opencv 좌표는 왼쪽 상단이 (0, 0)
# 0번 키포인트
cv2.circle(image, (int(x[0]), int(y[0])), 3, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
cv2.putText(image, "{}".format(0), (int(x[0]), int(y[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, lineType=cv2.LINE_AA)

# 이미지 복사
image1 = image.copy()
image2 = image.copy()
image3 = image.copy()
image4 = image.copy()
image5 = image.copy()
imageCopy = image.copy()



# 검출한 좌표를 올바른 젓가락질의 좌표와 비교
# 예시로서 하나의 이미지의 x, y좌표를 txt파일에서 읽어들인 뒤 비교
comp_file = "C:\\Users\\admin\\Desktop\\openpose\\complete1\\right\\right_point.txt"
with open(comp_file, 'r') as f:
    readfile = f.readline()
    locate = ast.literal_eval(readfile)


# 읽어들인 좌표를 이미지에 그려줌
for i in range(0,21):

    # x좌표, y좌표 분리
    comp_x = [cox[0] for cox in locate]
    comp_y = [coy[1] for coy in locate]


# 각각의 좌표의 각도를 구함(좌측 상단, )
degree = []
right_degree = []
def deg(i) :
    tmp1 = math.degrees(math.atan2(y[i]-y[0], x[i]-x[0]))
    tmp2 = math.degrees(math.atan2(comp_y[i]-comp_y[0], comp_x[i]-comp_x[0]))
    if tmp1 < 0 :
        tmp1 = abs(tmp1)
    else :
        tmp1 = 180 + (180 - tmp1)
    if tmp2 < 0 :
        tmp2 = abs(tmp2)
    else :
        tmp2 = 180 + (180 - tmp2)    
    degree.append(tmp1)
    right_degree.append(tmp2)
    for i in range(i, i+3) :
        tmp3 = math.degrees(math.atan2(y[i+1]-y[i], x[i+1]-x[i]))
        tmp4 = math.degrees(math.atan2(comp_y[i+1]-comp_y[i], comp_x[i+1]-comp_x[i]))
        if tmp3 < 0 :
            tmp3 = abs(tmp3)
        else :
            tmp3 = 180 + (180 - tmp3)
        if tmp4 < 0 :
            tmp4 = abs(tmp4)
        else :
            tmp4 = 180 + (180 - tmp4)    
        degree.append(tmp3)
        right_degree.append(tmp4)

deg(1)
deg(5)
deg(9)
deg(13)
deg(17)

diff = []
for i in range(0, 20) :
    diff.append(abs(degree[i] - right_degree[i]))

print("diff_sum : " , sum(diff))
print("정확도 : ", round(100 - (sum(diff)/7200*100), 2), "%" )
"""

# 좌표 맞춤을 위한 거리 계산
# 0번 키포인트를 중심으로 잡아서 계산한다.
move_x = x[0] - comp_x[0]
move_y = y[0] - comp_y[0]`
move_list = [move_x, move_y]
move_points = []
move_points_x = []
move_points_y = []
for i in range(0, 21):
    move_points_x.append(comp_x[i] + move_x)
    move_points_y.append(comp_y[i] + move_y)
    move_points.append((move_points_x[i], move_points_y[i]))
    
""" 
   
move_points = []
move_points_x = []
move_points_y = []
def mov_point(i) :
    move_x = x[i] - comp_x[i]
    move_y = y[i] - comp_y[i]
    print(move_x, move_y)

    for j in range(i, i+3):
        move_points_x.append(comp_x[j] + move_x)
        move_points_y.append(comp_y[j] + move_y)
        move_points.append((move_points_x[j], move_points_y[j]))
    return move_points, move_points_x, move_points_y
def append_mov(i) :
    move_points_x.append(comp_x[i] + x[i] - comp_x[i])
    move_points_y.append(comp_y[i] + y[i] - comp_y[i])
    move_points.append((move_points_x[i], move_points_y[i]))
    return move_points, move_points_x, move_points_y
    
append_mov(0)
# 키포인트를 얼마나 이동시킬지 계산
mov_point(1)
# 엄지
for i in range(1, 5):
    cv2.circle(image1, (int(x[i]), int(y[i])), 3, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
    cv2.putText(image1, "{}".format(i), (int(x[i]), int(y[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, lineType=cv2.LINE_AA)
    # 읽어들인 키포인트도 그림
    # cv2.circle(image1, (int(comp_x[i]), int(comp_y[i])), 3, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
    # cv2.putText(image1, "{}".format(i), (int(comp_x[i]), int(comp_y[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, lineType=cv2.LINE_AA)
    
# cv2.imshow("Output-Keypoints",image1)
# cv2.waitKey(0)

# 키포인트를 얼마나 이동시킬지 계산
append_mov(4)
append_mov(5)
mov_point(6)
# 검지
for i in range(6, 9):
    # cv2.circle(image2, (int(x[i]), int(y[i])), 3, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
    # cv2.putText(image2, "{}".format(i), (int(x[i]), int(y[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, lineType=cv2.LINE_AA)
    # 키포인트의 각도 차이가 심한 경우(잘못된 손가락 관절인 경우)
    if diff[i-1] > 30 :
        cv2.circle(image2, (int(x[i]), int(y[i])), 3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
        cv2.putText(image2, "{}".format(i), (int(x[i]), int(y[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
     # 키포인트 각도 차이가 약간 나는 경우
    elif diff[i-1] > 20 :
        cv2.circle(image2, (int(x[i]), int(y[i])), 3, (0, 127, 255), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
        cv2.putText(image2, "{}".format(i), (int(x[i]), int(y[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 127, 255), 1, lineType=cv2.LINE_AA)
    else :
        cv2.circle(image2, (int(x[i]), int(y[i])), 3, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
        cv2.putText(image2, "{}".format(i), (int(x[i]), int(y[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, lineType=cv2.LINE_AA)
    
    # 읽어들인 키포인트도 그림
    # cv2.circle(image2, (int(comp_x[i]), int(comp_y[i])), 3, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
    # cv2.putText(image2, "{}".format(i), (int(comp_x[i]), int(comp_y[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, lineType=cv2.LINE_AA)
    
    # 이동시킨 키포인트 그림
    cv2.circle(image2, (int(move_points_x[i]), int(move_points_y[i])), 3, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
    cv2.putText(image2, "{}".format(i), (int(move_points_x[i]), int(move_points_y[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, lineType=cv2.LINE_AA)
    
# cv2.imshow("Output-Keypoints",image2)
# cv2.waitKey(0)


# 키포인트를 얼마나 이동시킬지 계산
append_mov(9)
mov_point(10)
# 중지
for i in range(10, 13):
    # cv2.circle(image3, (int(x[i]), int(y[i])), 3, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
    # cv2.putText(image3, "{}".format(i), (int(x[i]), int(y[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, lineType=cv2.LINE_AA)
    # 키포인트의 각도 차이가 심한 경우(잘못된 손가락 관절인 경우)
    if diff[i-1] > 30 :
        cv2.circle(image3, (int(x[i]), int(y[i])), 3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
        cv2.putText(image3, "{}".format(i), (int(x[i]), int(y[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
     # 키포인트 각도 차이가 약간 나는 경우
    elif diff[i-1] > 20 :
        cv2.circle(image3, (int(x[i]), int(y[i])), 3, (0, 127, 255), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
        cv2.putText(image3, "{}".format(i), (int(x[i]), int(y[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 127, 255), 1, lineType=cv2.LINE_AA)
    else :
        cv2.circle(image3, (int(x[i]), int(y[i])), 3, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
        cv2.putText(image3, "{}".format(i), (int(x[i]), int(y[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, lineType=cv2.LINE_AA)
    
    
    # 읽어들인 키포인트도 그림
    # cv2.circle(image3, (int(comp_x[i]), int(comp_y[i])), 3, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
    # cv2.putText(image3, "{}".format(i), (int(comp_x[i]), int(comp_y[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, lineType=cv2.LINE_AA)
       
    # 이동시킨 키포인트 그림
    cv2.circle(image3, (int(move_points_x[i]), int(move_points_y[i])), 3, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
    cv2.putText(image3, "{}".format(i), (int(move_points_x[i]), int(move_points_y[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, lineType=cv2.LINE_AA)
    
# cv2.imshow("Output-Keypoints",image3)
# cv2.waitKey(0)

# 키포인트를 얼마나 이동시킬지 계산
append_mov(13)
mov_point(14)
# 약지
for i in range(14, 17):
    # cv2.circle(image4, (int(x[i]), int(y[i])), 3, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
    # cv2.putText(image4, "{}".format(i), (int(x[i]), int(y[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, lineType=cv2.LINE_AA)
    # 키포인트의 각도 차이가 심한 경우(잘못된 손가락 관절인 경우)
    if diff[i-1] > 30 :
        cv2.circle(image4, (int(x[i]), int(y[i])), 3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
        cv2.putText(image4, "{}".format(i), (int(x[i]), int(y[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 2555), 1, lineType=cv2.LINE_AA)
       # 키포인트 각도 차이가 약간 나는 경우
    elif diff[i-1] > 20 :
        cv2.circle(image4, (int(x[i]), int(y[i])), 3, (0, 127, 255), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
        cv2.putText(image4, "{}".format(i), (int(x[i]), int(y[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 127, 255), 1, lineType=cv2.LINE_AA)
    else :
        cv2.circle(image4, (int(x[i]), int(y[i])), 3, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
        cv2.putText(image4, "{}".format(i), (int(x[i]), int(y[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, lineType=cv2.LINE_AA)
    
    
    # 읽어들인 키포인트도 그림
    # cv2.circle(image4, (int comp_x[i]), int(comp_y[i])), 3, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
    # cv2.putText(image4, "{}".format(i), (int(comp_x[i]), int(comp_y[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, lineType=cv2.LINE_AA)
    
    # 이동시킨 키포인트 그림
    cv2.circle(image4, (int(move_points_x[i]), int(move_points_y[i])), 3, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
    cv2.putText(image4, "{}".format(i), (int(move_points_x[i]), int(move_points_y[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, lineType=cv2.LINE_AA)
    
# cv2.imshow("Output-Keypoints",image4)
# cv2.waitKey(0)

# 키포인트를 얼마나 이동시킬지 계산
append_mov(16)
mov_point(17)
# 소지
for i in range(17, 21):
    cv2.circle(image5, (int(x[i]), int(y[i])), 3, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
    cv2.putText(image5, "{}".format(i), (int(x[i]), int(y[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, lineType=cv2.LINE_AA)
    
    # 읽어들인 키포인트도 그림
    # cv2.circle(image5, (int(comp_x[i]), int(comp_y[i])), 3, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
    # cv2.putText(image5, "{}".format(i), (int(comp_x[i]), int(comp_y[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, lineType=cv2.LINE_AA)
    
# cv2.imshow("Output-Keypoints",image5)
# cv2.waitKey(0)



# 에지 검출
# 각 POSE_PAIRS별로 선 그어줌
for pair in HAND_PAIRS:
    partA = pair[0]             # 
    partA = HAND_PARTS[partA]   # 0
    partB = pair[1]             # 
    partB = HAND_PARTS[partB]   # 1
    
    
    
    #print(partA," 와 ", partB, " 연결\n")
    if points[partA] and points[partB]:
        cv2.line(imageCopy, points[partA], points[partB], (0, 255, 0), 2)


# test
edges1 = cv2.Canny(src, 450, 500)
lines1 = cv2.HoughLinesP(edges1, 1, np.pi / 180., 50, minLineLength=30, maxLineGap=2)

dst1 = src.copy()
dst1 = cv2.cvtColor(edges1, cv2.COLOR_GRAY2BGR)

stand_x1 = 100
stand_x2 = 600
stand_y1 = 100
stand_y2 = 750

straight1 = []
straight2 = []
if lines1 is not None: # 라인 정보를 받았으면
    for i in range(lines1.shape[0]):
        pt1 = (lines1[i][0][0], lines1[i][0][1]) # 시작점 좌표 x,y
        pt2 = (lines1[i][0][2], lines1[i][0][3]) # 끝점 좌표, 가운데는 무조건 0
        straight1.append(pt1)
        straight2.append(pt2)
        cv2.line(dst1, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

# x좌표, y좌표 분리
x1 = [cox[0] for cox in straight1] # 시작점 좌표
y1 = [coy[1] for coy in straight1]
x2 = [cox[0] for cox in straight2] # 끝점 좌표
y2 = [coy[1] for coy in straight2]


straight_x1 = []
straight_y1 = []
straight_x2 = []
straight_y2 = []
for i in range(len(x1)) :
    if x1[i] > stand_x1 and y1[i] > stand_y1 and x1[i] < stand_x2 and y1[i] < stand_y2 and \
        x2[i] > stand_x1 and y2[i] > stand_y1 and x2[i] < stand_x2 and y2[i] < stand_y2 :
        print("시작점 좌표 : ", x1[i], y1[i], "끝 점 좌표 : ", x2[i], y2[i])
        straight_x1.append(x1[i])
        straight_x2.append(x2[i])
        straight_y1.append(y1[i])
        straight_y2.append(y2[i])




# 시작점과 끝점의 좌표를 이용해서 기울기 계산
# m = (y2 - y1) / (x2 - x1)
m1 = (straight_y2[0] - straight_y1[0]) / (straight_x2[0] - straight_x1[0])
m2 = (straight_y2[1] - straight_y1[1]) / (straight_x2[1] - straight_x1[1])
tmp_count = 1

while m2 > 0 :
    tmp_count += 1
    m2 = (straight_y2[tmp_count] - straight_y1[tmp_count]) / (straight_x2[tmp_count] - straight_x1[tmp_count])
if abs(m2 - m1) < 0.05 :
    tmp_count += 1
    m2 = (straight_y2[tmp_count] - straight_y1[tmp_count]) / (straight_x2[tmp_count] - straight_x1[tmp_count])


# x좌표를 임의로 설정
tmp = 480

# 위에서 정한 x좌표에 해당하는 y좌표를 발견 : y = mx + b
# 점 (x1, y1)을 지나고 기울기가 m인 직선의 방정식 : y - y1 = m(x - x1)
# x값에 tmp를 대입
result_y1 = int(m1 * (tmp - straight_x1[0]) + straight_y1[0])
result_y2 = int(m2 * (tmp - straight_x1[tmp_count]) + straight_y1[tmp_count])

# tmp값은 x값, result_y값은 y값으로 검출된 직선을 길게 해서 젓가락 검출
cv2.line(imageCopy, (straight_x1[0], straight_y1[0]), (tmp, result_y1), (0, 0, 255), 1, cv2.LINE_AA)
cv2.putText(imageCopy, "{}".format(1), (straight_x1[0], straight_y1[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
cv2.line(imageCopy, (straight_x1[tmp_count], straight_y1[tmp_count]), (tmp, result_y2), (0, 0, 255), 1, cv2.LINE_AA)
cv2.putText(imageCopy, "{}".format(2), (straight_x1[tmp_count], straight_y1[tmp_count]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)



    

# 신경 써야 하는 키포인트는 6, 7, 8(검지)   10, 11, 12(중지)   14, 15, 16(약지)
# 직선을 구했으니, 직선이 어느 손가락 위에 있는지 구해야한다.
# 직선과 손가락을 같은 x좌표를 두고, y좌표를 비교하여 어느 손가락 사이에 있는지 비교할 수 있다.
# 0번 키포인트를 제외한 20개의 키포인트를 비교. y좌표가 손가락보다 위인지 아래인지만 확인하면된다.
# x를 대입해서 y를 비교, 위에 있을 경우 1, 아래에 있을 경우 0
# 젓가락은 2개이므로
chopstick1 = []
chopstick2 = []
def chop_loc(i) :
    res_y1 = int(m1 * (x[i] - straight_x1[0]) + straight_y1[0])
    res_y2 = int(m2 * (x[i] - straight_x1[tmp_count]) + straight_y1[tmp_count])
    if res_y1 > y[i] :
        chopstick1.append(0)
    else : chopstick1.append(1)
    if res_y2 > y[i] :
        chopstick2.append(0)
    else : chopstick2.append(1)

for i in range(0, 21) :
    chop_loc(i)

# 8번 검지, 12번 중지, 16번 약지
# 아래에 있을 때 0, 위에 있을 때 1
# 하나의 젓가락은 중지와 약지 사이에 있어야 하고,
# 다른 하나의 젓가락은 검지와 중지 사이에 있어야 한다.
cho1 = 0
cho2 = 0
if chopstick1[8] == 1:
    print("1번 젓가락이 검지 위에 있습니다.")
elif chopstick1[12] == 1:
    print("1번 젓가락이 검지와 중지 사이에 있습니다.")
    cho1 = 1
elif chopstick1[16] == 1:
    print("1번 젓가락이 중지와 약지 사이에 있습니다.")
    cho1 = 2
else : print("젓가락 위치를 알 수 없습니다.")

if chopstick2[8] == 1:
    print("2번 젓가락이 검지 위에 있습니다.")
elif chopstick2[12] == 1:
    print("2번 젓가락이 검지와 중지 사이에 있습니다.")
    cho2 = 1
elif chopstick2[16] == 1:
    print("2번 젓가락이 중지와 약지 사이에 있습니다.")
    cho2 = 2
else : print("젓가락 위치를 알 수 없습니다.")
if (cho1 == 1 and cho2 == 2) or (cho1 == 2 and cho2 == 1) :
    print("올바른 젓가락 위치입니다.")
else : 
    print("잘못된 젓가락 위치입니다.")
    print("검지와 중지로 젓가락 하나를 잡고, 약지로 젓가락 하나를 받쳐야 합니다.")

    
"""
dst1 = dst1[stand_y1:stand_y2, stand_x1:stand_x2]  

cv2.imshow('dst1', dst1)
cv2.waitKey()
"""

cv2.imshow("Connected Output-Keypoints",imageCopy)
cv2.waitKey(0)




# 검출한 x좌표, y좌표를 txt파일로 저장
filename = os.path.splitext(FILE_PATH)
path_file = filename[0] + '.txt.'
loc = str(points)
with open(path_file, 'w') as f:
    f.writelines(loc)

# 검출한 좌표, 각를 json 으로 저장
file_data = {
    'loc' : points, 
    'deg' : degree
    }
path_json = filename[0] + '.json'
with open(path_json, 'w', encoding='utf-8') as make_file :
    json.dump(file_data, make_file, indent='\t')



    
"""    
    
# 검출한 좌표를 올바른 젓가락질의 좌표와 비교
# 예시로서 하나의 이미지의 x, y좌표를 txt파일에서 읽어들인 뒤 비교
comp_file = "C:\\Users\\admin\\Desktop\\openpose\\img\\comp_loc\\002.txt"
with open(comp_file, 'r') as f:
    readfile = f.readline()
    locate = ast.literal_eval(readfile)


# 읽어들인 좌표를 이미지에 그려줌
for i in range(0,21):

    # x좌표, y좌표 분리
    comp_x = [cox[0] for cox in locate]
    comp_y = [coy[1] for coy in locate]

    cv2.circle(image, (int(comp_x[i]), int(comp_y[i])), 3, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
    cv2.putText(image, "{}".format(i), (int(comp_x[i]), int(comp_y[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, lineType=cv2.LINE_AA)
    
cv2.imshow("Output-Keypoints",image)
cv2.waitKey(0)

"""



# 이미지 복사
imageCopy1 = image1
imageCopy2 = image2
imageCopy3 = image3
imageCopy4 = image4
imageCopy5 = image5


dist = []
user_dist = []
# 구해야 하는 각 키포인트간의 거리
# 엄지 - 1-2, 2-3, 3-4
# 검지 - 5-6, 6-7, 7-8
# 중지 - 9-10, 10-11, 11-12
# 소지 - 13-14, 14-15, 15-16
# 약지 - 17-18, 18-19, 19-20
# 0번(손의 중심)과 연결 1, 5, 9, 13, 17
# 총 20개

FIRST_PAIRS = [ ["A", "B"], ["B", "C"], ["C", "D"], ["D", "E"] ]
SECOND_PAIRS = [ ["A", "F"], ["F", "G"], ["G", "H"], ["H", "I"] ]
THIRD_PAIRS = [ ["A", "J"], ["J", "K"], ["K", "L"], ["L", "N"] ]
FOURTH_PAIRS = [ ["A", "M"], ["M", "O"], ["O", "P"], ["P", "Q"] ]
FIFTH_PAIRS = [ ["A", "R"], ["R", "S"], ["S", "T"], ["T", "U"] ]


# 엄지 연결
for pair in FIRST_PAIRS:
    partA = pair[0]             
    partA = HAND_PARTS[partA]  
    partB = pair[1]            
    partB = HAND_PARTS[partB]  
    if points[partA] and points[partB]:
        cv2.line(imageCopy1, points[partA], points[partB], (255, 0, 0), 2)
         # 키포인트 간 거리를 구한다.
        dist.append(math.sqrt((x[partA]-x[partB])**2+(y[partA]-y[partB])**2))
        # 읽어들인 이미지에서 키포인트 거리 구하기
        user_dist.append(math.sqrt((comp_x[partA]-comp_x[partB])**2+(comp_y[partA]-comp_y[partB])**2))
    # 비교할 대상도 연결시켜줌
    # if locate[partA] and locate[partB]:
    #     cv2.line(imageCopy4, locate[partA], locate[partB], (0, 255, 0), 2)

# 검지 연결
for pair in SECOND_PAIRS:
    partA = pair[0]             
    partA = HAND_PARTS[partA]  
    partB = pair[1]            
    partB = HAND_PARTS[partB]  
    if points[partA] and points[partB]:
        if diff[partA] > 30 :
            cv2.line(imageCopy2, points[partA], points[partB], (0, 0, 255), 2)
        # 키포인트 각도 차이가 약간 나는 경우
        elif diff[partA] > 20 :
            cv2.line(imageCopy2, points[partA], points[partB], (0, 127, 255), 2)
        else :
            cv2.line(imageCopy2, points[partA], points[partB], (255, 0, 0), 2)
            # 키포인트 간 거리를 구한다.
        dist.append(math.sqrt((x[partA]-x[partB])**2+(y[partA]-y[partB])**2))
        # 읽어들인 이미지에서 키포인트 거리 구하기
        user_dist.append(math.sqrt((comp_x[partA]-comp_x[partB])**2+(comp_y[partA]-comp_y[partB])**2))
    # 비교할 대상도 연결시켜줌
    if locate[partA] and locate[partB]:
        cv2.line(imageCopy2, move_points[partA], move_points[partB], (0, 255, 0), 2)

# 중지 연결
for pair in THIRD_PAIRS:
    partA = pair[0]             
    partA = HAND_PARTS[partA]  
    partB = pair[1]            
    partB = HAND_PARTS[partB]  
    if points[partA] and points[partB]:
        if diff[partA] > 30 :
            cv2.line(imageCopy3, points[partA], points[partB], (0, 0, 255), 2)
        # 키포인트 각도 차이가 약간 나는 경우
        elif diff[partA] > 20 :
            cv2.line(imageCopy3, points[partA], points[partB], (0, 127, 255), 2)
        else :
            cv2.line(imageCopy3, points[partA], points[partB], (255, 0, 0), 2)
         # 키포인트 간 거리를 구한다.
        dist.append(math.sqrt((x[partA]-x[partB])**2+(y[partA]-y[partB])**2))
        # 읽어들인 이미지에서 키포인트 거리 구하기
        user_dist.append(math.sqrt((comp_x[partA]-comp_x[partB])**2+(comp_y[partA]-comp_y[partB])**2))
    # 비교할 대상도 연결시켜줌
    if locate[partA] and locate[partB]:
        cv2.line(imageCopy3, move_points[partA], move_points[partB], (0, 255, 0), 2)
        
# 약지 연결
for pair in FOURTH_PAIRS:
    partA = pair[0]             
    partA = HAND_PARTS[partA]  
    partB = pair[1]            
    partB = HAND_PARTS[partB]  
    if points[partA] and points[partB]:
        if diff[partA] > 30 :
            cv2.line(imageCopy4, points[partA], points[partB], (0, 0, 255), 2)
        # 키포인트 각도 차이가 약간 나는 경우
        elif diff[partA] > 20 :
            cv2.line(imageCopy4, points[partA], points[partB], (0, 127, 255), 2)
        else :
            cv2.line(imageCopy4, points[partA], points[partB], (255, 0, 0), 2)
         # 키포인트 간 거리를 구한다.
        dist.append(math.sqrt((x[partA]-x[partB])**2+(y[partA]-y[partB])**2))
        # 읽어들인 이미지에서 키포인트 거리 구하기
        user_dist.append(math.sqrt((comp_x[partA]-comp_x[partB])**2+(comp_y[partA]-comp_y[partB])**2))
    # 비교할 대상도 연결시켜줌
    if locate[partA] and locate[partB]:
        cv2.line(imageCopy4, move_points[partA], move_points[partB], (0, 255, 0), 2)
        
# 소지 연결
for pair in FIFTH_PAIRS:
    partA = pair[0]             
    partA = HAND_PARTS[partA]  
    partB = pair[1]            
    partB = HAND_PARTS[partB]  
    if points[partA] and points[partB]:
        cv2.line(imageCopy5, points[partA], points[partB], (255, 0, 0), 2)  
         # 키포인트 간 거리를 구한다.
        dist.append(math.sqrt((x[partA]-x[partB])**2+(y[partA]-y[partB])**2))
        # 읽어들인 이미지에서 키포인트 거리 구하기
        user_dist.append(math.sqrt((comp_x[partA]-comp_x[partB])**2+(comp_y[partA]-comp_y[partB])**2))
    # 비교할 대상도 연결시켜줌
    # if locate[partA] and locate[partB]:
    #     cv2.line(imageCopy4, locate[partA], locate[partB], (0, 255, 0), 2)
        
"""
# 각 POSE_PAIRS별로 선 그어줌
for pair in HAND_PAIRS:
    partA = pair[0]             # 
    partA = HAND_PARTS[partA]   # 0
    partB = pair[1]             # 
    partB = HAND_PARTS[partB]   # 1
    
    #print(partA," 와 ", partB, " 연결\n")
    if points[partA] and points[partB]:
        cv2.line(imageCopy, points[partA], points[partB], (0, 255, 0), 2)
        # 키포인트 간 거리를 구한다.
        dist.append(math.sqrt((comp_x[partA]-comp_x[partB])**2+(comp_y[partA]-comp_y[partB])**2))
        
    # 비교할 대상도 연결시켜줌
    if locate[partA] and locate[partB]:
        cv2.line(imageCopy, locate[partA], locate[partB], (255, 0, 0), 2)
"""


"""

# 허프 변환을 통한 직선 검출
edges = cv2.Canny(src, 50, 200)

lines = cv2.HoughLinesP(edges, 1, np.pi / 180., 160, minLineLength=150, maxLineGap=15)
dst = src.copy()
dst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
stand_x1 = 180
stand_x2 = 550
stand_y1 = 450
stand_y2 = 730
straight1 = []
straight2 = []
if lines is not None: # 라인 정보를 받았으면
    for i in range(lines.shape[0]):
        pt1 = (lines[i][0][0], lines[i][0][1]) # 시작점 좌표 x,y
        pt2 = (lines[i][0][2], lines[i][0][3]) # 끝점 좌표, 가운데는 무조건 0
        straight1.append(pt1)
        straight2.append(pt2)
        cv2.line(dst, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)
        
dst = dst[stand_y1:stand_y2, stand_x1:stand_x2]  
cv2.imshow('dst', dst)
cv2.waitKey()

"""



# cv2.imshow("Connected Output-Keypoints",imageCopy1)
# cv2.waitKey(0)
cv2.imshow("Connected Output-Keypoints",imageCopy2)
cv2.waitKey(0)
cv2.imshow("Connected Output-Keypoints",imageCopy3)
cv2.waitKey(0)
cv2.imshow("Connected Output-Keypoints",imageCopy4)
cv2.waitKey(0)
# cv2.imshow("Connected Output-Keypoints",imageCopy5)
# cv2.waitKey(0)



cv2.destroyAllWindows()

