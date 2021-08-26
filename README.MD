# Chopsticks Calibration System
### 젓가락 교정 시스템

OpenPose를 이용한 젓가락질 교정 시스템입니다.

이미지를 입력받고, 입력받은 이미지에서 OpenPose를 이용하여 손가락의 관절을 검출합니다.

검출된 손가락과, 올바른 젓가락질을 비교하여 얼마나 잘못되었는지를 예측합니다.

예측한 결과를 OpenCV를 이용하여 이미지에서 잘못된 부분을 어떻게 옮겨서 올바른 젓가락질을 하는지에 대해서 알 수 있도록 시각화해서 나타냅니다.

손의 관절만이 아니라 젓가락의 위치에서도 중요한 점을 파악하여 젓가락을 어떻게 잡고 있는지 등을 추측하여 잘못된 부분을 파악해서 사용자에게 알려줍니다.

---------------------------------------------------------------------------------------------------------------------------------------------------------------

졸업작품