# Facial animation based on OpenFace(an open source facial behavior analysis toolkit)

OpenFace is get from https://github.com/TadasBaltrusaitis/OpenFace, a state-of-the art open source tool intended for facial landmark detection, head pose estimation, facial action unit recognition, and eye-gaze estimation. 
And I create a facial animation project on Windows, it can work in both real-time videos and off-line videos. Using the facial landmarks and head pose obtained from the openface tool. I created a mapping relation between 
the model and the facial landmarks so that the vertices of the model will deformed by the variation of landmarks. 


## OpenFace

The system is capable of performing a number of facial analysis tasks:

- Facial Landmark Detection

![Sample facial landmark detection image](https://github.com/TadasBaltrusaitis/OpenFace/blob/master/imgs/multi_face_img.png)

- Facial Landmark and head pose tracking (links to YouTube videos)

<a href="https://www.youtube.com/watch?v=V7rV0uy7heQ" target="_blank"><img src="http://img.youtube.com/vi/V7rV0uy7heQ/0.jpg" alt="Multiple Face Tracking" width="240" height="180" border="10" /></a>
<a href="https://www.youtube.com/watch?v=vYOa8Pif5lY" target="_blank"><img src="http://img.youtube.com/vi/vYOa8Pif5lY/0.jpg" alt="Multiple Face Tracking" width="240" height="180" border="10" /></a>

- Gaze tracking (image of it in action)

<img src="https://github.com/TadasBaltrusaitis/OpenFace/blob/master/imgs/gaze_ex.png" height="378" width="567" >

- Facial Feature Extraction (aligned faces and HOG features)

![Sample aligned face and HOG image](https://github.com/TadasBaltrusaitis/OpenFace/blob/master/imgs/appearance.png)

## Facial Animation

![Facial animation](https://github.com/HanaRen/FacialAnimation_OpenFace/blob/master/videos/fa.jpg)
![Facial animation1](https://github.com/HanaRen/FacialAnimation_OpenFace/blob/master/videos/fa1.jpg)
![Facial animation2](https://github.com/HanaRen/FacialAnimation_OpenFace/blob/master/videos/fa2.jpg)



