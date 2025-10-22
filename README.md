# Detection
Using built-in front face detection model from dlib library with pre-trained model 68 face landmarks detector and dynamic threshold to predicted the event of blink and drowsy eyes.
# Installation
Just copy or download my code
# Warning
The accuracy of the model is still not calculated yet, however, the practical test on record video had show somewhat decent result.
# Eye Aspect Ratio (EAR)
![Image](https://github.com/user-attachments/assets/ae5b5bb8-ede7-427e-b07d-6439b5730994)
<img width="783" height="171" alt="Image" src="https://github.com/user-attachments/assets/25c0dd26-f40a-4cd7-8802-40f534a3dc58" />
Calculate the distance between two point on plane using Euclidean method, then use EAR equation to get Eye Aspect Ratio value. The reason EAR is better than directly measure the distance between upper eyelid and lower eyelid is because of how image or video pixels map to camera. The resolution vary depend on the distance between the face and camera, but by using Eye Ratio, the ratio remain the same even the resoltuion are change.
Code implementation :
```python
def blink_detection(points, face_landmarks):
    A = math.dist([face_landmarks.part(points[1]).x , face_landmarks.part(points[1]).y],[face_landmarks.part(points[5]).x , face_landmarks.part(points[5]).y])
    B = math.dist([face_landmarks.part(points[2]).x , face_landmarks.part(points[2]).y],[face_landmarks.part(points[4]).x , face_landmarks.part(points[4]).y])
    C = math.dist([face_landmarks.part(points[0]).x , face_landmarks.part(points[0]).y],[face_landmarks.part(points[3]).x , face_landmarks.part(points[3]).y])
    EAR = (A+B) / (C*2)
    return EAR
```
