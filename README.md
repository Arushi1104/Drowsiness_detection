 <!DOCTYPE html>
<html>
<head>
   
</head>
<body>

<h1>DROWSINESS DETECTION MODEL</h1>
<p>This project is a deep learning-based model that detects whether eyes are open or closed from an image. 
It can be used for drowsiness detection systems, especially for drivers or real-time monitoring.</p>

<h2>Tech Stack Used</h2>
<ul>
    <li><b>Python</b></li>
    <li><b>Libraries:</b> TensorFlow, Keras, OpenCV, Matplotlib, NumPy</li>
</ul>

<h2>Dataset</h2>
<p>Images of eyes classified into two categories:</p>
<ul>
    <li>Closed</li>
    <li>Open</li>
</ul>
<p>Folder structure:</p>
<pre>
0 FaceImages
   ├── Active Subjects (Awake)
   └── Fatigue Subjects (Drowsy)
</pre>
<p>source: https://www.kaggle.com/datasets/rakibuleceruet/drowsiness-prediction-dataset</p>


<h2>Preprocessing</h2>
<ul>
    <li>Images are loaded and resized.</li>
    <li>Values are scaled between 0 and 1 (instead of 0 to 255) for faster and smoother training.</li>
    <li>Dataset is split into:
        <ul>
            <li>70% — Training</li>
            <li>20% — Validation</li>
            <li>10% — Testing</li>
        </ul>
    </li>
</ul>

<h2>Model Architecture</h2>
<p>Built using Keras Sequential API</p>
<p>Layers used:</p>
<ul>
    <li>Conv2D + ReLU</li>
    <li>MaxPooling</li>
    <li>Flatten</li>
    <li>Dense Layers with sigmoid activation for binary classification</li>
</ul>
<pre>
Conv2D(16) -> MaxPooling  
Conv2D(64) -> MaxPooling  
Conv2D(32) -> MaxPooling  
Flatten -> Dense(256) -> Dense(1) with sigmoid
</pre>

<h2>Training</h2>
<ul>
    <li>Optimizer: Adam</li>
    <li>Loss: BinaryCrossentropy</li>
    <li>Epochs: 20</li>
    <li>Callback: TensorBoard logging</li>
</ul>

<h2>Evaluation Metrics</h2>
<ul>
    <li><strong>Training Accuracy:</strong> Reached ~97%</li>
    <li><strong>Validation Accuracy:</strong> Stabilized around ~90%</li>
    <li><strong>Loss Curves:</strong> Smooth downward trend with minimal overfitting</li>
</ul>
<h2>Testing</h2>
<p>Tested with custom screenshots/images.</p>
<p>The model predicts:</p>
<ul>
    <li>If yhat > 0.5 → Awake</li>
    <li>Else → Drowsy</li>
</ul>

<h2>How to Run /*still working on the real-time implementaion*/</h2>
<!-- <ol>
    <li>Clone the repo</li>
    <li>Keep your dataset in <b>train/Closed</b> and <b>train/Open</b> folders.</li>
    <li>Run the Python file or Jupyter Notebook.</li>
    <li>Predict with:</li>
</ol> -->
<!-- <pre>
import cv2
import tensorflow as tf
import numpy as np
img = cv2.imread('your_image.png')
resize = tf.image.resize(img, (256,256))
yhat = model.predict(np.expand_dims(resize/255, 0))
if yhat > 0.5:
    print("Eyes are Open")
else:
    print("Eyes are Closed")
</pre> -->

<h2>Future Work</h2>
<ul>
    <li>Add real-time video stream detection.</li>
    <li>Deploy as a web app or mobile app. </li>
    <li>Integrate with alert systems for drowsiness detection.</li>
</ul>

<h2>Contributing</h2>
<p>Feel free to raise issues or contribute by improving the model or code.</p>

</body>
</html>



