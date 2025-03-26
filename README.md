<h1 align="center"><b>🧠 Drowsiness Detection (Image-based Classification)</b></h1>

<h2><b>📍 Objective</b></h2>
<p>
Train a Convolutional Neural Network (CNN) to classify drowsiness vs. alert states based on pre-collected eye state images.
</p>

<h2><b>🛠 Tech Stack</b></h2>
<ul>
  <li><b>Language:</b> Python</li>
  <li><b>Libraries:</b> TensorFlow, Keras, NumPy, Matplotlib, Scikit-learn</li>
  <li><b>Development Environment:</b> Jupyter Notebook</li>
</ul>

<h2><b>📂 Dataset</b></h2>
<ul>
  <li>Loaded from pre-saved <b>NumPy arrays (X.npy and y.npy)</b>.</li>
  <li>Images of eye states labeled as drowsy/alert.</li>
  <li>Dataset was split into training and validation sets.</li>
</ul>

<h2><b>📈 Model Training</b></h2>
<ul>
  <li><b>Architecture:</b> Convolutional Neural Network (CNN) with multiple Conv2D & MaxPooling layers.</li>
  <li><b>Loss Function:</b> Sparse Categorical Crossentropy</li>
  <li><b>Epochs:</b> 25 (as per notebook)</li>
  <li><b>Evaluation:</b> Accuracy and loss curves plotted.</li>
</ul>

<h2><b>📦 Dependencies</b></h2>
<pre>
tensorflow==2.15.0
keras==2.15.0
numpy==1.24.3
matplotlib==3.7.3
scikit-learn==1.3.0
</pre>


<h2><b>📊 Results</b></h2>
<ul>
  <li>Achieved good classification accuracy on validation data.</li>
  <li>Clear training vs. validation accuracy & loss plots included.</li>
  <li>Model saved in <code>.h5</code> format for future use.</li>
</ul>

<h2><b>🚀 Future Enhancements</b></h2>
<ul>
  <li>Add real-time webcam integration for practical application.</li>
  <li>Use larger datasets for more robustness.</li>
  <li>Deploy using Flask or Streamlit as a web app.</li>
</ul>


