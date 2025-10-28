Tuberculosis (TB) Detection Model using Deep Learning
=====================================================

Overview
--------

This project uses deep learning techniques to detect **Tuberculosis (TB)** from **Chest X-ray (CXR)** images. The model is based on **ResNet50**, a pre-trained Convolutional Neural Network (CNN) architecture, and fine-tuned on the **TB vs. Normal** image dataset.

### Project Features:

* **Binary Classification**: Detects whether a given Chest X-ray image contains TB or is normal.

* **Transfer Learning**: Uses a pre-trained **ResNet50** model fine-tuned on the TB dataset.

* **Data Augmentation**: Enhances training with techniques like rotation, shifting, and zooming.

* **Model Evaluation**: Uses accuracy, loss, and confusion matrix for performance evaluation.

* **Model Saving**: The trained model can be saved and used for predictions on new data.

Project Structure
-----------------

```bash
tb-detection/
├── data/
│   ├── train/
│   │   ├── TB/
│   │   └── Normal/
│   ├── validation/
│   │   ├── TB/
│   │   └── Normal/
│   └── test/
│       ├── TB/
│       └── Normal/
├── tb_detection_model.h5         # Saved model after training
├── tb_detection_model.ipynb      # Jupyter notebook for training the model
├── README.md              # Project overview and instructions
└── requirements.txt        # List of required Python packages
```

Installation
------------

### Prerequisites:

To run this project, you'll need Python (>= 3.6) and several Python libraries, which are specified in the `requirements.txt` file.

1. **Clone the repository**:

```bash
git clone https://github.com/edmondweb/Tuberculosis-Detection.git
cd Tuberculosis-Detection
```

2. **Set up a virtual environment (optional but recommended):**

```bash
python -m venv venv
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate     # For Windows
```

3. **Install required dependencies**:

```bash
pip install -r requirements.txt
```

### Required Libraries:

* **TensorFlow** (for model creation and training)

* **Keras** (for deep learning operations)

* **NumPy** (for numerical computations)

* **Pillow** (for image processing)

* **scikit-learn** (for performance evaluation)

Create a `requirements.txt` file with the following content:

```context
tensorflow>=2.0.0
keras>=2.4.3
numpy>=1.19.0
pillow>=7.1.2
scikit-learn>=0.24.0
```

Dataset
-------

### Data Directory Structure:

The dataset is expected to be organized as follows:

```css
data/
├── train/
│   ├── TB/
│   └── Normal/
├── validation/
│   ├── TB/
│   └── Normal/
└── test/
    ├── TB/
    └── Normal/
```

* **TB**: Chest X-ray images containing tuberculosis.

* **Normal**: Chest X-ray images that are normal (no TB detected).

### Dataset Sources:

* **TB Data**: Includes 700 images from publicly accessible datasets and 2800 images from the NIAID TB portal dataset.

* **Normal Data**: Includes 3500 normal images collected from the NLM and RSNA datasets.

Please ensure the data is organized as shown above for compatibility with the script.

## Usage

### 1. **Train the Model**

To train the model, run the following command:

```bash
python tb_detection_model.ipynb
```

The script will:

* Load images from the `train`, `validation`, and `test` directories.

* Preprocess images (resize, normalize, augment).

* Fine-tune a **ResNet50** model using transfer learning.

* Train the model for 10 epochs (adjustable).

* Save the trained model as `tb_detection_model.h5`.

### 2. **Evaluate the Model**

Once the model is trained, it will be evaluated on the test set to provide performance metrics, including:

* **Accuracy**

* **Loss**

* **Precision, Recall, F1-Score**

* **Confusion Matrix**

### 3. **Make Predictions**

To make predictions on new Chest X-ray images, use the saved model (`tb_detection_model.h5`):

```python
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model

model = load_model('tb_detection_model.h5')

# Load a new image

img_path = 'path_to_image.png'
img = image.load_img(img_path, target_size=(512, 512))

# Preprocess the image

img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict whether the image is TB or Normal

prediction = model.predict(img_array)

if prediction > 0.5:
    print("The image is classified as TB.")
else:
    print("The image is classified as Normal.")

```

Model Performance
-----------------

After training, the model’s **accuracy** and **loss** will be displayed. We recommend monitoring these metrics and adjusting the training process (e.g., number of epochs, data augmentation) if necessary.

### Troubleshooting

1. **Out of Memory Errors**: If you experience memory issues, try reducing the batch size in the `ImageDataGenerator` or using a smaller model.

2. **Training Performance**: You may increase the number of epochs or adjust the learning rate to improve accuracy, especially for larger datasets.

Contributing
------------

Feel free to fork this repository, make changes, and submit pull requests. Contributions are welcome!

## License

This project is licensed under the MIT License.

## Acknowledgments

The dataset used in this project is based on the work of:

Rahman, T., Khandakar, A., Kadir, M. A., Islam, K. R., Islam, K. F., Mahbub, Z. B., Ayari, M. A., & Chowdhury, M. E. H. (2020). Reliable tuberculosis detection using chest X-ray with deep learning, segmentation, and visualization. *IEEE Access*, *8*, 191586–191601. https://doi.org/10.1109/ACCESS.2020.3031384
