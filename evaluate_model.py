import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
model_path = 'vgg16_transfer_model.h5'
test_dir = 'data_split/test'
image_size = (150, 150)
batch_size = 32

# Load the model
model = load_model(model_path)

# Prepare test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False  # Important to match predictions with true labels
)

# Predict
predictions = model.predict(test_generator)
y_pred = (predictions > 0.5).astype(int).reshape(-1)
y_true = test_generator.classes

# Classification report
print("📊 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=['alert', 'drowsy']))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['alert', 'drowsy'], yticklabels=['alert', 'drowsy'], cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
