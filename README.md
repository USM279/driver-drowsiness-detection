Sure! Here's your complete professional `README.md` content in **one clean English text block**, ready to copy and paste into your file:

---

```markdown
# Driver Drowsiness Detection 🚗💤

A real-time driver drowsiness detection system using a combination of Eye Aspect Ratio (EAR) via MediaPipe and a Convolutional Neural Network (CNN) model based on VGG16. The system triggers an audible alert if the driver shows signs of drowsiness.

---

## Features

- 🧠 CNN-based classification (`alert` vs `drowsy`)
- 👁️ Real-time eye closure detection using EAR
- 🔊 Plays a warning sound (`alert.wav`) when drowsiness is detected
- 🎥 Webcam integration for live monitoring
- 📊 Evaluation tools: confusion matrix, accuracy/loss plots, and sample image testing

---

## Project Structure

```

├── data/                   # Raw dataset folders
├── data\_split/            # Organized train/test folders
├── charts/                # Training accuracy/loss graphs
├── external/              # Test images, sample outputs
├── drowsiness\_detection\_combined.py   # Main real-time detection script
├── train\_vgg16\_model.py   # Model training script
├── evaluate\_model.py      # Accuracy report and confusion matrix
├── alert.wav              # Alarm sound (must be in project root)
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

````

---

## Requirements

Install required packages:

```bash
pip install -r requirements.txt
````

Make sure you place `alert.wav` in the root directory.

---

## How to Run

Run the detection system:

```bash
python drowsiness_detection_combined.py
```

* Press `Q` to quit the webcam window.
* Alarm will trigger only if:

  * Eyes are closed for a few frames (via EAR), **and**
  * The CNN model predicts the frame as drowsy.

---

## Model Overview

* Dataset: 4 classes → `Open`, `Closed`, `Yawn`, `No Yawn`
* Image size: 150x150
* Architecture: VGG16 (transfer learning) + Dense layers
* Classification: Binary (`alert`, `drowsy`)
* Accuracy: **\~95.7% on test set**

---

## Evaluation (Sample)

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 95.7% |
| Precision | 96%   |
| Recall    | 96%   |
| F1-score  | 96%   |

Training performance charts and confusion matrix are available in the `/charts` folder.

---

## Author

**Fadi Obada**
Software Engineering Student
Sakarya University – 2025

---

## License

This project is intended for educational and academic purposes. For any commercial use, please contact the author.

---

## Future Enhancements

* Detect yawning more accurately
* Optimize for deployment on mobile (TensorFlow Lite)
* Add driver ID recognition and alert logging

```
```
