This application is a hybrid computer vision and deep learning tool for the automated detection 
and validation of tied figure-eight knots. The system combines a **geometric topology analysis** 
(skeletonization and intersection counting) with a **fine-tuned Convolutional Neural Network (ResNet18)** 
to ensure maximum accuracy. Image acquisition is supported both via local file upload and 
**live smartphone camera** using the DroidCam app.

---

## 🛠️ Prerequisites & Installation

### 1. Python Version
The program requires **Python 3.8** or newer (recommended: 3.10+).

### 2. Install Required Libraries
Open your terminal / command prompt and install all required packages:

```bash
pip install opencv-python opencv-contrib-python torch torchvision Pillow numpy
```

**IMPORTANT:** The `opencv-contrib-python` package is strictly required because it contains the 
`ximgproc` module (Zhang-Suen Thinning Algorithm) used for skeletonization of the knot!

### 3. Model File
Ensure that the trained model file **`knoten_resnet18_finetuned.pth`** is located in the same 
directory as the Python scripts. The model is a ResNet18 fine-tuned on 7,000 images of 
correct and incorrect figure-eight knots.

---

## 📂 Project File Structure

```
📁 project/
├── main.py                            ← Main GUI application (entry point)
├── image_processor.py                 ← CV pipeline: masking, skeletonization, CNN predictor
├── pipeline.py                        ← Standalone Canny pipeline for batch processing
├── live_capture.py                    ← DroidCam live capture window
├── zoomable_canvas.py                 ← Reusable zoomable image viewer widget
├── knoten_resnet18_finetuned.pth      ← Trained ResNet18 model weights
├── train_resnet18.py                  ← Training script (offline, not needed to run the app)
├── batch_process_knots.py             ← Dataset preprocessing script (offline)
└── split_dataset.py                   ← Train/Val split script (offline)
```

---

## 📚 Libraries Used

| Library | Purpose |
|---|---|
| `opencv-python` (`cv2`) | Core image processing: filtering, morphology, Canny, GrabCut, contours |
| `opencv-contrib-python` (`cv2.ximgproc`) | Zhang-Suen skeletonization for topology analysis |
| `torch`, `torch.nn` | Deep learning framework for running ResNet18 inference |
| `torchvision` | ResNet18 architecture and image normalization transforms |
| `Pillow` (`PIL`) | Converting OpenCV arrays to PIL images for PyTorch |
| `numpy` | Efficient array/matrix computations on image data |
| `tkinter` | Cross-platform GUI framework |
| `threading` | Background thread for analysis to keep GUI responsive |

---

## 🚀 How to Use the Application

Start the application by running:

```bash
python main.py
```

### Option A: Upload an Image
1. Click **Upload Image** in the left sidebar.
2. Select a `.jpg`, `.jpeg`, `.png`, or `.bmp` file of a knot.

### Option B: Live Capture via DroidCam 📷
1. Install the **DroidCam** app on your smartphone (available for Android & iOS).
2. Open the DroidCam app — it will display the device's **IP address** and **port**.
3. Click **Live Capture** in the sidebar.
4. Enter the IP address and port from the app into the connection dialog.
5. Click **"Connect"** — a live preview will appear immediately.
6. Click **capture image** to freeze the current frame.
7. Click **accept** to load the captured image into the main application,
   **retake** to retake, or **close** to close and disconnect.

### Running the Analysis
1. After loading an image, click **Analyze**.
2. The result appears in the bottom-left. Switch between image tabs on the right:

| Tab | Content |
|---|---|
| **Original** | The uploaded or captured raw image |
| **Mask** | The cleaned binary mask from the geometric pipeline |
| **Crossing** | Overlay with computed skeleton (red) and intersection points (yellow circles) |
| **CNN picture** | The isolated knot crop exactly as seen by the neural network, with confidence % |

> **Zoom:** Use the **mouse wheel** to zoom in any tab. **Left-click + drag** to pan.

---

## ⚙️ Parameter Explanation (GUI)

Two independent pipelines can be configured live in the left sidebar.

### Tab 1: Geometry (Skeleton) — Intersection Counting
Controls the creation of the rope mask and detection of topology crossings.

| Parameter | Description |
|---|---|
| **Close / Open Kernel (Geo)** | Size of the oval morphological brush to close gaps (Close) or remove noise (Open) |
| **Close Iterations (Geo)** | Repetitions of the close operation. Higher = more solid rope shape |
| **Flatten Kernel & Brightness** | Shadow removal via morphological background subtraction. Compensates uneven lighting |
| **Border Margin** | Pixel margin at image edge. Crossings inside this zone are ignored |
| **Region Merge & Min Merge px** | Merges nearby candidate crossings into a single logical center point |
| **Prune Length (px)** | Trims dead-end branches from the skeleton that would falsely count as crossings |

### Tab 2: CNN Pipeline — Neural Network Input
Controls how the rope image is prepared before being passed to ResNet18.

| Parameter | Description |
|---|---|
| **Blur Kernel** | Pre-Canny blur strength. Important for striped ropes to suppress texture |
| **Canny Low / High** | Thresholds for edge detection. Contrast below Low is ignored; above High is always an edge |
| **Close / Dilate Iterations** | Inflates detected edges to form a solid filled mask before GrabCut |
| **Border Thickness** | Black border around image. Prevents `fill_holes` from flooding the background |
| **Smooth Blur & Open Kernel** | Softens the binary mask for cleaner GrabCut segmentation |
| **GrabCut Iterations** | How hard GrabCut tries to fit mask boundaries to the actual rope |

---

## 🧠 Decision Logic: Is it a Figure-Eight Knot?

The program combines results from both systems. A mathematically correct figure-eight knot 
has exactly **4 to 6 intersection points** in its skeleton topology.

| Condition | Output |
|---|---|
| CNN confidence `wrong` **> 95%** | 🛑 **FALSE (CNN Veto)** — structural flaw detected, overrides geometry |
| 4–6 crossings **AND** CNN `correct` **> 85%** | ✅ **CORRECT (Double Validated)** — both methods agree |
| 4–6 crossings, CNN unsure or mildly negative | ⚠️ **CORRECT (Topology)** — structure is right, CNN may be confused by lighting |
| Any other case | ❌ **FALSE (X Crossings)** — topology is wrong |

---

## 🤖 Model Training (Background Information)

The ResNet18 model was trained using a custom dataset of **7,000 images** (3,500 correct, 
3,500 incorrect figure-eight knots). Images were preprocessed through the Canny pipeline 
(masking, GrabCut, CLAHE enhancement) before training. A two-phase transfer learning 
strategy was used:

- **Phase 1 (Warm-up, 5 epochs):** Only the final classification layer was trained. All 
  ResNet feature extraction layers were frozen.
- **Phase 2 (Fine-tuning, 10 epochs):** All layers were unfrozen and trained with a 
  low learning rate (`1e-4`) and a Cosine Annealing scheduler.

**Final validation accuracy: 97.75%**

---

## ⚠️ Important Notes on Code Originality

The following external resources were used in this project:

- **ResNet18 pre-trained weights** from `torchvision.models` (ImageNet, PyTorch Hub)
- **Zhang-Suen Thinning** via `cv2.ximgproc.thinning` (OpenCV Contrib)
- **GrabCut algorithm** via `cv2.grabCut` (OpenCV)

All other logic (pipeline design, topology analysis, decision fusion, GUI, training scripts, 
data collection) is original work developed for this project.