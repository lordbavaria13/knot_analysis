This application is a hybrid computer vision and deep learning tool for the automated detection and validation of tied figure-eight knots. The system combines a **geometric topology analysis** (skeletonization and intersection counting) with a **fine-tuned Convolutional Neural Network (ResNet18)** to ensure maximum accuracy.

---

## 🛠️ Prerequisites & Installation

### 1. Python Version
The program requires **Python 3.8** or newer (recommended: 3.10+).

### 2. Install Required Libraries
Open your terminal / command prompt and install the required packages using the following command:

```bash
pip install opencv-python opencv-contrib-python torch torchvision Pillow numpy
```

**IMPORTANT:** The `opencv-contrib-python` package is strictly required because it contains the `ximgproc` module (Zhang-Suen Thinning Algorithm) needed for the skeletonization of the knot!

### 3. Model File
Ensure that the trained model file **`knoten_resnet18_finetuned.pth`** is located in the same directory as the Python scripts.

---

## 📚 Libraries Used

*   **OpenCV (`cv2`) & `cv2.ximgproc`**: Image processing, filtering, Canny edge detection, GrabCut segmentation, morphology, and skeletonization.
*   **PyTorch (`torch`, `torch.nn`)**: Deep learning framework for executing the neural network.
*   **Torchvision (`torchvision.models`, `transforms`)**: Provides the ResNet18 architecture and image transformations (scaling, normalization).
*   **Tkinter (`tkinter`)**: Creation of the Graphical User Interface (GUI).
*   **NumPy (`numpy`)**: Efficient matrix and array computations (images are processed as NumPy arrays).
*   **Pillow (`PIL`)**: Format conversion between OpenCV matrices and PyTorch tensors.
*   **Threading (`threading`)**: Offloading the analysis to a background thread so the GUI remains responsive during computation.

---

## How to Use the Application

1.  Start the main program via the terminal:
    ```bash
    python main.py
    ```
2.  Click on **"Bild hochladen" (Upload Image)** in the left sidebar and select a photo of a knot.
3.  *(Optional)* Adjust the parameters in the **"Geometrie" (Geometry)** or **"CNN Pipeline"** tabs to match the current lighting conditions or rope type.
4.  Click on **"Analysieren" (Analyze)**.
5.  The result will be displayed in the bottom left. On the right side, you can switch between different views using the **tabs**:
    *   **Originalbild (Original Image)**: The uploaded image.
    *   **Mask Clean**: The cleaned black-and-white mask from the geometric analysis.
    *   **Kreuzungen (Intersections)**: The original image overlaid with the computed skeleton (red) and the detected intersection points (yellow circles).
    *   **CNN Beweisbild (CNN Proof Image)**: Shows the isolated knot exactly as the neural network sees it, including the percentage-based confidence.
6.  **Zoom Function:** In any tab, you can use the **mouse wheel** to zoom in and hold the **left mouse button** to pan the image.

## Parameter Explanation (GUI)

The program uses two separate pipelines, whose parameters can be adjusted live in the sidebar.

### Tab 1: Geometry (Skeleton)
These parameters control the creation of the solid rope mask and the detection of intersection points.

*   **Close / Open Kernel (Geo):** Size of the oval "brush" (in pixels) to close gaps in the mask (Close) or remove noise (Open).
*   **Close Iterations (Geo):** How often the close operation is repeated. Higher values make the rope more solid.
*   **Flatten Kernel & Brightness:** Parameters for shadow removal (CLAHE/Morphology). Evens out uneven background lighting.
*   **Border Margin:** Distance to the image edge in pixels. Intersections in this area are ignored (prevents errors from cut-off rope ends).
*   **Region Merge & Min Merge px:** Merges multiple closely clustered intersection points into a single logical center.
*   **Prune Length (px):** "Trims" the skeleton. Removes protruding, dead branches at the edge of the skeleton that could falsely be detected as intersections.

### Tab 2: CNN Pipeline
These parameters prepare the image specifically for the GrabCut algorithm and ResNet18.

*   **Blur Kernel:** Strength of the blur before edge detection. Very important for striped ropes to blur the texture.
*   **Canny Low / High:** Thresholds for edge detection. Determines the contrast difference required for a pixel to be considered an "edge".
*   **Close / Dilate Iterations:** How strongly the found edges are inflated and thickened before being flooded into a solid area (`fill_holes`).
*   **Border Thickness:** Draws an artificial black border around the image. Strictly necessary so the `fill_holes` algorithm doesn't flood the background if the rope touches the edge.
*   **Smooth Blur & Open Kernel:** Rounds off the hard, blocky edges after mask creation, as GrabCut works better with smooth masks.
*   **GrabCut Iterations:** How persistently the GrabCut algorithm attempts to fit the edges of the preliminary mask to the actual rope in the image.

---

## Decision Logic: Is it a Figure-Eight Knot?

The program does not make a blind decision, but combines the "understanding" of both systems (Topology + Pattern Recognition). A perfect figure-eight knot mathematically has exactly **4 to 6 intersection points**.

The final output is calculated according to the following weighting:

1.  🛑 **FALSE (CNN Veto):**
    If the CNN is **> 95% sure that the knot is wrong**, it overrides the geometry. *(Reason: The intersection count might be correct by chance, but the CNN detected a severe structural flaw).*
2.  ✅ **CORRECT (Double Validated):**
    If 4 to 6 intersection points were found **AND** the CNN is > 85% sure that it is a figure-eight knot.
3.  ⚠️ **CORRECT (Topology):**
    If 4 to 6 intersection points were found, but the CNN is unsure (or leans slightly towards 'wrong'). *(Reason: The physical structure of the knot is correct; the CNN is likely just confused by strange lighting or an unusual angle).*
4.  ❌ **FALSE (X Intersections):**
    In all other cases (e.g., 2 intersections found and no strong CNN veto). The physical topology of the knot is incorrect.