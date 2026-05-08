import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
import numpy as np

from pipeline import process_image  

class KnotPredictor:
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ['correct', 'wrong'] 
        
        self.out_dir = Path("prediction-images")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        print("Lade Modell-Architektur...")
        self.model = models.resnet18()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(self.class_names))

        print(f"Lade Gewichte von '{model_path}'...")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model = self.model.to(self.device)
        self.model.eval() 
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def save_debug_image(self, orig_path, cnn_img, prediction, confidence):
        orig_img = cv2.imread(str(orig_path))
        h, w = orig_img.shape[:2]
        scale = 224.0 / h
        orig_resized = cv2.resize(orig_img, (int(w * scale), 224))
        
        if len(cnn_img.shape) == 2:
            cnn_bgr = cv2.cvtColor(cnn_img, cv2.COLOR_GRAY2BGR)
        else:
            cnn_bgr = cnn_img.copy()

        combined = cv2.hconcat([orig_resized, cnn_bgr])
        
        canvas_h = combined.shape[0] + 80
        canvas_w = combined.shape[1]
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        canvas[80:, :] = combined
        
        color = (0, 255, 0) if prediction == 'correct' else (0, 0, 255) # Grün oder Rot
        text = f"Pred: {prediction.upper()} ({confidence:.1f}%)"
        
        cv2.putText(canvas, text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
        cv2.putText(canvas, "Original", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, "CNN Input", (orig_resized.shape[1] + 20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        save_path = self.out_dir / f"debug_{orig_path.name}"
        cv2.imwrite(str(save_path), canvas)
        print(f"  -> Debug-Bild gespeichert: {save_path}")

    def predict(self, img_path):
        img_path = Path(img_path)
        if not img_path.exists():
            return "Bild nicht gefunden", 0.0
            
        print(f"\nAnalysiere: {img_path.name}")
        
        # OPENCV-PIPELINE
        cnn_ready_img = process_image(str(img_path))
        
        if cnn_ready_img is None:
            print("  -> Fehler: Knoten in der Pipeline nicht gefunden.")
            return "Fehler in der Vorverarbeitung", 0.0
            
        if len(cnn_ready_img.shape) == 2:
            cnn_ready_rgb = cv2.cvtColor(cnn_ready_img, cv2.COLOR_GRAY2RGB)
        else:
            cnn_ready_rgb = cv2.cvtColor(cnn_ready_img, cv2.COLOR_BGR2RGB)
            
        pil_img = Image.fromarray(cnn_ready_rgb)

        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            max_prob, predicted_idx = torch.max(probabilities, 0)
            
            predicted_class = self.class_names[predicted_idx.item()]
            confidence = max_prob.item() * 100

        self.save_debug_image(img_path, cnn_ready_img, predicted_class, confidence)
            
        return predicted_class, confidence

if __name__ == '__main__':
    model_path = "knoten_resnet18_finetuned.pth"
    predictor = KnotPredictor(model_path)
    
    test_image_path = Path("./test_images/IMG_6836.jpg")
    
    prediction, confidence = predictor.predict(test_image_path)
    
    print("-" * 40)
    print(f"Ergebnis: Der Knoten ist '{prediction}'")
    print(f"Sicherheit: {confidence:.2f}%")
    print("-" * 40)