from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import cv2
import numpy as np
from pathlib import Path
import shutil
from pydantic import BaseModel
from typing import List
import base64
from ultralytics import YOLO
import uuid
import json
from datetime import datetime
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dossiers
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")
# dossier pour stocker les nouvelles annotations pour reentrainer yolo
YOLO_DIR = OUTPUT_DIR / "YOLO"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
YOLO_DIR.mkdir(exist_ok=True)


model = YOLO('modele_photo_legende_2.pt')

image_store = {}

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
    confidence: float
    class_name: str

class ExtractionRequest(BaseModel):
    image_id: str
    boxes: List[BoundingBox]

def segment_image_yolo(image_path: str) -> List[dict]:
    """Segmente l'image pour détecter les photos individuelles avec YOLO"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Impossible de charger l'image")
    
    # Inference avec YOLO
    results = model(img, conf=0.25, iou=0.45)
    
    boxes = []
    
    # Extraire les bounding boxes
    for result in results:
        for box in result.boxes:
            # Obtenir les coordonnées xyxy
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Convertir en format x, y, width, height
            x = int(x1)
            y = int(y1)
            width = int(x2 - x1)
            height = int(y2 - y1)
            
            # Filtrer les boîtes trop petites
            img_area = img.shape[0] * img.shape[1]
            box_area = width * height
            
            if box_area > img_area * 0.01 and box_area < img_area * 0.95:
                # Obtenir le nom de la classe depuis le modèle
                class_id = int(box.cls[0])
                class_name = model.names[class_id] if class_id < len(model.names) else f"class_{class_id}"
                
                boxes.append({
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "confidence": float(box.conf[0]),
                    "class": class_name
                })
    
    # Trier par position (haut vers bas, gauche vers droite)
    boxes.sort(key=lambda b: (b["y"], b["x"]))
    
    return boxes

@app.get("/")
async def root():
    return FileResponse("main.html")

@app.post("/api/detect")
async def detect_objects(file: UploadFile = File(...)):
    """Upload une image et retourne les bounding boxes détectées par YOLO"""
    try:
        # Générer un ID unique pour cette image
        image_id = str(uuid.uuid4())
        
        # Sauvegarder le fichier avec l'ID unique
        file_extension = Path(file.filename).suffix
        file_path = UPLOAD_DIR / f"{image_id}{file_extension}"
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Stocker les informations de l'image
        image_store[image_id] = {
            "path": str(file_path),
            "original_name": file.filename
        }
        
        # Segmenter l'image avec YOLO
        boxes = segment_image_yolo(str(file_path))
        
        return {
            "image_id": image_id,
            "boxes": boxes,
            "count": len(boxes)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/extract")
async def extract_images(request: ExtractionRequest):
    """Extrait les images selon les bounding boxes validées (Pillow + JPEG + fichier .txt YOLO)."""
    try:
        # Vérifier l’existence de l’image
        if request.image_id not in image_store:
            raise HTTPException(status_code=404, detail="Image non trouvée")
        
        image_info = image_store[request.image_id]
        input_path = Path(image_info["path"])

        if not input_path.exists():
            raise HTTPException(status_code=404, detail="Fichier non trouvé sur le disque")

        # Charger via Pillow
        try:
            img = Image.open(input_path).convert("RGB")
        except Exception:
            raise HTTPException(status_code=500, detail="Impossible de charger l'image")

        width, height = img.size
        base_name = Path(image_info["original_name"]).stem

        extracted_files = []
        mapping_entries = []

        # Fichier YOLO : nom identique à l’image source
        yolo_txt_path = YOLO_DIR / f"{base_name}.txt"
        yolo_lines = []

        for idx, box in enumerate(request.boxes, 1):

            x = max(0, int(box.x))
            y = max(0, int(box.y))
            w = min(int(box.width), width - x)
            h = min(int(box.height), height - y)

            if w <= 0 or h <= 0:
                continue

            crop_box = (x, y, x + w, y + h)
            cropped = img.crop(crop_box)

            class_suffix = f"_{box.class_name}" if getattr(box, "class_name", None) else ""
            output_filename = f"{base_name}_crop_{idx:02d}{class_suffix}.jpg"
            output_path = OUTPUT_DIR / output_filename

            # Sauvegarde JPEG
            cropped.save(output_path, format="JPEG", quality=95)

            extracted_files.append(output_filename)

            
            # générer les lignes de réentrainement yolo
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height

            w_norm = w / width
            h_norm = h / height

            # Classe 0 partout
            # TODO: changer si on veut du multiclasses
            yolo_line = f"0 {x_center} {y_center} {w_norm} {h_norm}"
            yolo_lines.append(yolo_line)


            mapping_entries.append({
                "crop_filename": output_filename,
                "source_image": image_info["original_name"],
                "source_image_id": request.image_id,
                "bbox": {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h
                },
                "class": getattr(box, "class_name", None),
                "confidence": getattr(box, "confidence", None),
                "timestamp": datetime.now().isoformat()
            })

        # Sauvegarde du fichier YOLO
        with open(yolo_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        # Gestion du fichier JSON global
        mapping_file = OUTPUT_DIR / "crops_mapping.json"
        if mapping_file.exists():
            with open(mapping_file, "r", encoding="utf-8") as f:
                all_mappings = json.load(f)
        else:
            all_mappings = []

        all_mappings.extend(mapping_entries)

        with open(mapping_file, "w", encoding="utf-8") as f:
            json.dump(all_mappings, f, indent=2, ensure_ascii=False)

        return {"success": True}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/images")
async def list_images():
    """Liste toutes les images uploadées"""
    return {
        "images": [
            {
                "id": img_id,
                "name": info["original_name"]
            }
            for img_id, info in image_store.items()
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)