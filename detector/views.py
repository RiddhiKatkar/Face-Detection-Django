from django.shortcuts import render, redirect
from .forms import UploadImageForm
from .models import FaceImage
import cv2
import numpy as np
from django.core.files.base import ContentFile
import os
import math

def _nms(boxes, iou=0.3):
    if not boxes:
        return []
    b = [(x, y, x + w, y + h, w * h) for (x, y, w, h) in boxes]
    b.sort(key=lambda t: t[4], reverse=True)
    keep = []
    while b:
        x1, y1, x2, y2, a = b.pop(0)
        keep.append((x1, y1, x2 - x1, y2 - y1))
        nb = []
        for (xx1, yy1, xx2, yy2, aa) in b:
            ix1 = max(x1, xx1)
            iy1 = max(y1, yy1)
            ix2 = min(x2, xx2)
            iy2 = min(y2, yy2)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            u = a + aa - inter
            r = inter / u if u > 0 else 0
            if r <= iou:
                nb.append((xx1, yy1, xx2, yy2, aa))
        b = nb
    return keep
def detect_faces(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            instance = form.save()
            
            # Read the image
            image_path = instance.image.path
            img = cv2.imread(image_path)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            
            faces_frontal = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7, minSize=(80, 80))
            faces_profile_left = profile_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7, minSize=(80, 80))
            gray_flip = cv2.flip(gray, 1)
            faces_profile_right_flip = profile_cascade.detectMultiScale(gray_flip, scaleFactor=1.05, minNeighbors=7, minSize=(80, 80))
            width_img = img.shape[1]
            faces_profile_right = [(width_img - x - w, y, w, h) for (x, y, w, h) in faces_profile_right_flip]
            faces = list(faces_frontal) + list(faces_profile_left) + list(faces_profile_right)
            filtered_faces = []
            for (x, y, w, h) in faces:
                aspect = w / float(h)
                if 0.6 <= aspect <= 1.5 and (w * h) >= 4000:
                    filtered_faces.append((x, y, w, h))
            filtered_faces = _nms(filtered_faces, iou=0.35)
            faces_count = len(faces)
            
            # Draw rectangles
            for (x, y, w, h) in filtered_faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            ret, buf = cv2.imencode('.jpg', img)
            content = ContentFile(buf.tobytes())
            filename = os.path.basename(instance.image.name)
            instance.processed_image.save(f'processed_{filename}', content)
            
            faces_count = len(filtered_faces)
            return render(request, 'detector/index.html', {'form': UploadImageForm(), 'instance': instance, 'faces_count': faces_count})
    else:
        form = UploadImageForm()
    
    return render(request, 'detector/index.html', {'form': form})
