import os
 
import cv2
 
PHOTOS_DIR = "reference-photos"
OUTPUT_DIR = "reference-faces"
 
haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
print("Cascade loaded OK:", not haar_cascade.empty())
 
for actor_name in sorted(os.listdir(PHOTOS_DIR)):
    actor_in = os.path.join(PHOTOS_DIR, actor_name)
    if not os.path.isdir(actor_in):
        continue
 
    actor_out = os.path.join(OUTPUT_DIR, actor_name)
    os.makedirs(actor_out, exist_ok=True)
 
    saved = 0
    for filename in sorted(os.listdir(actor_in)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
 
        img = cv2.imread(os.path.join(actor_in, filename), 0)
        if img is None:
            print(f"  {actor_name}/{filename}: could not read image, skipping")
            continue
 
        faces = haar_cascade.detectMultiScale(
            img, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100)
        )
 
        # reference photos should contain exactly one face
        if len(faces) != 1:
            print(f"  {actor_name}/{filename}: found {len(faces)} faces, skipping")
            continue
 
        x, y, w, h = faces[0]
        saved += 1
        cv2.imwrite(os.path.join(actor_out, f"{saved}.jpg"), img[y : y + h, x : x + w])
 
    print(f"{actor_name}: saved {saved} face crops")
 