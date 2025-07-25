import cv2
import time

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Failed to open camera.")
    exit()

print("✅ Camera opened. Capturing 5 frames...")

for i in range(5):
    ret, frame = cap.read()
    if not ret:
        print(f"❌ Failed to capture frame {i}")
        continue
    filename = f"frame_{i+1}.jpg"
    cv2.imwrite(filename, frame)
    print(f"✅ Saved {filename}")
    time.sleep(1)

cap.release()
print("📸 Done. Check the image files in this folder.")
