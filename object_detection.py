import os
import cv2
from ultralytics import YOLO

def main():
  img_size = (1920, 1080)
  vid = cv2.VideoCapture(0)

  if type(img_size) is tuple:
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, img_size[0])
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, img_size[1])

  model = YOLO("yolov8n.pt")

  while (True):
    ret, frame = vid.read()

    results = model.track(frame, conf=0.30, show=True, tracker="bytetrack.yaml")

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  vid.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()

