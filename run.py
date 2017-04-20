#!/usr/bin/env python

# run.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import sys
import cv2
import traceback
from app import create_silhouette

def main():
  try:
    video_file = os.path.join(os.getcwd(), "assets/videos/small-video.mp4")

    # show error when the file does not exist
    if not os.path.exists(video_file):
      print("Path to video file does not exist: %s" % video_file)
      sys.exit(1)

    # read the video twice
    normal_video = cv2.VideoCapture(video_file)
    adjusted_video = cv2.VideoCapture(video_file)

    # show message on how to quit
    print("To quit: press 'ctrl + c' when focused on the command line. When focused on the video window press, 'q'.")

    # create silhouette of the video
    create_silhouette(normal_video, adjusted_video, frame_difference=15, debug=True)
  except KeyboardInterrupt:
    print("Shutdown requested. Cleaning up resources.")
  except Exception:
    traceback.print_exc(file=sys.stdout)

  # release the resources used
  normal_video.release()
  adjusted_video.release()

  # remove existing windows not yet closed
  cv2.destroyAllWindows()

# execute the main function
if __name__ == "__main__":
  main()


