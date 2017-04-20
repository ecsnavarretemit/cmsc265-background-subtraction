#!/usr/bin/env python

# run.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import cv2

video_file = os.path.join(os.getcwd(), "assets/videos/small-video.mp4")
D_FRAME = 15

# read the video
normal_video = cv2.VideoCapture(video_file)

_, current_frame1 = normal_video.read()
previous_frame1 = current_frame1

# read the video again but once every n-1 frames in advance since the CAP_PROP_POS_FRAMES uses 0-based index
advanced_video = cv2.VideoCapture(video_file)
advanced_video.set(cv2.CAP_PROP_POS_FRAMES, (D_FRAME - 1))

_, current_frame2 = normal_video.read(current_frame1)
previous_frame2 = current_frame2

# read the file
while normal_video.isOpened():
  # break the loop when the frame of the present equates to None
  if current_frame1 is None:
    break

  # convert the current and previous frames to grayscale
  current_frame1_gray = cv2.cvtColor(current_frame1, cv2.COLOR_BGR2GRAY)
  previous_frame1_gray = cv2.cvtColor(previous_frame1, cv2.COLOR_BGR2GRAY)

  # get the difference of the previous and the current frames to get the moving objects
  frame_diff1 = cv2.absdiff(current_frame1_gray, previous_frame1_gray)

  # set the value of combined to true as the default value
  combined = frame_diff1

  # perform the grayscale conversion and frame difference to the advanced video
  # the combine them using the addWeighted function to make them
  if current_frame2 is not None:
    current_frame2_gray = cv2.cvtColor(current_frame2, cv2.COLOR_BGR2GRAY)
    previous_frame2_gray = cv2.cvtColor(previous_frame2, cv2.COLOR_BGR2GRAY)

    frame_diff2 = cv2.absdiff(current_frame2_gray, previous_frame2_gray)

    combined = cv2.addWeighted(frame_diff1, 1, frame_diff2, 1, 0)

  # show the combined result
  cv2.imshow('combined', combined)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

  # get the next frame and process i
  previous_frame1 = current_frame1.copy()
  _, current_frame1 = normal_video.read()

  # get the next frame of the advanced video
  if current_frame2 is not None:
    previous_frame2 = current_frame2.copy()
    _, current_frame2 = advanced_video.read()

# release all the resources used
normal_video.release()
advanced_video.release()
cv2.destroyAllWindows()


