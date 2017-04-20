#!/usr/bin/env python

# __init__.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import sys
import cv2

def create_silhouette(video_file, **kwargs):
  debug = kwargs.get('debug', False)
  method = kwargs.get('method', 'absdiff')
  frame_difference = kwargs.get('frame_difference', 15)

  # show error when the file does not exist
  if not os.path.exists(video_file):
    print("Path to video file does not exist: %s" % video_file)
    sys.exit(1)

  # check if the method specified is available or not
  methods = ['absdiff', 'mog']
  if method not in methods:
    print("Method %s not available. Available methods: %s" % (method, ",".join(methods)))
    sys.exit(1)

  # read the video
  normal_video = cv2.VideoCapture(video_file)

  # read the video again but once every n-1 frames in advance since the CAP_PROP_POS_FRAMES uses 0-based index
  advanced_video = cv2.VideoCapture(video_file)
  advanced_video.set(cv2.CAP_PROP_POS_FRAMES, (frame_difference - 1))

  # get the first frames of the normal_video
  _, normal_frame = normal_video.read()
  previous_normal_frame = normal_frame

  # get the first frames of the advanced_video but this time query the current frame of the normal
  # video plus the n - 1 frames to get the future silhouette
  _, advanced_frame = normal_video.read(normal_frame)
  previous_advanced_frame = advanced_frame

  # set default frame difference function
  fn = frame_difference_absdiff

  if method == 'mog':
    print("Not yet implemented")
    sys.exit(1)

  # read the file
  while normal_video.isOpened():
    # break the loop when the normal_frame equates to None
    if normal_frame is None:
      break

    # get the frame difference for normal and previous_normal frames
    normal_fd = fn(normal_frame, previous_normal_frame)

    combined = normal_fd

    # if advance_frame is not `None`, get the frame difference and combine it to the
    # frame difference of the normal frame using addWeighted function
    if advanced_frame is not None:
      advanced_fd = fn(advanced_frame, previous_advanced_frame)
      combined = cv2.addWeighted(normal_fd, 1, advanced_fd, 1, 0)

    # show the combined result
    if debug is True:
      cv2.imshow('combined', combined)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # get the next frame and process it
    previous_normal_frame = normal_frame.copy()
    _, normal_frame = normal_video.read()

    # get the next frame of the advanced video and store the previous
    if advanced_frame is not None:
      previous_advanced_frame = advanced_frame.copy()

    _, advanced_frame = advanced_video.read()

def frame_difference_absdiff(current_frame, previous_frame):
  # convert the current and previous frames to grayscale
  current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
  previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

  # get the difference of the previous and the current frames to get the moving objects
  return cv2.absdiff(current_frame_gray, previous_frame_gray)

def create_silhouette_mog(video_file, **kwargs):
  debug = kwargs.get('debug', False)
  frame_difference = kwargs.get('frame_difference', 15)

  if not os.path.exists(video_file):
    print("Path to video file does not exist: %s" % video_file)
    sys.exit(1)

  # read the video
  normal_video = cv2.VideoCapture(video_file)

  # get the first frames of the normal_video
  _, normal_frame = normal_video.read()

  # read the video again but once every n-1 frames in advance since the CAP_PROP_POS_FRAMES uses 0-based index
  advanced_video = cv2.VideoCapture(video_file)
  advanced_video.set(cv2.CAP_PROP_POS_FRAMES, (frame_difference - 1))

  # get the first frames of the advanced_video but this time query the current frame of the normal
  # video plus the n - 1 frames to get the future silhouette
  _, advanced_frame = advanced_video.read(normal_frame)

  # create instance for background subtraction using MOG (Mixture of Gaussian)
  normal_bg_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
  advanced_bg_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

  while normal_video.isOpened():
    # apply the background subtractor to the normal and advanced frames
    normal_subtracted = normal_bg_subtractor.apply(normal_frame)
    advanced_subtracted = advanced_bg_subtractor.apply(advanced_frame)

    # set the value of the combine frames to the bg subtracted of the normal frame
    combined = normal_subtracted

    # combine the advanced and normal frames using addWeighted function
    if advanced_subtracted is not None:
      combined = cv2.addWeighted(normal_subtracted, 1, advanced_subtracted, 1, 0)

    # if there is no frames that can be rendered break the loop
    if combined is None:
      break

    # show the combined result
    if debug is True:
      cv2.imshow('combined', combined)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # retrieve the next frames to be processed
    _, normal_frame = normal_video.read()
    _, advanced_frame = advanced_video.read()

  # release all the resources used
  normal_video.release()
  advanced_video.release()
  cv2.destroyAllWindows()


