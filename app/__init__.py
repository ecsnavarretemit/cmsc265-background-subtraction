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
  adjusted_frame_start_point = frame_difference - 1
  adjusted_video = cv2.VideoCapture(video_file)
  adjusted_video.set(cv2.CAP_PROP_POS_FRAMES, adjusted_frame_start_point)

  # get the first frames of the normal_video
  _, normal_frame = normal_video.read()
  previous_normal_frame = normal_frame

  adjusted_frame = None
  previous_adjusted_frame = None
  has_initial_adjusted_frame = False

  # get the first frames of the adjusted_video but this time query the current frame of the normal
  # video plus the n - 1 frames to get the future silhouette when starting point is greater than 0
  if adjusted_frame_start_point >= 0:
    _, adjusted_frame = adjusted_video.read(normal_frame)
    previous_adjusted_frame = adjusted_frame

    has_initial_adjusted_frame = True

  # set default frame difference function
  normal_fn = frame_difference_absdiff
  adjusted_fn = frame_difference_absdiff

  # swap the frame differencing function for the normal and adjusted
  if method == 'mog':
    normal_fn = frame_difference_mog()
    adjusted_fn = frame_difference_mog()

  # initial value for countint frames
  frame_counter = 0
  if has_initial_adjusted_frame is True:
    frame_counter = 1

  # read the file
  while normal_video.isOpened():
    # break the loop when the normal_frame equates to None
    if normal_frame is None and frame_difference >= 0:
      break

    # break the loop when the provided frame difference is less than 0 and the adjusted frame contains `None` value
    if adjusted_frame is None and frame_difference < 0 and has_initial_adjusted_frame is True:
      break

    normal_fd = None

    # get the frame difference for normal and previous_normal frames
    if normal_frame is not None:
      normal_fd = normal_fn(normal_frame, previous_normal_frame)

    # set the value of the combined_fd to the normal_fd as a default value
    combined_fd = normal_fd

    # if advance_frame is not `None`, get the frame difference and combine it to the
    # frame difference of the normal frame using addWeighted function
    if adjusted_frame is not None:
      adjusted_fd = adjusted_fn(adjusted_frame, previous_adjusted_frame)

      # combine the normal_fd and adjusted_fd if normal_frame is not `None`
      # or else set the combined_fd to the value of the adjusted_fd
      if normal_frame is not None:
        combined_fd = cv2.addWeighted(normal_fd, 1, adjusted_fd, 1, 0)
      else:
        combined_fd = adjusted_fd

    # show the combined result
    if debug is True:
      cv2.imshow('combined', combined_fd)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # get the next frame and process it
    if normal_frame is not None:
      previous_normal_frame = normal_frame.copy()

    _, normal_frame = normal_video.read()

    # get the next frame of the adjusted video and store the previous
    if adjusted_frame is not None:
      previous_adjusted_frame = adjusted_frame.copy()

    if has_initial_adjusted_frame is True:
      _, adjusted_frame = adjusted_video.read()

    # begin reading the adjusted frame after normalizing the start point
    # for frame_difference that is set to negative value
    if (frame_counter + adjusted_frame_start_point) >= 0 and has_initial_adjusted_frame is False:
      _, adjusted_frame = adjusted_video.read()
      previous_adjusted_frame = adjusted_frame

      has_initial_adjusted_frame = True

    # increment the frame counter
    frame_counter += 1

def frame_difference_absdiff(current_frame, previous_frame):
  # convert the current and previous frames to grayscale
  current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
  previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

  # get the difference of the previous and the current frames to get the moving objects
  return cv2.absdiff(current_frame_gray, previous_frame_gray)

def frame_difference_mog():
  # create instance for background subtraction using MOG (Mixture of Gaussian)
  subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

  def apply_subtraction(current_frame, _):
    # apply the background subtractor to the frame and return it
    return subtractor.apply(current_frame)

  # return the inner function for usage
  return apply_subtraction


