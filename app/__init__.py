# __init__.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import cv2
from multiprocessing.pool import ThreadPool
from collections import deque

# list of available background subtraction methods
SUBTRACTION_METHODS = ['absdiff', 'mog', 'mog2', 'knn']

# task to emulate threads
class DummyTask:
  def __init__(self, data):
    self.data = data

  def ready(self):
    return True

  def get(self):
    return self.data

def create_silhouette(normal_video, adjusted_video, **kwargs): # pylint: disable=R0914, R0915
  debug = kwargs.get('debug', False)
  method = kwargs.get('method', 'absdiff')
  multithreaded = kwargs.get('multithreaded', False)
  frame_difference = kwargs.get('frame_difference', 15)
  video_writer = kwargs.get('video_writer', None)
  no_silhouette = kwargs.get('no_silhouette', False)

  # check if the method specified is available or not
  if method not in SUBTRACTION_METHODS:
    raise ValueError(f"Method \"{method}\" not available. Available methods: {', '.join(SUBTRACTION_METHODS)}")

  return_value = True

  # compute the adjusted_frame_start_point
  adjusted_frame_start_point = frame_difference - 1

  # get the first frames of the normal_video
  _, normal_frame = normal_video.read()
  previous_normal_frame = normal_frame

  adjusted_frame = None
  previous_adjusted_frame = None
  has_initial_adjusted_frame = False

  # get the first frames of the adjusted_video but this time query the current frame of the normal
  # video plus the n - 1 frames to get the future silhouette when starting point is greater than 0
  if adjusted_frame_start_point >= 0 and no_silhouette is False:
    adjusted_video.set(cv2.CAP_PROP_POS_FRAMES, adjusted_frame_start_point)

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
  elif method == 'mog2':
    normal_fn = frame_difference_mog2()
    adjusted_fn = frame_difference_mog2()
  elif method == 'knn':
    normal_fn = frame_difference_knn()
    adjusted_fn = frame_difference_knn()

  # initial value for countint frames
  frame_counter = 0
  if has_initial_adjusted_frame is True:
    frame_counter = 1

  # initialize thread pool
  num_threads = cv2.getNumberOfCPUs()
  pool = ThreadPool(processes=num_threads)
  pending = deque()

  # read the file
  while normal_video.isOpened():
    # break the loop when the normal_frame equates to None
    if normal_frame is None and frame_difference >= 0:
      break

    # break the loop when the provided frame difference is less than 0 and the adjusted frame contains `None` value
    if adjusted_frame is None and frame_difference < 0 and has_initial_adjusted_frame is True:
      break

    # process all the pending processed on the queue and show the result
    while len(pending) > 0 and pending[0].ready(): # pylint: disable=C1801
      combined = pending.popleft().get()

      # write to the file
      if video_writer is not None:
        video_writer.write(combined)

      # show the combined result
      if debug is True:
        cv2.imshow('combined', combined)

    # append tasks to the pending tasks pool, it can be performed in single-threaded or multithreaded mode
    if len(pending) < num_threads:
      if multithreaded is True:
        params = (normal_frame, previous_normal_frame, normal_fn, adjusted_frame, previous_adjusted_frame, adjusted_fn)
        task = pool.apply_async(process_frame, params)
      else:
        process = process_frame(normal_frame, previous_normal_frame, normal_fn, adjusted_frame,
                                previous_adjusted_frame, adjusted_fn)
        task = DummyTask(process)

      # append the derived task to the pending pool
      pending.append(task)

      # get the next frame and process it
      if normal_frame is not None:
        previous_normal_frame = normal_frame.copy()

      _, normal_frame = normal_video.read()

      if no_silhouette is False:
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

      # quit the process when q is pressed
      if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Shutdown requested. Cleaning up resources.")
        return_value = False
        break

  # remove existing windows not yet closed
  cv2.destroyAllWindows()

  return return_value

def process_frame(normal_frame, previous_normal_frame, normal_fn, adjusted_frame, previous_adjusted_frame, adjusted_fn):
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

  return combined_fd

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

def frame_difference_mog2():
  # create instance for background subtraction using MOG2 (Mixture of Gaussian)
  subtractor = cv2.createBackgroundSubtractorMOG2()

  def apply_subtraction(current_frame, _):
    # apply the background subtractor to the frame and return it
    return subtractor.apply(current_frame)

  # return the inner function for usage
  return apply_subtraction

def frame_difference_knn():
  # create instance for background subtraction using KNN (K-Nearest Neighbors)
  subtractor = cv2.createBackgroundSubtractorKNN()

  def apply_subtraction(current_frame, _):
    # apply the background subtractor to the frame and return it
    return subtractor.apply(current_frame)

  # return the inner function for usage
  return apply_subtraction


