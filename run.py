#!/usr/bin/env python

# run.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import sys
import cv2
import click
import traceback
from app import create_silhouette, SUBTRACTION_METHODS

@click.command()
@click.argument('video', type=click.Path(exists=True))
@click.option('--frame-difference', default=15,
              help='Positive/Negative integer value to specify the difference of frames between moving objects.')
@click.option('--method', type=click.Choice(SUBTRACTION_METHODS), default="mog",
              help='The method that will be used in removing background.')
@click.option('--multithreaded', default=False, is_flag=True,
              help='Enables multithreading to improve processing and rendering performance.')
@click.option('--show-video', default=True,
              help='Shows video in a window.')
def main(video, frame_difference, method, multithreaded, show_video):
  try:
    # read the video twice
    normal_video = cv2.VideoCapture(video)
    adjusted_video = cv2.VideoCapture(video)

    # show notice about performance when running single-threaded mode on methods other than absdiff
    if multithreaded is False and method != 'absdiff':
      thread_message = f"Running on single-threaded mode using \"{method}\". If you are experiencing some jank/lag, "
      thread_message += "re-run the program with --multithreaded flag present."

      click.echo(thread_message)

    intro_message = "To quit: press 'ctrl + c' when focused on the command line."
    intro_message += "When focused on the video window press, 'q'."

    # show message on how to quit
    click.echo(intro_message)

    # create silhouette of the video
    create_silhouette(normal_video, adjusted_video,
                      frame_difference=frame_difference,
                      method=method,
                      multithreaded=multithreaded,
                      debug=show_video)
  except KeyboardInterrupt:
    click.echo("Shutdown requested. Cleaning up resources.")
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


