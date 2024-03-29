#!/usr/bin/env python

# run.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0

import os
import sys
import cv2
import click
import traceback
from app import create_silhouette, SUBTRACTION_METHODS

# show version
def print_version(ctx, param, value): # pylint: disable=W0613
  if not value or ctx.resilient_parsing:
    return

  click.echo('Version 1.0.0')
  ctx.exit()

@click.command()
@click.argument('video', type=click.Path(exists=True))
@click.option('--frame-difference', default=15,
              help='Positive/Negative integer value to specify the difference of frames between moving objects.')
@click.option('--method', type=click.Choice(SUBTRACTION_METHODS), default="mog",
              help='The method that will be used in removing background.')
@click.option('--multithreaded', default=False, is_flag=True,
              help="""
              Enables multithreading to improve processing and rendering performance. This is dependent on how much logical CPUs you have on your computer.
              """)
@click.option('--show-video/--no-show-video', default=True,
              help='Shows video in a window.')
@click.option('--save-to-file', type=click.Path(writable=True), help='Path where the output file should be saved.')
@click.option('--disable-silhouette', default=False, is_flag=True, help="Disable silhouette generation.")
@click.option('--bg-learning-rate', default=None, type=click.FLOAT,
              help="Background Subtraction Algorithm learning rate.")
@click.option('--bg-history', default=None, type=click.INT, help="Background Subtraction Algorithm length of history.")
@click.option('--version', is_flag=True, callback=print_version, expose_value=False, is_eager=True,
              help='Show the version of the program.')
def main(video, frame_difference, method, multithreaded, show_video, save_to_file,
         disable_silhouette, bg_learning_rate, bg_history): # pylint: disable=R0915
  try:
    # read the video twice
    normal_video = cv2.VideoCapture(video)
    adjusted_video = None

    # video again if the silhouette generation is enabled
    if disable_silhouette is False:
      adjusted_video = cv2.VideoCapture(video)

    video_writer = None
    if save_to_file is not None and isinstance(save_to_file, str):
      splitted = os.path.splitext(save_to_file)
      formats_available = ['.mp4', '.avi']

      # check if the format is supported
      if splitted[1] not in formats_available:
        raise ValueError(f"Unsupported format. Supported formats are{', '.join(formats_available)}.")
      else:
        # get the width and height of the video
        width = normal_video.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = normal_video.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # set default fourcc for mp4
        fourcc = 0x21
        if splitted[1] == '.avi':
          fourcc = cv2.VideoWriter_fourcc(*'XVID')

        # create the video writer
        video_writer = cv2.VideoWriter(save_to_file, fourcc, 20.0, (int(width), int(height)), False)

    # disable multithreading if the number of logical CPUs is less than 2
    num_threads = cv2.getNumberOfCPUs()
    if num_threads < 2 and multithreaded is True:
      multithreaded = False

      thread_message = "Cannot run in multithreaded mode. Reverting to single-threaded mode since the "
      thread_message += "number of logical CPUs is less than 2."

      click.echo(thread_message)

    # show notice about performance when running single-threaded mode on methods other than absdiff
    if multithreaded is False and method != 'absdiff' and num_threads >= 2:
      thread_message = f"Running on single-threaded mode using \"{method}\". If you are experiencing some jank/lag, "
      thread_message += "re-run the program with --multithreaded flag present."

      click.echo(thread_message)

    intro_message = "To quit: press 'ctrl + c' when focused on the command line."
    intro_message += " When focused on the video window press, 'q'."

    # show message on how to quit
    click.echo(intro_message)

    # create silhouette of the video
    result = create_silhouette(normal_video, adjusted_video,
                               frame_difference=frame_difference,
                               method=method,
                               multithreaded=multithreaded,
                               debug=show_video,
                               video_writer=video_writer,
                               no_silhouette=disable_silhouette,
                               bg_segm_history=bg_history,
                               bg_segm_lr=bg_learning_rate)

    # show message to the user
    if video_writer is not None and result is True:
      click.echo(f"File has been saved to: {save_to_file}")

    # delete the output file
    if video_writer is not None and result is False:
      os.remove(save_to_file)
  except KeyboardInterrupt:
    click.echo("Shutdown requested. Cleaning up resources.")

    # delete the output file
    if video_writer is not None:
      os.remove(save_to_file)
  except (Exception, ValueError): # pylint: disable=W0703
    traceback.print_exc(file=sys.stdout)

    # delete the output file
    if video_writer is not None:
      os.remove(save_to_file)

  # release the resources used
  normal_video.release()
  if adjusted_video is not None:
    adjusted_video.release()

  # release the output file writer
  if video_writer is not None:
    video_writer.release()

  # remove existing windows not yet closed
  cv2.destroyAllWindows()

# execute the main function
if __name__ == "__main__":
  main() # pylint: disable=E1120


