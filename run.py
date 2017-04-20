#!/usr/bin/env python

# run.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
from app import create_silhouette

video_file = os.path.join(os.getcwd(), "assets/videos/small-video.mp4")
create_silhouette(video_file, frame_difference=15, debug=True)


