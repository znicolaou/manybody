#!/bin/bash
if [ $# -ne 3 ]; then
  echo usage: encode.sh infolder outfile rate
else
  ffmpeg -y -r $3 -i $1/%04d.png -c:v libx264 -preset fast -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p" $2.mp4
fi
