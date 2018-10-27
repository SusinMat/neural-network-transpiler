#!/usr/bin/env bash

set -e
set -x

./nnt -w -m mobilenet_v1_1.0_224.tflite > mobilenet_v1.txt
./nnt -w -m mobilenet_quant_v1_224.tflite > mobilenet_v1_quant.txt
./nnt -w -m inception_v3_quant.tflite > inception_v3_quant.txt
./nnt -w -m inception_v4.tflite > inception_v4.txt
./nnt -w -m squeezenet.tflite > squeezenet.txt 
./nnt -w -m mobilenet_v2_1.0_224.tflite > mobilenet_v2.txt
