# Neural Network Transpiler

Convert a model from tflite to a C++ source code using Android Neural Network API.

## Prerequisites

- Boost library
    - `sudo apt-get install libboost-all-dev #ubuntu`
    - `sudo pacman -S boost #arch-linux`
    - CMake ver. 3.8 or higher
    - compiler with C++17 support

## Build

In the project, directory create a build directory
```
$ mkdir build
$ cd build
```

In the build directory, call cmake to generate the makefile, and call the make program
```
$ cmake .. -DCMAKE_VERBOSE_MAKEFILE=true
$ make
```

Test the compiled program
```
$ ./nnt -h
```

## How to use

Assume that I have a `mobilenet_quant_v1_224.tflite` flatbuffer model file in build directory, the same directory from where I am executing the nnt executaeble.

### Model info
```
./nnt -m mobilenet_quant_v1_224.tflite -i
```

This outputs:

```
::Inputs::
 Placeholder<UINT8> [1, 224, 224, 3] (quantized)
   └─ Quant: {min:[0], max:[1], scale: [0.00392157], zero_point:[0]}

::Outputs::
 MobilenetV1/Predictions/Softmax<UINT8> [1, 1001] (quantized)
```

### Generating dot file
```
./nnt -m mobilenet_quant_v1_224.tflite -d mobnet.dot
```
The file mobnet.dot os generated in the same directory

### Generating NNAPI files to use on Android
```
./nnt -m mobilenet_quant_v1_224.tflite -j com.nnt.nnexample -p mobnet_path
```
This creates a directory named `mobnet_path` with files: `[jni.cc, nn.h, nn.cc, weights_biases.bin]`
where the Java package is com.nnt.nnexample
