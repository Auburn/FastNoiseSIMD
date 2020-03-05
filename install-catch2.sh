#!/bin/sh
git clone https://github.com/catchorg/Catch2.git
mkdir Catch2/build
cd Catch2/build
cmake -DBUILD_TESTING=OFF ..
make && sudo make install
