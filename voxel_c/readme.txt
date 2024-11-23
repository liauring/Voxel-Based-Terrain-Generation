Compile:
g++ terrain.cpp -o terrain -I/usr/local/Cellar/opencv/4.10.0_12/include/opencv4 -L/usr/local/Cellar/opencv/4.10.0_12/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -arch x86_64 -std=c++17 -lm

Run：
./terrain