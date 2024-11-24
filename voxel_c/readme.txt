Compile:
g++ terrain.cpp -o terrain -I/usr/local/Cellar/opencv/4.10.0_12/include/opencv4 -L/usr/local/Cellar/opencv/4.10.0_12/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -arch x86_64 -std=c++17 -lm


ARM compile:
g++ terrain.cpp -o terrain \
-I/opt/homebrew/Cellar/opencv/4.10.0_12/include/opencv4 \
-L/opt/homebrew/Cellar/opencv/4.10.0_12/lib \
-lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc \
-std=c++17


Run：
./terrain


g++ terrain.cpp -o terrain -I/usr/local/Cellar/opencv/4.10.0_12/include/opencv4 -L/usr/local/Cellar/opencv/4.10.0_12/lib -std=c++17 -lm