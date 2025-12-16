# CSCI_576_Proj_1

OpenCV must be installed prior and configured on the CMake before running

To run the code go into the working folder and configure the build

cmake -B .\build\

Then build the program

cmake --build .build\

Then run and execute

build\Debug\MyImageApplication.exe

#note Edward. These are the commands I had to run to get this working
cmake -S . -B build `-G "Visual Studio 17 2022" -A x64 `
cmake --build build --config Release
then add openCV dll files into folder where MyImageApplication.exe exists
./build/Release/MyImageApplication.exe
