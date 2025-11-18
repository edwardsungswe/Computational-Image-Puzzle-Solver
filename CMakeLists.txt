cmake_minimum_required(VERSION 3.25)

project(ImageDisplay)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files
set(SOURCES
    src/Main.cpp
    src/FeatureExtractor.cpp
    src/Matcher.cpp
    src/Assembler.cpp
)

# OpenCV
set(OpenCV_DIR "C:/opencv")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#Collect all .cpp files automatically
file(GLOB_RECURSE SOURCES
    src/*.cpp
)

#Include all headers
include_directories(src)

#Build executable
add_executable(MyImageApplication ${SOURCES})

#Link OpenCV
target_link_libraries(MyImageApplication PRIVATE ${OpenCV_LIBS})
