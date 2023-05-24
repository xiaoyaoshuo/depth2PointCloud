cmake_minimum_required(VERSION 2.6)

project(depth2PointCloud)


set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)

find_package(PCL 1.2 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package( OpenCV 3.2 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(Boost COMPONENTS filesystem REQUIRED)
include_directories(include ${Boost_INCLUDE_DIRS})


add_executable(depth2PointCloud depth2PointCloud.cpp)

target_link_libraries(depth2PointCloud ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
install(TARGETS depth2PointCloud RUNTIME DESTINATION bin)

