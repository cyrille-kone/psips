cmake_minimum_required(VERSION 3.18)
project(code)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3 -fopenmp")
find_package(xtensor REQUIRED)
include_directories(${xtensor_INCLUDE_DIRS})
link_libraries(xtensor xtensor::optimize)
message("xtensor dir is " ${xtensor_INCLUDE_DIRS})
add_definitions(-DHAVE_CBLAS=1)
if (WIN32)
    find_package(OpenBLAS REQUIRED)
    set(BLAS_LIBRARIES ${CMAKE_INSTALL_PREFIX}${OpenBLAS_LIBRARIES})
else()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
endif()
#set(XTENSOR_USE_XSIMD 1)
#set(XTENSOR_USE_TBB 1)
#set(XTENSOR_USE_OPENMP 1)
include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)


message(STATUS "BLAS VENDOR:    " ${BLA_VENDOR})
message(STATUS "BLAS LIBRARIES: " ${BLAS_LIBRARIES})
link_libraries(nlohmann_json::nlohmann_json)
link_libraries(${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
add_executable(code main.cpp src/bandits.cxx
                    src/utils.cxx
                    src/policies.cxx
                    src/xp.hpp
                    src/learners.cxx
                    src/learners.hpp src/fcts.cxx src/fcts.hpp src/noc.hpp)