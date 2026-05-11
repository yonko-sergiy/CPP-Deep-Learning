# Встановлення мінімальної версії cmake
cmake_minimum_required(VERSION 4.2)

# Назва проєкту
project(CPP_Deep_Learning)

# Встановлення C++ стандарту
set(CMAKE_CXX_STANDARD 23)

#   Автоматична регенерація проєкту
# при модифікації CMakeLists.txt
set(CMAKE_SUPPRESS_REGENERATION true)

# Знайти бібліотекі у системі
find_package (Eigen3 REQUIRED NO_MODULE)
find_package(Boost REQUIRED COMPONENTS serialization)
find_package(GTest)

# включити CTest
include(CTest)

add_subdirectory(apps)
add_subdirectory(src)

# Тести
option(BUILD_TESTS "Build the tests" ON)
if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()


