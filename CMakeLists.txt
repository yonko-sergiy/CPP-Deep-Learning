# Встановлення мінімальної версії cmake
cmake_minimum_required(VERSION 4.2)

# Назва проєкту
project(CPP_Deep_Learning)

# Встановлення C++ стандарту
set(CMAKE_CXX_STANDARD 23)

#   Автоматична регенерація проєкту
# при модифікації CMakeLists.txt
set(CMAKE_SUPPRESS_REGENERATION true)

# Знайти бібліотеку Eigen3 у системі
find_package (Eigen3 REQUIRED NO_MODULE)
find_package(Boost REQUIRED COMPONENTS serialization)

include(CTest)

add_subdirectory(apps)
add_subdirectory(src)

option(BUILD_TESTS "Build the tests" ON)

if(BUILD_TESTS)
  # Add test-related targets here
  enable_testing()
  add_subdirectory(tests)
endif()

# Декларувати виконувальний файл
# add_executable(
# ${PROJECT_NAME}
#   src/main.cpp
#   src/nn.cpp
#   src/activation.cpp
#   src/layer.cpp
# )



#   Додати директорію include
# для заголовкових файлів
# include_directories(include)

# Підключення Eigen3 до проєкту
# target_link_libraries(
# ${PROJECT_NAME}
# PRIVATE
#   Eigen3::Eigen
# )


# enable_testing()
#
# add_subdirectory(tests)
