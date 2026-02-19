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

# Декларувати виконувальний файл
add_executable(
${PROJECT_NAME}
  src/main.cpp
  src/nn.cpp
)

#   Додати директорію include
# для заголовкових файлів
include_directories(include)

# Підключення Eigen3 до проєкту
target_link_libraries(
${PROJECT_NAME}
PRIVATE
  Eigen3::Eigen
)


