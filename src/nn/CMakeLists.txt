
# Додати бібліотеки
add_library(
NN
  activation.cpp
  layer.cpp
  loss.cpp
)

# Локалізувати шлях до заголовків
target_include_directories(
NN
PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Зв'язати покликання на бібліотеки
target_link_libraries(
NN
PUBLIC
  Eigen3::Eigen
  Boost::serialization
)
