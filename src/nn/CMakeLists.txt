

add_library(
NN
  activation.cpp
  layer.cpp
  loss.cpp
)

target_include_directories(
NN
PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(
NN
PUBLIC
  Eigen3::Eigen
  Boost::serialization
)
