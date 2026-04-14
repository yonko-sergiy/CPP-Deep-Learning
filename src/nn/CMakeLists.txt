

add_library(
myml
  activation.cpp
  layer.cpp
  loss.cpp
  nn.cpp
)

target_include_directories(
myml
PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(
myml
PUBLIC
  Eigen3::Eigen
  Boost::serialization
)
