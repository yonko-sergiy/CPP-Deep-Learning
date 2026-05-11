
add_executable(simple_example simple_example.cpp)
add_executable(titanic titanic.cpp)

target_link_libraries(
simple_example
PRIVATE
  NN
  Eigen3::Eigen
)

target_link_libraries(
titanic
PRIVATE
  NN
  ${OpenCV_LIBS}
  Eigen3::Eigen
)

