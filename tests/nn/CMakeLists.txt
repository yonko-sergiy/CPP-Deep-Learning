
add_executable(
test_activation
  test_activation.cpp
)

add_executable(
test_sigmoid_model
  test_sigmoid_model.cpp
)

target_link_libraries(
test_activation
PRIVATE
  NN
  gtest
)

target_link_libraries(
test_sigmoid_model
PRIVATE
  NN
  gtest
)

target_include_directories(test_activation PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(test_sigmoid_model PRIVATE ${CMAKE_SOURCE_DIR})
add_test(NAME test_activation COMMAND test_activation)
add_test(NAME test_sigmoid_model COMMAND test_sigmoid_model)

