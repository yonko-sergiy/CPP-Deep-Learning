
add_executable(
test_activation
  test_activation.cpp
)

target_link_libraries(
test_activation
PRIVATE
  myml
  gtest
)

target_include_directories(test_activation PRIVATE ${CMAKE_SOURCE_DIR})
add_test(NAME test_activation COMMAND test_activation)

