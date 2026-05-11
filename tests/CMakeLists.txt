


add_executable(
tests
  test_main.cpp
)

target_link_libraries(
tests
PRIVATE
  NN
  gtest
  Eigen3::Eigen
)

target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR})
add_test(NAME Tests COMMAND tests)

add_subdirectory(nn)

