

add_executable(
my_tests
  test_main.cpp
)

target_link_libraries(
my_tests
PRIVATE
  myml
  # Eigen3::Eigen
)

target_include_directories(my_tests PRIVATE ${CMAKE_SOURCE_DIR})

add_test(NAME myTests COMMAND my_tests)
