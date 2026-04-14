
add_executable(example example.cpp)
# add_executable(example2 example2.cpp)

target_link_libraries(
example
PRIVATE
  myml
  # Eigen3::Eigen
)

# target_link_libraries(
# example2
# PRIVATE
#   myml
#   # Eigen3::Eigen
# )

