

# CPP Deep Learning

Build without tests:
  cmake -S . -B build/ -DBUILD_TESTS=OFF
  cmake --build build/

Build with tests:
  cmake -S . -B build/ -DBUILD_TESTS=ON
  cmake --build build/
  ctest --test-dir tests/


OR:
  mkdir build
  cd build 

Build without tests:
  cmake .. -DBUILD_TESTS=OFF
  cmake --build .

Build with tests:
  cmake .. -DBUILD_TESTS=ON
  cmake --build .
  ctest
