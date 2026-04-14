
// #include "serialization.h"
// //
// namespace boost {
// namespace serialization {
// //
// // // Serialize Eigen::MatrixXd
// template<class Archive>
// void serialize(Archive & ar, Eigen::MatrixXd & matrix, const unsigned int version) {
//     int rows = matrix.rows();
//     int cols = matrix.cols();
//
//     if (Archive::is_saving::value) {
//         ar & rows;
//         ar & cols;
//         // Save data as a vector
//         std::vector<double> data(matrix.data(), matrix.data() + rows * cols);
//         ar & data;
//     } else {
//         ar & rows;
//         ar & cols;
//         std::vector<double> data;
//         ar & data;
//         matrix.resize(rows, cols);
//         std::copy(data.begin(), data.end(), matrix.data());
//     }
// }
//
// } // namespace serialization
// } // namespace boost



