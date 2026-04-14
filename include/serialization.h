#pragma once

#include <Eigen/Dense>
#include <cstddef>

#include <memory>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/make_shared.hpp>
#include <boost/serialization/nvp.hpp>

#include <boost/serialization/vector.hpp>
#include <vector>

namespace boost {
namespace serialization {

// Serialize Eigen::MatrixXd
template<class Archive>
void serialize(Archive & ar, Eigen::MatrixXd & matrix, const unsigned int version) {
    int rows = matrix.rows();
    int cols = matrix.cols();

    if (Archive::is_saving::value) {
        ar & rows;
        ar & cols;
        // Save data as a vector
        std::vector<double> data(matrix.data(), matrix.data() + rows * cols);
        ar & data;
    } else {
        ar & rows;
        ar & cols;
        std::vector<double> data;
        ar & data;
        matrix.resize(rows, cols);
        std::copy(data.begin(), data.end(), matrix.data());
    }
}

} // namespace serialization
} // namespace boost

