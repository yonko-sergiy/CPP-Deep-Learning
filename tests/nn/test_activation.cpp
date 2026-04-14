#include <iostream>
#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "nn/activation.h"


using namespace NN;


TEST(SigmoidTests, zeroInput) {
  Eigen::MatrixXd X(1, 1);
  Eigen::MatrixXd T(1, 1);
  X << 0;
  T << 0.5;

  auto sigmoid = Activations::Sigmoid::create();
  EXPECT_EQ( sigmoid->function(X), T);
}

TEST(SigmoidTests, limits) {
  Eigen::MatrixXd X(1, 2);
  Eigen::MatrixXd T(1, 2);
  Eigen::MatrixXd Y(1, 2);
  X << -10, 10;
  T << 0,   1;

  auto sigmoid = Activations::Sigmoid::create();
  Y = sigmoid->function(X);
  EXPECT_NEAR( Y(0, 0), T(0, 0), 0.0001 );
  EXPECT_NEAR( Y(0, 1), T(0, 1), 0.0001 );
}

TEST(SigmoidDerivativeTests, zeroInput) {
  Eigen::MatrixXd X(1, 1);
  Eigen::MatrixXd T(1, 1);
  X << 0;
  T << 0.25;

  auto sigmoid = Activations::Sigmoid::create();
  EXPECT_EQ( sigmoid->gradient(X), T);
}

TEST(SigmoidDerivativeTests, limits) {
  Eigen::MatrixXd X(1, 2);
  Eigen::MatrixXd T(1, 2);
  Eigen::MatrixXd Y(1, 2);
  X << -10, 10;
  T << 0,   0;

  auto sigmoid = Activations::Sigmoid::create();
  Y = sigmoid->gradient(X);
  EXPECT_NEAR( Y(0, 0), T(0, 0), 0.0001 );
  EXPECT_NEAR( Y(0, 1), T(0, 1), 0.0001 );
}


int main (int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
