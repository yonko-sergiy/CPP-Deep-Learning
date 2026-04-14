
#include "loss.h"
#include <cmath>

namespace Loss {

Loss::Loss() {}

double MAE::function (Eigen::MatrixXd target, Eigen::MatrixXd predict) {
  Eigen::MatrixXd difference = (target.array() - predict.array()).cwiseAbs();
  return (difference.sum() / difference.size());
}

std::shared_ptr<MAE> MAE::create() {
  return std::make_shared<MAE>();
}


double MSE::function (Eigen::MatrixXd target, Eigen::MatrixXd predict) {
  Eigen::MatrixXd difference = (target.array() - predict.array()).cwiseSquare();
  return difference.sum() / difference.size();
}

std::shared_ptr<MSE> MSE::create() {
  return std::make_shared<MSE>();
}


double RMSE::function (Eigen::MatrixXd target, Eigen::MatrixXd predict) {
  Eigen::MatrixXd difference = (target.array() - predict.array()).cwiseSquare();
  return std::sqrt(difference.sum() / difference.size());
}

std::shared_ptr<RMSE> RMSE::create() {
  return std::make_shared<RMSE>();
}

}
