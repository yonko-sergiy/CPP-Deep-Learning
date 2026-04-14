
#include <Eigen/Dense>

namespace Loss {

class Loss {
public:
  Loss();
  virtual double function (Eigen::MatrixXd target, Eigen::MatrixXd predict) = 0;
  static std::shared_ptr<Loss> create();
};

class MAE : public Loss {
public:
  using Loss::Loss;
  double function (Eigen::MatrixXd target, Eigen::MatrixXd predict) override;
  static std::shared_ptr<MAE> create();
};

class MSE : public Loss {
public:
  using Loss::Loss;
  double function (Eigen::MatrixXd target, Eigen::MatrixXd predict) override;
  static std::shared_ptr<MSE> create();
};

class RMSE : public Loss {
public:
  using Loss::Loss;
  double function (Eigen::MatrixXd target, Eigen::MatrixXd predict) override;
  static std::shared_ptr<RMSE> create();
};




}
