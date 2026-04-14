
#include <Eigen/Dense>


namespace Loss {

// Loss -- base class
class Loss {
public:
  Loss();
  virtual double function (Eigen::MatrixXd target, Eigen::MatrixXd predict) = 0;
  static std::shared_ptr<Loss> create();
}; // end Loss -- base class

class MAE : public Loss {
public:
  using Loss::Loss;
  double function (Eigen::MatrixXd target, Eigen::MatrixXd predict) override;
  static std::shared_ptr<MAE> create();
}; // end MAE

class MSE : public Loss {
public:
  using Loss::Loss;
  double function (Eigen::MatrixXd target, Eigen::MatrixXd predict) override;
  static std::shared_ptr<MSE> create();
}; // end MSE

class RMSE : public Loss {
public:
  using Loss::Loss;
  double function (Eigen::MatrixXd target, Eigen::MatrixXd predict) override;
  static std::shared_ptr<RMSE> create();
}; // end RMSE

} // namespace Loss
