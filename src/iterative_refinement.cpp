#include <iostream>
#include <iomanip>
#include <cmath>
#include <tuple>
#include <vector>
#include <Eigen/LU> 
#include <Eigen/SVD>





/*Название статьи:  Uchino Yuki, Terao Takeshi, Ozaki Katsuhisa. 2022.08.05 – Acceleration of 
Iterative Refinement for Singular Value Decomposition. 10.21203/rs.3.rs1931986/v1
* Обязательные условия: 1)в матрицах колво строк больше или равному количеству столбцов
*2)каждое сингулярное значение больше последующего (d1>d2>...>dn)
* 3)каждое приближенное сингулярное значение отлично друг от друга (di != dj для i!=j)
* 
* Выполнил Едренников Д.А. КМБО-01-20
*/



std::tuple<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
MSVD_SVD(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &A, const  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &Ui, const  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &Vi)
{
    const int m = A.rows();
    const int n = A.cols();

    if ( A.rows()< A.cols())
    {
        std::cout << "Attention! Number of the rows must be greater or equal  than  number of the columns";
        return std::make_tuple(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Constant(m, m, 0.0), Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Constant(n, n, 0.0),
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Constant(m, n, 0.0));
    }

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Ad = A.cast<double>();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Ud = Ui.cast<double>();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Vd = Vi.cast<double>();

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> P = Ad * Vd;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Q = Ad.transpose() * Ud.block(0,0,m, n);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> ViT = Vd.transpose();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> UiT = Ud.transpose();
    std::vector<double> r (n, 0.0);
    std::vector<double> s (n, 0.0);
    std::vector<double> t (n, 0.0);
    Eigen::Matrix<double, Eigen::Dynamic,Eigen::Dynamic> Sigma_n(n,n);
    Sigma_n.setZero();

    for (int i = 0; i < n; i++)
    {
        r[i] = 1.0 - UiT.row(i) * Ud.col(i);
        s[i] = 1.0 - ViT.row(i) * Vd.col(i);
        t[i] = UiT.row(i) *P.col(i);
        Sigma_n(i,i) = t[i] / (1 - ((r[i] + s[i]) * 0.5));

    };

    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Sigma(m,n);
    Sigma.setZero();
    Sigma.block(0,0,n, n) = Sigma_n.transpose();

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Cgamma = P - Ud.block(0,0,m, n) * Sigma_n;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Cdelta = Q - Vd * Sigma_n;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Calpha = Ud.block(0,0, m, n).transpose() * Cgamma;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Cbetta = ViT * Cdelta;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> D = Sigma_n * Calpha + Cbetta * Sigma_n;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> E = Calpha * Sigma_n + Sigma_n * Cbetta;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> G(n,n);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> F11(n, n);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            G(i, j) = D(i, j) / (Sigma_n(j,j) * Sigma_n(j,j) - Sigma_n(i,i) * Sigma_n(i,i));
            F11(i, j) = E(i, j) / (Sigma_n(j,j) * Sigma_n(j,j) - Sigma_n(i,i) * Sigma_n(i,i));
        }
    }

    for (int i = 0; i < n; i++)
    {
        G(i, i) = s[i] * 0.5;
        F11(i, i) = r[i] * 0.5;
    }


    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic > F12(n, m-n);
    F12 = Sigma_n.inverse() * P.transpose() * Ud.block(0, n, m, m - n);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> F21 = Ud.block(0,n-1,m, m - n).transpose() * Cgamma * Sigma_n.inverse();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic > F22 = 0.5 * (Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Constant(m - n, m - n, 1.0) -
        Ud.block(0,n-1,m, m - n).transpose() * Ud.block(0, n, m, m - n));


    
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic > F(m,m);
    F.block(0, 0, n, n) = F11;
    F.block(0, n , n, m-n) = F12;
    F.block(n, 0, m-n, n) = F21;
    F.block(n , n, m - n, m - n) = F22;


    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic > U = Ud+ Ud*F;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic > V = Vd + Vd * G;


   return std::make_tuple(U, V, Sigma);
}

std::tuple<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
Accurate_BDCSVD(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &A)
{
    Eigen::BDCSVD< Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    return  MSVD_SVD(A, svd.matrixU(), svd.matrixV());
}

int main()
{
    using namespace std;
    using namespace Eigen;
    Eigen::Matrix<float, Dynamic, Dynamic> A(10,9);
    A << 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29, 30, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 45,
        46, 47, 48, 49, 50, 51, 52, 53, 54,
        55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 66, 67, 68, 68, 70, 71, 72,
        73, 74, 75, 76, 77, 78, 79, 80, 81,
        3, 9, (float)4.98942, (float)0.324235,  443534, 345, (float)56.543853, (float)450.435234, (float)43.34353221;

   BDCSVD<MatrixXf> svd(A, ComputeFullU | ComputeFullV);
   std::tuple<Eigen::Matrix<double, 10, 10>, Eigen::Matrix<double, 9, 9>, Eigen::Matrix<double, 10, 9>> a =
       MSVD_SVD(A, svd.matrixU(), svd.matrixV());
   std::cout << get<0>(a)*get<2>(a)*get<1>(a).transpose() <<"\n" << "\n";
    Array<float,1, Dynamic> sigm(9);
    sigm  = svd.singularValues();
    Eigen::Matrix<float, Dynamic, Dynamic > I(10, 9);
    I.setZero();
    I.block(0,0,9,9) = sigm.matrix().asDiagonal();
    Eigen::Matrix<float, Dynamic, Dynamic > U = svd.matrixU();
    cout << U*I* svd.matrixV().transpose() << "\n" << "\n";

    return 0;
}

