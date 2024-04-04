#include <iostream>
#include <iomanip>
#include <cmath>
#include <tuple>
#include <vector>

#if __has_include(<Eigen/LU>)
#include <Eigen/LU> 
#elif __has_include(<eigen3/Eigen/LU>)
#include <eigen3/Eigen/LU>
#endif


#if __has_include(<Eigen/SVD>)
#include <Eigen/SVD> 
#elif __has_include(<eigen3/Eigen/SVD>)
#include <eigen3/Eigen/SVD>
#endif


/*�������� ������:  Uchino Yuki, Terao Takeshi, Ozaki Katsuhisa. 2022.08.05 � Acceleration of
Iterative Refinement for Singular Value Decomposition.
(https://www.researchgate.net/publication/362642883_Acceleration_of_Iterative_Refinement_for_Singular_Value_Decomposition)
* ������������ �������: 1)� �������� ����� ����� ������ ��� ������� ���������� ��������
*2)������ ����������� �������� ������ ������������ (d1>d2>...>dn)
* 3)������ ������������ ����������� �������� ������� ���� �� ����� (di != dj ��� i!=j)
*
* �������� ���������� �.�. ����-01-20
*/

template<typename T, int M, int N>
class AOIR_SVD;

template<typename T, int M, int N>
class AOIR_SVD
{
private:
    Eigen::Matrix<T, M, M> U;
    Eigen::Matrix<T, M, N> S;
    Eigen::Matrix<T, N, N> V;


    void Set_U(Eigen::Matrix<T, M, M> A)
    {
        U = A;
    }

    void Set_V(Eigen::Matrix<T, N, N> A)
    {
        V = A;
    }

    void Set_S(Eigen::Matrix<T, M, N> A)
    {
        S = A;
    }

protected:
    AOIR_SVD  MSVD_SVD(const Eigen::Matrix<T, M, N>& A, const  Eigen::Matrix<T, M, M>& Ui, const  Eigen::Matrix<T, N, N>& Vi)
    {
        const int m = A.rows();//�������� ������� ����������� �������
        const int n = A.cols();


        if (A.rows() < A.cols())
        {
            std::cout << "Attention! Number of the rows must be greater or equal  than  number of the columns";
            AOIR_SVD ANS;
            return ANS;
        }

        using matrix_dd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
        using matrix_nn = Eigen::Matrix<double, N, N>; //V
        using matrix_mn = Eigen::Matrix<double, M, N>; //A
        using matrix_mm = Eigen::Matrix<double, M, M>; //U

        matrix_mn Ad = A.cast<double>();//�������� ������ ��������� ��� ����� ��� ���������� ������ 
        matrix_mm Ud = Ui.cast<double>();//������������ � ������� ���������, ������� ������������ ���
        matrix_nn Vd = Vi.cast<double>();//���� ������ � double
        //����� ��� ��� ���� ��������� ����� ����� double. �� � �����, ��� ������� ����� ��������� �
        //������������ ����.

        matrix_mn Udmn = Ud.block(0, 0, m, n);


        matrix_mn P = Ad * Vd;
        matrix_nn Q = Ad.transpose() * Udmn;

        matrix_nn ViT = Vd.transpose();
        matrix_mm UiT = Ud.transpose();
        std::vector<double> r(n, 0.0);
        std::vector<double> s(n, 0.0);
        std::vector<double> t(n, 0.0);
        matrix_nn Sigma_n(n, n);
        Sigma_n.setZero();

        for (int i = 0; i < n; i++)
        {
            r[i] = 1.0 - UiT.row(i) * Ud.col(i);
            s[i] = 1.0 - ViT.row(i) * Vd.col(i);
            t[i] = UiT.row(i) * P.col(i);
            Sigma_n(i, i) = t[i] / (1 - ((r[i] + s[i]) * 0.5));

        };

        matrix_mn Sigma(m, n);//������� ����������� ��������
        Sigma.setZero();
        Sigma.block(0, 0, n, n) = Sigma_n.transpose();

        matrix_mn Cgamma = P - Udmn * Sigma_n;
        matrix_nn Cdelta = Q - Vd * Sigma_n;

        matrix_nn Calpha = Udmn.transpose() * Cgamma;
        matrix_nn Cbetta = ViT * Cdelta;

        matrix_nn D = Sigma_n * Calpha + Cbetta * Sigma_n;
        matrix_nn E = Calpha * Sigma_n + Sigma_n * Cbetta;

        matrix_nn G(n, n);//�������, ����������� ��� ����� ������� ���������� ������ ����������� ��������
        matrix_nn F11(n, n);

        double temp1;//��������� ����������, ����������� ��� ����� �������� ���������� ��������� ������
        double temp2;//G � F11

        for (int i = 0; i < n; i++)//���������� ��������� ������ G � F11
        {
            temp1 = Sigma_n(i, i) * Sigma_n(i, i);

            for (int j = 0; j < n; j++)
            {
                temp2 = Sigma_n(j, j) * Sigma_n(j, j);

                G(i, j) = D(i, j) / (temp2 - temp1);//�������� ������������ ��������� �� ������ ��������� 
                F11(i, j) = E(i, j) / (temp2 - temp1);//�������������. ��� ����� �����������
            }
        }

        for (int i = 0; i < n; i++)//�������� ������������ ��������� ��� ������ G � F11
        {
            G(i, i) = s[i] * 0.5;
            F11(i, i) = r[i] * 0.5;
        }


        matrix_dd F12(n, m - n);
        F12 = Sigma_n.inverse() * P.transpose() * Ud.block(0, n, m, m - n);
        matrix_dd F21 = Ud.block(0, n - 1, m, m - n).transpose() * Cgamma * Sigma_n.inverse();
        matrix_dd F22 = 0.5 * (matrix_dd::Constant(m - n, m - n, 1.0) -
            Ud.block(0, n - 1, m, m - n).transpose() * Ud.block(0, n, m, m - n));



        matrix_mm F(m, m); //�������, ����������� ��� ����� ������� ���������� ����� ����������� ��������
        F.block(0, 0, n, n) = F11;
        F.block(0, n, n, m - n) = F12;
        F.block(n, 0, m - n, n) = F21;
        F.block(n, n, m - n, m - n) = F22;


        matrix_mm U = Ud + Ud * F;//���������� ���������� �������� ����� ����������� ��������
        matrix_nn V = Vd + Vd * G;//���������� ���������� �������� ������ ����������� ��������

        AOIR_SVD<T, M, N> ANS;
        ANS.Set_V(V.cast<T>());// ���������� ������ � ������������ ����
        ANS.Set_U(U.cast<T>());
        ANS.Set_S(Sigma.cast<T>());
        return ANS;
    }
public:

    AOIR_SVD() {};

    AOIR_SVD(Eigen::Matrix<T, M, N> A)
    {
        Eigen::BDCSVD< Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        //�������� ������ ����� � ������ ����������� �������� � ������� ������� �� ���������� Eigen
        AOIR_SVD<T, M, N> temp;
        temp = MSVD_SVD(A, svd.matrixU(), svd.matrixV());// ��������� ���������� ����� �������� 
        this->U = temp.matrixU();
        this->V = temp.matrixV();
        this->S = temp.singularValues();
    }


    Eigen::Matrix<T, N, N> matrixV()
    {
        return V;
    }

    Eigen::Matrix<T, M, M> matrixU()
    {
        return U;
    }

    Eigen::Matrix<T, M, N> singularValues()
    {
        return S;
    }

};