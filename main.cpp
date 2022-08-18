#include "nmf.h"

int main(int argc, char** argv)
{
    /*mat A ={{1,2,3},{2,2,4},{3,4,4}};
    colvec b = {1,2,3};
    colvec x0 = {1,1,1};
    colvec xr = CorGrant(x0,A,b);
    cout<<xr<<endl;*/

    /*mat A={{ 27.6433,24.6292,23.8429},
           {24.6292,26.8634,23.5002},
           {23.8429,23.5002,30.3976}};

    cout << A((1,2),(0,2))<<endl;*/

    /*colvec b ={50.1539,
               44.5837,
               43.4728};
    colvec x0 ={ 1.8138,
                 0.0005,
                 0.0307};
    colvec y0={ 0.5740,
            0.0085,
            0.6398};
    //cout<< A*x0-b<<endl;
    /*mat ATA = A.t() * A;
    colvec Ab = A.t() * b;*/
    //cout << innpot(A,b,x0,y0);
    //cout<<diagmat(y0/x0);

    /*colvec B = A.col(2);
    cout << B;
    umat A = {{1,3,4,2},{1,2,3,4}};
    vec B = {1,3,4,5};
    sp_mat X(A,B);
    cout << X;
    sp_colvec x = X.col(3);
    cout << x;
    uvec p = conv_to<uvec>::from(find(x));
    cout << p;
    cout << nonzeros(x);*/

    /*mat dt;
    dt = readCSV("iris.csv");
    //cout << dt << endl;
    //string md="cg";
    std::tuple<mat, mat> Res = m2NMF(dt,3, 1, 1);*/
    //print(Res);*/


    mat B;
    B= readCSV("ratings1.csv");
    B.col(1)-=1;
    B.col(0)-=1;
    //string md="cg";
    clock_t start,end;
    start = clock();
    std::tuple<mat, mat> Res = m3NMF_pal(B,4,1,1);
    end = clock();
    cout << "time = "<<double(end - start)/CLOCKS_PER_SEC<<"s"<<endl;
    //print(Res);

    /*mat dt = readCSV("ratings1.csv");
    int k =3;
    double lamh = 0.1;
    double lamw = 0.1;
    dt.col(0) = dt.col(0)-1;
    dt.col(1) = dt.col(1)-1;
    colvec data = dt.col(2);
    umat loc = conv_to<umat>::from(dt.cols(0,1).t());
    sp_mat V(loc,data);
    int n = V.n_rows;
    int m = V.n_cols;
    cout << n << ","<<m<<endl;
    mat W (n,k,fill::randu);
    mat H (k,m,fill::randu);
    colvec Ik (k,fill::ones);
    colvec y0 (k,fill::randu);
    int i = 0;
    sp_mat Vi = V.col(i);
    mat mat_w = W.rows(find(Vi));
    cout << mat_w.n_rows << ","<<mat_w.n_cols<<endl;
    mat A = mat_w.t() * mat_w;
    mat x0 = H.col(i);
    mat v0 = nonzeros(Vi);
    cout << v0.n_rows << ","<<v0.n_cols<<endl;
    //mat b = (mat_w.t() )* nonzeros(Vi) - 0.5 * lamh * Ik;
    //cout <<b<< endl;
    //H.col(i) = innpot( A, b, x0, y0);

    /*cout << "hi"<<endl;
    int j = 0;
    sp_mat Vj = V.row(j);
    mat mat_h = H.cols(find(Vj));
    cout << mat_h.n_rows << ","<<mat_h.n_cols<<endl;
    mat A = mat_h * mat_h.t();
    mat x0 = W.row(j).t();
    mat v = nonzeros(Vj);
    cout << v.n_rows << ","<<v.n_cols<<endl;
    mat b = mat_h * v;*/
    //b = mat_h * nonzeros(Vj)- 0.5 * lamw * Ik;
    //cout << b;
    //W.row(j) = innpot( A, b, x0, y0).t();

    //sp_mat Vj = V.row(1);
    //cout << H.cols(find(Vj));






    return 0;
}

