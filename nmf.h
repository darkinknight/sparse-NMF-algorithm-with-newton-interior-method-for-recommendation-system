//
// Created by Administrator on 2022/7/18.
//

#ifndef RESCODE_NMF_H
#define RESCODE_NMF_H

#endif //RESCODE_NMF_H
#include <iostream>
#include <armadillo>
#include <tuple>
#include <math.h>
#include <time.h>

using namespace std;
using namespace arma;

mat CorGrant(mat x0, mat A, mat b , double err=1e-3)
{
    mat r0 = A * x0 - b;
    mat p0 = -r0;
    mat rk = r0;
    mat pk = p0;
    mat xk = x0;
    mat rr;
    mat ak;
    mat aj;
    while (norm(rk) >= err)
    {   rr = rk.t() * rk;
        ak = rr / (pk.t() * A * pk);
        xk = xk + ak(0,0) * pk;
        rk = rk +ak(0,0) * (A * pk);
        aj = (rk.t() * rk) / rr;
        pk = -rk + aj(0,0) * pk;
        cout << norm(rk)<<endl;
    }
    return xk;
}

double theta(colvec x, colvec u,double tk = 1)
{
    if (min(u) < 0){
        colvec xu = -x/u;
        tk = min(xu.elem(find(u<0)));
    }
    return tk;
}

colvec innpot(mat A, colvec b, colvec x0, colvec y0, string method = "direct" ,double tol1=1e-3,double tol2 = 1e-5, int maxiter = 100)
{   colvec x = x0;
    colvec y = y0;
    int n = A.n_rows;
    int iter;
    mat t1;
    double t2,theta0;
    for (iter = 0 ; iter < maxiter ; iter += 1){
        mat mu = ( x.t() * y ) / (n * n);
        mat cgA = A + diagmat( y/x ) ;
        mat cgB = mu(0,0) / x - A * x + b;
        mat u;
        if (method == "direct"){
            u = solve (cgA,cgB );
            if (rcond(cgA)<1e-15){
                cout<<"rcond:"<<rcond(cgA)<<endl;
                cout<<"cgA:"<<cgA<<endl;
                cout<<"cgB:"<<cgB<<endl;
            }
        }
        else if (method == "cg"){
            colvec t0 ( n, fill::randu);
            u = CorGrant(t0, cgA, cgB );
        }
        else{
            cout << "option not exists"<<endl;
            return 0;
        }
        mat v = -y + mu(0,0)/x - u % y / x;
        double theta_max = min(theta( x, u),theta(y,v));
        mat uv = u.t() * v;
        mat xy = x.t() * y;
        if (uv(0,0) <= 0) {
            theta0 = 0.9995 * theta_max;
        }
        else if (uv(0,0) > 0){
            double theta_hat = (xy(0,0) - n * mu(0,0))/uv(0,0);
            theta0 = 0.9995 * min(theta_max,theta_hat);
        }
        x = x + theta0 * u;
        y = y + theta0 * v;
        t1 = x.t() * y;
        t2 = norm(A * x - b- y);
        if ((t1(0,0) < tol1 && t2 < tol1)||(norm(v) < tol2)||(norm(u) < tol2)){
            break;
        }
    }
    if(iter == maxiter){
        cout <<"exceed maximum iteration"<<endl;
        cout << t1(0,0)<<","<<t2<<endl;
        cout<<"x:"<<x0<<endl;
        cout<<"y:"<<y0<<endl;
        cout<<"A:"<<A<<endl;
        cout<<"B:"<<b<<endl;
    }
    return x;
}


colvec FBS(colvec z, mat Q, colvec C,double gamma, double lambda,int maxiter = 3){
    for(int iter = 0; iter < maxiter; iter ++){
        colvec grad = Q * z + C;
        colvec M = z - gamma * grad;
        M.for_each([](mat::elem_type & val){val = val < 0 ? 0 : val;});
        z = ( 1 - lambda ) * z + lambda * M;
    }
    return z;
}

colvec DRS(colvec z, mat Q,colvec C,double gamma,double lambda,int maxiter = 3){
    uword k = Q.n_rows;
    mat I {k,k,fill::eye};
    for(int iter = 0; iter <maxiter; iter++){
        mat U = inv( I + gamma * Q);
        z = (1-lambda) * z + lambda * U * (z - gamma * C);
        z.for_each([](mat::elem_type & val){val = val < 0 ? 0 : val;});
    }
    return z;
}


colvec innpot_FBS(mat A_inn, colvec b_inn, colvec x0_inn, colvec y0_inn, string method_inn = "direct" ,double tol_inn=1e-3, int maxiter_inn = 100, double gamma_fbs = 0.002, double lambda_fbs = 1, int maxiter_fbs = 5){
    colvec x = innpot(A_inn, b_inn, x0_inn, y0_inn, method_inn ,tol_inn, maxiter_inn);
    x = FBS(x,A_inn,-b_inn,gamma_fbs,lambda_fbs,maxiter_fbs);
    return x;
}

double m3NMF_loss(mat dt,mat W,mat H){
    mat WH = W * H;
    double loss = 0;
    for(int iter = 0; iter < dt.n_rows;iter ++){
        unsigned int row = dt(iter,0);
        unsigned int col = dt(iter,1);
        loss += pow(dt(iter,2)- WH(row,col),2);
    }
    return loss;
}
std::tuple<mat, mat> m3NMF(mat dt,int k, double lamw,double lamh,string method = "direct",int maxiter = 1000,double tol=1e-3){
    colvec data = dt.col(2);
    umat loc = conv_to<umat>::from(dt.cols(0,1).t());
    sp_mat V(loc,data);
    int n = V.n_rows;
    int m = V.n_cols;
    cout <<"row&col:"<< n << "," << m<<endl;
    mat W (n,k,fill::randu);
    mat H (k,m,fill::randu);
    colvec Ik (k,fill::ones);
    colvec y0 (k,fill::randu);
    double loss0 = m3NMF_loss(dt,W,H);
    cout<<"loss0:"<<loss0<<endl;
    int iter;
    double loss=loss0,loss1;
    for (iter = 0; iter < maxiter ; iter ++){
        for(int i = 0; i < m; i ++){
            //cout <<"i:"<< i<<endl;
            sp_mat Vi = V.col(i);
            mat mat_w = W.rows(find(Vi));
            mat A = mat_w.t() * mat_w;
            mat x0 = H.col(i);
            mat b = mat_w.t() * nonzeros(Vi) - 0.5 * lamh * Ik;
            H.col(i) = innpot( A, b, x0, y0,method);
        }
        for(int j = 0; j < n; j ++){
            //cout << "j:"<<j <<endl;
            sp_mat Vj = V.row(j);
            mat mat_h = H.cols(find(Vj));
            mat A = mat_h * mat_h.t();
            mat x0 = W.row(j).t();
            mat b = mat_h * nonzeros(Vj) - 0.5 * lamw * Ik;
            W.row(j) = innpot( A, b, x0, y0,method).t();
        }
        loss1 = m3NMF_loss(dt,W,H);
        cout<<"loss:"<<loss1<<endl;
        if((loss0-loss1) < tol){
            break;
        }
        loss0 = loss1;
    }
    cout<<"iter:"<<iter<<",err:"<<loss1<<endl;
    return std::make_tuple(W, H);
}

std::tuple<mat, mat> m3NMF_fbs(mat dt,int k, double lamw,double lamh,string method = "direct",int maxiter = 1000,double tol=1e-3,int thread = 4){
    const int NUM_THREAD = thread;
    colvec data = dt.col(2);
    umat loc = conv_to<umat>::from(dt.cols(0,1).t());
    sp_mat V(loc,data);
    int n = V.n_rows;
    int m = V.n_cols;
    cout << n << "," << m;
    mat W (n,k,fill::randu);
    mat H (k,m,fill::randu);
    colvec Ik (k,fill::ones);
    colvec y0 (k,fill::randu);
    double loss0 = m3NMF_loss(dt,W,H);
    cout<<"loss0:"<<loss0<<endl;
    double loss=loss0,loss1;
    int st = 0;
    sp_mat Vi,Vj;
    mat mat_w, mat_h,A,x0,b;
    for (int iter = 0; iter < maxiter ; iter ++){
# pragma omp parallel for num_threads(NUM_THREAD) private(Vi,A,x0,b,mat_w)
        for(int i = 0; i < m; i ++){
            Vi = V.col(i);
            mat_w = W.rows(find(Vi));
            A = mat_w.t() * mat_w;
            x0 = H.col(i);
            b = (mat_w.t() )* nonzeros(Vi) - 0.5 * lamh * Ik;
            if (st == 0){
                H.col(i) = innpot( A, b, x0, y0,method);}
            else if (st == 1){
                H.col(i) = innpot_FBS( A, b, x0, y0,method);
            }
        }
# pragma omp parallel for num_threads(NUM_THREAD) private(Vj,A,x0,b,mat_h)
        for(int j = 0; j < n; j ++){
            Vj = V.row(j);
            mat_h = H.cols(find(Vj));
            A = mat_h * mat_h.t();
            x0 = W.row(j).t();
            b = mat_h * nonzeros(Vj) - 0.5 * lamw * Ik;
            if (st == 0){
                W.row(j) = innpot( A, b, x0, y0,method).t();}
            else if(st == 1){
                W.row(j) = innpot_FBS(A, b, x0, y0,method).t();
            }
        }
        loss1 = m3NMF_loss(dt,W,H);
        cout<<"loss:"<<loss1<<endl;
        if((loss0-loss1) < tol){
            st = 1;
        }
        if (st == 1){
            break;
        }
        loss0 = loss1;
    }
    return std::make_tuple(W, H);
}


std::tuple<mat, mat> m3NMF_pal(mat dt,int k, double lamw,double lamh,string method = "direct",int maxiter = 1000,double tol=1e-3,int thread = 4){
    const int NUM_THREAD = thread;
    colvec data = dt.col(2);
    umat loc = conv_to<umat>::from(dt.cols(0,1).t());
    sp_mat V(loc,data);
    int n = V.n_rows;
    int m = V.n_cols;
    cout <<"row&col:"<< n << "," << m<<endl;
    mat W (n,k,fill::randu);
    mat H (k,m,fill::randu);
    colvec Ik (k,fill::ones);
    colvec y0 (k,fill::randu);
    double loss0 = m3NMF_loss(dt,W,H);
    cout<<"loss0:"<<loss0<<endl;
    int iter;
    double loss=loss0,loss1;
    sp_mat Vi,Vj;
    mat mat_w, mat_h,A,x0,b;
    for (iter = 0; iter < maxiter ; iter ++){
# pragma omp parallel for num_threads(NUM_THREAD) private(Vi,A,x0,b,mat_w)
        for(int i = 0; i < m; i ++){
            //cout <<"i:"<< i<<endl;
            Vi = V.col(i);
            mat_w = W.rows(find(Vi));
            A = mat_w.t() * mat_w;
            x0 = H.col(i);
            b = mat_w.t() * nonzeros(Vi) - 0.5 * lamh * Ik;
            H.col(i) = innpot( A, b, x0, y0,method);
        }
# pragma omp parallel for num_threads(NUM_THREAD) private(Vj,A,x0,b,mat_h)
        for(int j = 0; j < n; j ++){
            //cout << "j:"<<j <<endl;
            Vj = V.row(j);
            mat_h = H.cols(find(Vj));
            A = mat_h * mat_h.t();
            x0 = W.row(j).t();
            b = mat_h * nonzeros(Vj) - 0.5 * lamw * Ik;
            W.row(j) = innpot( A, b, x0, y0,method).t();
        }
        loss1 = m3NMF_loss(dt,W,H);
        cout<<"loss:"<<loss1<<endl;
        if((loss0-loss1) < tol){
            break;
        }
        loss0 = loss1;
    }
    cout<<"iter:"<<iter<<",err:"<<loss1<<endl;
    return std::make_tuple(W, H);
}

std::tuple<mat, mat> m2NMF(mat dt,int k, double lamw, double lamh, double tol = 1e-3, int maxiter = 15,string method = "direct"){
    int n = dt.n_rows;
    int m = dt.n_cols;
    mat W (n,k,fill::randu);
    mat H (k,m,fill::randu);
    colvec Ik (k,fill::ones);
    colvec y0 (k,fill::randu);
    double loss0 = norm(dt - W * H);
    //cout<<loss0<<endl;
    int iter;
    double loss=loss0,loss1;
    for(iter = 0; iter < maxiter ; iter ++){
        for(int i = 0; i < m; i ++){
            mat Vi = dt.col(i);
            mat A = W.t() * W;
            mat x0 = H.col(i);
            mat  b = W.t() * Vi - 0.5 * lamh * Ik;
            H.col(i) = innpot(A,b,x0,y0,method);
        }
        for(int j = 0; j < n; j ++){
            mat Vj = dt.row(j);
            mat A = H * H.t();
            mat x0 = W.row(j).t();
            mat b = H * Vj.t() - 0.5 * lamw * Ik;
            W.row(j) = innpot(A,b,x0,y0,method).t();
        }
        loss1 = norm(dt - W * H);
        //cout<<"loss:"<<loss1<<endl;
        if((loss0-loss1)/loss < tol){
            break;
        }
        loss0 = loss1;
    }
    cout<<"iter:"<<iter<<",err:"<<loss1<<endl;
    return std::make_tuple(W, H);
}


std::tuple<mat, mat> m2NMF_fbs(mat dt,int k, double lamw, double lamh,string method = "direct" ,double tol = 1e-3, int maxiter = 100,int thread = 4){
    const int NUM_THREAD = thread;
    int n = dt.n_rows;
    int m = dt.n_cols;
    mat W (n,k,fill::randu);
    mat H (k,m,fill::randu);
    colvec Ik (k,fill::ones);
    colvec y0 (k,fill::randu);
    double loss0 = norm(dt - W * H);
    //cout<<loss0<<endl;
    int iter;
    double loss=loss0,loss1;
    mat Vj,Vi,A,x0,b;
    int st = 0;
    for(iter = 0; iter < maxiter ; iter ++){
# pragma omp parallel for num_threads(NUM_THREAD) private(Vi,A,x0,b)
        for(int i = 0; i < m; i ++){
            Vi = dt.col(i);
            A = W.t() * W;
            x0 = H.col(i);
            b = W.t() * Vi - 0.5 * lamh * Ik;
            if (st == 0){
                H.col(i) = innpot(A,b,x0,y0,method);
            }
            else if(st == 1){
                H.col(i) = innpot_FBS(A,b,x0,y0,method);
            }
        }
# pragma omp parallel for num_threads(NUM_THREAD) private(Vj,A,x0,b)
        for(int j = 0; j < n; j ++){
            Vj = dt.row(j);
            A = H * H.t();
            x0 = W.row(j).t();
            b = H * Vj.t() - 0.5 * lamw * Ik;
            if(st == 0){
                W.row(j) = innpot(A,b,x0,y0,method).t();
            }
            else if(st == 1){
                W.row(j) = innpot_FBS(A,b,x0,y0,method).t();
            }
        }
        loss1 = norm(dt - W * H);
        cout<<"loss:"<<loss1<<endl;
        if((loss0-loss1)/loss < tol){
            st = 1;
        }
        if (st == 1){
            break;
        }
        loss0 = loss1;
    }
    cout<<"iter:"<<iter<<",err:"<<loss1<<endl;
    return std::make_tuple(W, H);
}


std::tuple<mat, mat> m2NMF_pal(mat dt,int k, double lamw, double lamh, double tol = 1e-4, int maxiter = 200,string method = "direct",int thread = 4){
    const int NUM_THREAD = thread;
    int n = dt.n_rows;
    int m = dt.n_cols;
    mat W (n,k,fill::randu);
    mat H (k,m,fill::randu);
    colvec Ik (k,fill::ones);
    colvec y0 (k,fill::randu);
    double loss0 = norm(dt - W * H);
    //cout<<loss0<<endl;
    int iter;
    double loss=loss0,loss1;
    for(iter = 0; iter < maxiter ; iter ++){
        mat Vj,Vi,A,x0,b;
# pragma omp parallel for num_threads(NUM_THREAD) private(Vi,A,x0,b)
        for(int i = 0; i < m; i ++){
            Vi = dt.col(i);
            A = W.t() * W;
            x0 = H.col(i);
            b = W.t() * Vi - 0.5 * lamh * Ik;
            H.col(i) = innpot(A,b,x0,y0,method);
        }
# pragma omp parallel for num_threads(NUM_THREAD) private(Vj,A,x0,b)
        for(int j = 0; j < n; j ++){
            Vj = dt.row(j);
            A = H * H.t();
            x0 = W.row(j).t();
            b = H * Vj.t() - 0.5 * lamw * Ik;
            W.row(j) = innpot(A,b,x0,y0,method).t();
        }
        loss1 = norm(dt - W * H);
        cout<<"loss:"<<loss1<<endl;
        if((loss0-loss1)/loss < tol){
            break;
        }
        loss0 = loss1;
    }
    cout<<"iter:"<<iter<<",err:"<<loss1<<endl;
    return std::make_tuple(W, H);
}




/* tools for importing data*/
mat readCSV(const std::string &filename, const std::string &delimeter = ",")
{
    std::ifstream csv(filename);
    std::vector<std::vector<double>> datas;

    for(std::string line; std::getline(csv, line); ) {

        std::vector<double> data;

        // split string by delimeter
        auto start = 0U;
        auto end = line.find(delimeter);
        while (end != std::string::npos) {
            data.push_back(std::stod(line.substr(start, end - start)));
            start = end + delimeter.length();
            end = line.find(delimeter, start);
        }
        data.push_back(std::stod(line.substr(start, end)));
        datas.push_back(data);
    }

    arma::mat data_mat = arma::zeros<arma::mat>(datas.size(), datas[0].size());

    for (int i=0; i<datas.size(); i++) {
        arma::mat r(datas[i]);
        data_mat.row(i) = r.t();
    }

    return data_mat;
}

template<class Tuple, std::size_t N>
struct TuplePrinter {
    static void print(const Tuple& t)
    {
        TuplePrinter<Tuple, N-1>::print(t);
        std::cout << ", " << std::get<N-1>(t);
    }
};
template<class Tuple>
struct TuplePrinter<Tuple, 1> {
    static void print(const Tuple& t)
    {
        std::cout << std::get<0>(t);
    }
};
template<typename... Args, std::enable_if_t<sizeof...(Args) == 0, int> = 0>
void print(const std::tuple<Args...>& t)
{
    std::cout << "()\n";
}
template<typename... Args, std::enable_if_t<sizeof...(Args) != 0, int> = 0>
void print(const std::tuple<Args...>& t)
{
    std::cout << "(";
    TuplePrinter<decltype(t), sizeof...(Args)>::print(t);
    std::cout << ")\n";
}
