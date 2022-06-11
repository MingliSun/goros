//
// Created by sun on 2022/1/6.
//

#ifndef AUTOFUSION_PRODUCER_H
#define AUTOFUSION_PRODUCER_H
#include <string>
#include<vector>
#include<ctime>
#include<random>
#include<cassert>
#include <cstring>
#include"Affine.h"
#include"Polynomial.h"
using namespace std;
static string variable_name[][5]={
        "n", "c","d", "h","w",
        "oc","ic", "kd","kh", "kw",
};
const vector<vector<float>> interpolation_points={
        //invalid
        {},
        //01 {E=4.63E-08 on conv2d  [1]}
        {},
        // 02 {E=7.65E-08 on F( 2,3) [1]}
        {0, -1, 1},
        // 03 {E=2.35E-07 on F( 3,3) [1]}
        {0, -1, 1, 1.0 / 2},
        // 04 {E=3.29E-07 on F( 4,3) [1]}
        {0, -1, 1, 1.0/ 2, -2},
        // 05 {E=6.81E-07 on F( 5,3) [1]}
        {0, -1, 1, 1.0/ 2, -2, -1.0/ 2},
        // 06 {E=8.79E-07 on F( 6,3) [1]}
        {0, -1, 1, 1.0/ 2, -1.0/ 2, 2, -2},
        // 07 {E=3.71E-06 on F( 7,3) [1]}
        {0, -1, 1, 1.0/ 2, -1.0/ 2, 2, -2, -1.0/ 4},
        // 08 {E=7.35E-06 on F( 8,3) [1]}
        {0, -1, 1, 1.0/ 2, -1.0/ 2, 2, -2, -1.0/ 4, 4},
        // 09 {E=2.20E-05 on F( 9,3) [1]}
        {0, -1, 1, 1.0/ 2, -1.0/ 2, 2, -2, -1.0/ 4, 3.0/ 4, -4.0/ 3},
        // 10 {E=3.22E-05 on F(10,3) [1]}
        {0, -1, 1, 1.0/ 2, -1.0/ 2, 2, -2, -1.0/ 4, 4, 3.0/ 4, -4.0/ 3},
        // 11 {E=1.09E-04 on F(11,3) [1]}
        {0, -1, 1, 1.0/ 2, -1.0/ 2, 2, -2, -1.0/ 4, 4, 3.0/ 4, -4.0/ 3, 1.0/ 4},
        // 12 {E=1.99E-04 on F(12,3) [1]}
        {0, -1, 1, 1.0/ 2, -1.0/ 2, 2, -2, -1.0/ 4, 4, 1.0/ 4, -3.0/ 4, 4.0/ 3, -4},
        // 13 {E=5.54E-04 on F(13,3) [1]}
        {0, -1, 1, 1.0/ 2, -1.0/ 2, 2, -2, -1.0/ 4, 4, 1.0/ 4, -3.0/ 4, 4.0/ 3, 3.0/ 4, -4.0/ 3},
        // 14 {E=8.80E-04 on F(14,3) [1]}
        {0, -1, 1, 1.0/ 2, -1.0/ 2, 2, -2, -1.0/ 4, 4, 1.0/ 4, -3.0/ 4, 4.0/ 3, -4, 3.0/ 4, -4.0/ 3},
        // 15 {E=1.07E-02 on F(15,3) [1]}
        {0, -1, 1, 1.0/ 2, -1.0/ 2, 2, -2, -1.0/ 4, 4, 1.0/ 4, -3.0/ 4, 4.0/ 3, -4, 2.0 / 3, -3.0/ 2, 3.0/ 2},
        // 16 {E=1.93E-02 on F(16,3) [1]}
        {0, -1, 1, 1.0/ 2, -1.0/ 2, 2, -2, -1.0/ 4, 4, 1.0/ 4, -3.0/ 4, 4.0/ 3, -4, 2.0 / 3, -3.0/ 2, -2.0/ 3, 3.0/ 2},

};
class Producer{
public:

    static string get_unique_variable_name();
    static string get_unique_tensor_name();
    static string get_debug_variable_name(int row,int column);
    static string get_fused_operator_name();
    static int var_num ;
    static int tensor_num;
    static int fused_num;
    static vector<vector<int>> var_debug_num;
    //static int second;
    template < typename  T>
    static void* allocate_iconv(const vector<int>& shape){
        int n = shape.size();
        assert(n==4);
        assert(shape[0]==shape[1]);
        int size = shape[0]*shape[1]*shape[2]*shape[3];
        T *result = new T [size];
        for(int oc=0;oc<shape[0];oc++){
            for(int ic=0;ic<shape[1];ic++){
                for(int kh=0;kh<shape[2];kh++){
                    for(int kw=0;kw<shape[3];kw++){
                        int index = kw+kh*shape[3]+ic*shape[3]*shape[2]+oc*shape[3]*shape[2]*shape[1];
                        if(oc==ic&&kh==shape[2]/2&&kw==shape[3]/2) result[index] = 1;
                        else result[index] = 0;
                    }
                }
            }
        }
        return result;
    }
    template < typename  T>
    static void* allocate_iconv(const vector<Variable>& shape){
        int n = shape.size();
        assert(n==4);
        assert(shape[0].get_upper_bound()==shape[1].get_upper_bound());
        vector<int> size = {shape[0].get_upper_bound(),shape[1].get_upper_bound(),shape[2].get_upper_bound(),shape[3].get_upper_bound()};
        return allocate_iconv<T>(size);
    }
    template < typename  T>
    static void* allocate_sequential(const vector<int>& shape){
        int size = 1;
        for(auto& v:shape){
            size*= v;
        }
        T* result =  new T[size];
        for(int i=0;i<size;i++){
            result[i] = i+1;
        }
        return result;
    }

    template < typename  T>
    static void *allocate_sequential(const vector<Variable>& shape){
        //assert(!shape.empty());
        int size = 1;
        for(auto& v:shape){
            size*= v.get_upper_bound();
        }
        T* result =  new T[size];
        for(int i=0;i<size;i++){
            result[i] = i+1;
        }
        return result;
    }

    template < typename  T>
    static void* allocate(const vector<Variable>& shape,float value){
        int size = 1;
        for(auto& v:shape){
            size*= v.get_upper_bound();
        }
        T* result = new T[size]{reinterpret_cast<T&>(value)};
        return result;
    }
    template < typename  T>
    static void* allocate(const vector<Variable>& shape) {
        //allocate garbage
        int size = 1;
        for(auto& v:shape){
            size*= v.get_upper_bound();
        }
        void* result = new T[size];
        return result;
    }
    template < typename  T>
    static void* allocate_bernoulli(const vector<Variable>& shape,float prob){
        //todo use c++11 bernoulli_distribution ----done
        //assert(!shape.empty());
        int size = 1;
        for(auto& v:shape){
            size*= v.get_upper_bound();
        }
        T* result = new T[size];
        //srand((int)time(0));
        //for(int i=0;i<size;i++){
        //if(rand() < int(RAND_MAX * prob)) result[i] = 1;
        //else result[i] = 0;
        //}
        default_random_engine e(time(nullptr));
        bernoulli_distribution b(prob);
        for(int i=0;i<size;i++){
            result[i] = b(e);
        }
        return result;
    }
    template < typename  T>
    static void* allocate_winograd_A(const vector<float>& a,const size_t m,const size_t n){
        T* result = new T[m*n];
        for(size_t i=0;i<m;i++){
            for(size_t j=0;j<n;j++){
                size_t index = i*n+j;
                if(i==m-1&&j==n-1){
                    result[index] = 1;
                }else if(i==m-1){
                    result[index] = 0;
                }else{
                    result[index] = pow(a[i],j);
                }
            }
        }
        return result;
    }
    template < typename  T>
    static void* allocate_winograd_B(const vector<float>& a,const int alpha){
        T * f = reinterpret_cast<T*>(allocate_F<T>(a,alpha)); // shape: (alpha, alpha)
        if(f[0]<0){
            for(int j=0;j<alpha;j++) f[j] *=-1;
        }
        T* b = reinterpret_cast<T*>(allocate_B<T>(a,alpha));
        T* result = new T[alpha*alpha];
        transpose(f,alpha,alpha);
        dot(b,f,alpha,alpha,alpha,result);
        delete [] f;
        delete [] b;
        return result;
    }
    template < typename  T>
    static void* allocate_winograd_G(const vector<float>& a,const int alpha,const int r) {
        T* result = new T[alpha*r];
        T* g = reinterpret_cast<T*>(allocate_winograd_A<T>(a,alpha,r));
        transpose(g,alpha,r);
        T * f = reinterpret_cast<T*>(allocate_F<T>(a,alpha)); // shape: (alpha, alpha)
        if(f[0]<0){
            for(int j=0;j<alpha;j++) f[j] *=-1;
        }
        T* inv = reinterpret_cast<T*>(inverse(f,alpha));
        dot(g,inv,r,alpha,alpha,result);
        transpose(result,r,alpha);// shape : alpha, r
        delete [] g;
        delete [] f;
        delete [] inv;
        return result;
    }

    template < typename  T>
    static void* allocate_F(const vector<float>& a,const int n){
        T* result = new T[n*n]{0};
        for(int i=0;i<n-1;i++){
            for(int j=0;j<n-1;j++){
                if(i==j){
                    float total=1.0;
                    for(int k=0;k<n-1;k++){
                        if(k!=i) total*= (a[i]-a[k]);
                    }
                    int index = i*n+j;
                    result[index] = total;
                }
            }
        }
        result[n*n-1] = 1;
        return result;
    }
    template < typename  T>
    static void* allocate_B(const vector<float>& a,const int n) {
        T* result = new T[n*n]{0};
        T* f = new T[(n-1)*(n-1)];
        for(int i=0;i<n-1;i++){
            float total = 1.0;
            for(int k=0;k<n-1;k++) {
                if (k != i) total *= (a[i] - a[k]);
            }
            for(int j=0;j<n-1;j++){
                Poly1d poly;
                for(int k=0;k<n-1;k++){
                    if(k!=i) poly = poly* Poly1d({-a[k],1});
                }
                f[i*(n-1)+j] = poly.get_coef(j)/total; //better use float!!
            }
        }
        T* t = new T[(n-1)*n]{0};
        for(int i=0;i<n-1;i++){ //np.eye
            int index = i*n+i;
            t[index] = 1;
        }
        for(int i=0;i<n-1;i++){
            int index = i*n+n-1; //last column
            t[index] = -pow(a[i],n-1);
        }
        transpose(f,n-1,n-1);

        dot(f,t,n-1,n-1,n,result);
        result[n*n-1] = 1;
        // delete intermedia variable
        delete [] f;
        delete [] t;
        return result;

    }
    template < typename  T>
    static void dot(T* matrix_a,T* matrix_b,int m,int z,int n,T* result){//binary
        for(int i=0;i<m;i++){
            for(int k=0;k<z;k++){
                for(int j=0;j<n;j++){
                    int index_r = i*n+j;
                    int index_a = i*z+k;
                    int index_b = k*n+j;
                    result[index_r] += matrix_a[index_a]*matrix_b[index_b];
                }
            }
        }
    }
    template < typename  T>
    static void transpose(T* matrix,int m,int n){//todo transpose at own position --done
     // [m,n]  ===> [n,m]
        for(int i=0;i<m*n;i++){
            int next = (i%n)*m+i/n;
            while(next>i)
                next = (next%n)*m+next/n;
            if(next==i){
                T temp = matrix[i];
                int cur = i;
                int pre = (i%m)*n+i/m;
                while(pre!=i){
                    matrix[cur] = matrix[pre];
                    cur = pre;
                    pre = (pre%m)*n+pre/m;
                }
                matrix[cur] = temp;
            }
        }
    }
    template < typename  T>
    static void LUP_decomposition(T* matrix,int N,T* L,T* U,int* P){
        int row=0;
        for(int i=0;i<N;i++){
            P[i] = i;
        }
        for(int i=0;i<N-1;i++){
            T p =0.0;
            for(int j=i;j<N;j++){
                if(abs(matrix[j*N+i]>p)){
                    p = abs(matrix[j*N+i]);
                    row = j;
                }
            }
            swap(P[i],P[row]);
            for(int j=0;j<N;j++){
                swap(matrix[i*N+j],matrix[row*N+j]);
            }
            T l,u=matrix[i*N+i];
            for(int j=i+1;j<N;j++){
                l = matrix[j*N+i]/u;
                matrix[j*N+i] = l;
                for(int k=i+1;k<N;k++){
                    matrix[j*N+k]-= matrix[i*N+k]*l;
                }
            }
        }
        for(int i=0;i<N;i++){
            for(int j=0;j<=i;j++){
                if(i!=j) L[i*N+j] = matrix[i*N+j];
                else L[i*N+j]=1;
            }
            for(int k=i;k<N;k++){
                U[i*N+k] = matrix[i*N+k];
            }
        }
    }
    template < typename  T>
    static void* LUP_solve(T* L,T* U,int* P,T* b,int N){
        T* x = new T[N];
        T* y = new T[N];
        for(int i=0;i<N;i++){
            y[i] = b[P[i]];
            for(int j=0;j<i;j++){
                y[i] -= L[i*N+j]*y[i];
            }
        }

        for(int i=N-1;i>=0;i--){
            x[i] = y[i];
            for(int j=N-1;j>i;j--){
                x[i] -= U[i*N+j]*x[j];
            }
            x[i] /=U[i*N+i];
        }
        delete [] y;
        return x;
    }

    template < typename  T>
    static void* inverse(T* A,int N){
        T* mirror = new T[N*N];
        T* inv_A = new T[N*N];
        T* inv_A_each;
        T* b = new T[N];
        for(int i=0;i<N;i++){
            T* L = new T[N*N];
            T* U = new T[N*N];
            int*P = new int[N];
            for(int j=0;j<N;j++){
                b[j] = 0;
            }
            b[i] = 1;
            //copy
            for(int j=0;j<N*N;j++) mirror[j] = A[j];
            LUP_decomposition(A,N,L,U,P);
            inv_A_each = reinterpret_cast<T*>(LUP_solve(L,U,P,b,N));
            memcpy(inv_A+i*N,inv_A_each,N*sizeof(T));
            delete [] L;
            delete [] U;
            delete [] P;
        }
        transpose(inv_A,N,N);
        delete [] mirror;
        delete [] inv_A_each;
        return inv_A;
    }


};


class Math{
public:
    static vector<int> get_concrete_index(const vector<Variable>& vars,int n);
    static int get_concrete_index(const vector<Variable>& vars,const vector<int>& integers);
    static vector<int> get_concrete_index(const vector<Affine>& exprs,const vector<Variable>& vars,const vector<int>& integers);
    static vector<int> get_concrete_index(const vector<Variable>& out,const vector<Variable>& in,const vector<int>& src);
    static int floordiv(int a, int b);
    static int modulo(int a,int b);
    static int greatest_common_divisor(int a,int b);
    static int greatest_common_divisor(const vector<int>& v,int b);
    static float modulo(float a,float b);
};


#endif //AUTOFUSION_PRODUCER_H
