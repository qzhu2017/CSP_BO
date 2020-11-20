#include <random>
#include <iostream> // for standard input/output stream
#include <ctime>    // for timing
#include <cmath>    // for math such as exp
#include <chrono>   // track time, what is the difference between chrono and ctime?
#include <cstdio>   // for fopen() ?, also includes C keywords instead of C++
#include <iomanip>  // for setprecision()
#include <cstring>  // for dealing with C style strings. 
#include <iterator> // required for std::size

using namespace std;
using namespace std::chrono; // for the clock

void print_1d_int(int* arr, int m){
    int i;

    cout << "[";
    for(i=0;i<m;i++){
        cout << arr[i] << " ";
    }
    cout << "]" << endl;
}

void print_1d(double* arr, int m){
    int i;

    cout << "[";
    for(i=0;i<m;i++){
        cout << arr[i] << " ";
    }
    cout << "]" << endl;
}

void print_2d(double* arr, int m, int n, int cut){
    int i, j;

    for(i=0; i<m; i++){
        if(cut!=0 && i>cut) break;
        if(i==0) cout << "[[";
        else cout << " [";
        
        for(j=0;j<n;j++){
            cout << arr[i*n+j] << " ";
        }

        if(i==m-1) cout << "]]" << endl;
        else cout << "]" << endl;
    }
}

void print_2di(int* arr,int m,int n){
	int i,j;

	for(i=0;i<m;i++){
		if(i==0) cout << "[[";
		else cout << " [";
		for(j=0;j<n;j++){
		   cout << arr[i*n+j] << " ";
		}
		if(i==m-1) cout << "]]" << endl;
		else cout << "]" << endl;
	}
}

double *kee_single(double* x1, double* x2, int* x2_indices, double sigma, double l, double zeta, int M, int N, int P, double eps){
    // M: length of x1; 
    // N: length of x2; 
    // P: length of descriptors;
    //int n = sizeof(x2_indices)/sizeof(x2_indices[0]);

    int n = 5; // how to get the length of x2_indices?
    //int n = sizeof(x2_indices)/sizeof(x2_indices[0]);
    //cout << n << endl;
    
    double sigma2, l2;
    sigma2 = sigma*sigma;
    l2 = l*l;

    double* x1_norm = new double[M];
    double* x2_norm = new double[N];
    double* x1x2_dot = new double[M*N];
    double* d = new double[M*N];
    double* D = new double[M*N];
    double* k = new double[M*N];
    double* kee = new double[n];

    // x1_norm
    for(unsigned int i=0;i<M;i++){
        double res = 0.;
        for(unsigned int j=0;j<P;j++){
            res += x1[i*P+j] * x1[i*P+j];
        }
        x1_norm[i] = sqrt(res);
    }

    // x2_norm
    for(unsigned int i=0;i<N;i++){
        double res = 0.;
        for(unsigned int j=0;j<P;j++){
            res += x2[i*P+j] * x2[i*P+j];
        }
        x2_norm[i] = sqrt(res);
    }

    // Calculate x1x2_dot, d, D, and k
    for(unsigned int i=0;i<M;i++){
        for(unsigned int j=0;j<N;j++){
            double res = 0.;
            for (unsigned int k=0;k<P;k++){
                res += x1[i*P+k] * x2[j*P+k];
            }
            x1x2_dot[i*N+j] = res;
            d[i*N+j] = res / (eps+x1_norm[i]*x2_norm[j]);
            D[i*N+j] = pow(d[i*N+j], zeta);
            k[i*N+j] = sigma2 * exp((D[i*N+j]-1.0)/(2.0*l2));
        }
    }

    // Sum M
    double* kee_j = new double[N];
    for (unsigned int j=0;j<N;j++){
        double res = 0.;
        for (unsigned int i=0;i<M; i++){
            res += k[i*N+j];
        }
        kee_j[j] = res/M;
    }

    // Sum N based on x2_indices
    int count = 0;
    for (unsigned int i=0; i<n; i++){
        int n_tmp = x2_indices[i];
        double res = 0.;
        for (unsigned int j=0; j<n_tmp; j++){
            res += kee_j[count+j];
        }
        kee[i] = res;
        count += n_tmp;
    }
    return kee;
}

int main() {
    // Declarate constant variables
    const auto M = 5, N = 20, D = 5, n = 5;
    int x2_indices[n] = {2, 4, 6, 3, 5};
        
    // Random numbers
    mt19937_64 rnd;
    uniform_real_distribution<double> doubleDist(0,1);

    // Create arrays that represent the matrices x1 and x2
    double* x1 = new double[M*D];
    double* x2 = new double[N*D];
    
    // Fill x1 with random numbers
    for (unsigned int i=0; i<M; i++){
        for (unsigned int j=0; j<D; j++){
            x1[i*D+j] = doubleDist(rnd);
        }
    }

    // Fill x2 with random numbers
    for (unsigned int i=0; i<N; i++){
        for (unsigned int j=0; j<D; j++){
            x2[i*D+j] = doubleDist(rnd);
        }
    }

    double sigma=28.835, l=1., zeta=2.0, eps=1e-8;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    double *kee = kee_single(x1, x2, x2_indices, sigma, l, zeta, M, N, D, eps);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

    print_1d(kee, 5);

    std::cout << "time in C " << time_span.count() << " sec.";
    std::cout << std::endl;
    return 0;
}
