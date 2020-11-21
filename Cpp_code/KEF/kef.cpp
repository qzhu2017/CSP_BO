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

void print_3d(double* arr,int m,int n,int d,int cut){

	int i,j,k;

	for(i=0;i<m;i++){
		if(cut !=0 && i> cut) break;
		if(i==0) cout << "[[";
		else cout << " [";
		for(j=0;j<n;j++){
		   if(j==0) cout << "[";
		   else cout << "  [";
		   for(k=0;k<d;k++){
		      cout << arr[(i*n+j)*d+k] << " ";
		   }
		   if(j==n-1) cout << "]";
		   else cout << "]" << endl;
		}
		if(i==m-1) cout << "]]" << endl;
		else cout << "]" << endl;
		cout << endl;
	}
}

void print_4d(double* arr,int m,int n,int d,int h,int cut){

	int i,j,k,l;

	for(i=0;i<m;i++){
		if(cut !=0 && i> cut) break;
		if(i==0) cout << "[[";
		else cout << " [";
		for(j=0;j<n;j++){
		   if(j==0) cout << "[";
		   else cout << "  [";
		   for(k=0;k<d;k++){
			   if(k==0) cout << "[";
			   else cout << "   [";
			   for(l=0;l<h;l++){
		         cout << arr[((i*n+j)*d+k)*h+l] << " ";
			   }
			   if(k==d-1) cout << "]";
			   else cout << "]" << endl;
		   }
		   if(j==n-1) cout << "]";
		   else cout << "]" << endl << endl;
		}
		if(i==m-1) cout << "]]" << endl;
		else cout << "]" << endl;
		cout << endl;
	}
}

double *kef_single(double* x1, double* x2, double* dx2dr, int* x2_indices, double sigma, double l, double zeta, int M, int N, int P, int n, double eps){
    // M: length of x1; 
    // N: length of x2; 
    // P: length of descriptors;

    double sigma2, l2;
    sigma2 = sigma*sigma;
    l2 = l*l;

    double* x1_norm = new double[M];
    double* x2_norm = new double[N];
    double* x2_norm_sq = new double[N];
    double* x1x2_dot = new double[M*N];
    double* d = new double[M*N];
    double* D = new double[M*N];
    double* zD = new double[M*N]; // zeta * D
    double* k = new double[M*N];
    double* dk_dD = new double[M*N];
    double* dd_dx2 = new double[M*N*P];
    double* dD_dx2 = new double[M*N*P];
    double* kef_ = new double[M*N*P*3];
    double* kef_j = new double[N*3];
    double* kef = new double[n*3];

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
        x2_norm_sq[i] = res;
        x2_norm[i] = sqrt(res);
    }
    
    // Calculate x1x2_dot, d, D, and k
    for (unsigned int i=0;i<M;i++){
        for (unsigned int j=0;j<N;j++){
            double res = 0.;
            for (unsigned int k=0;k<P;k++){
                res += x1[i*P+k] * x2[j*P+k];
            }
            x1x2_dot[i*N+j] = res;
            d[i*N+j] = res / (eps+x1_norm[i]*x2_norm[j]);
            D[i*N+j] = pow(d[i*N+j], zeta);
            zD[i*N+j] = zeta * D[i*N+j] / d[i*N+j];
            k[i*N+j] = sigma2 * exp((D[i*N+j]-1.0)/(2.0*l2));
            dk_dD[i*N+j] = -1 * k[i*N+j] / (2.0*l2);  // in the original python script this is negative
        }
    }

    for (unsigned int i=0;i<M;i++){
        for (unsigned int j=0;j<N;j++){
            for (unsigned int k=0;k<P;k++){
                dd_dx2[(i*N+j)*P+k] = ((x1[i*P+k] * x2_norm[j]) - (x1x2_dot[i*N+j] * x2[j*P+k] / x2_norm[j])) / (x1_norm[i] * x2_norm_sq[j]);
                dD_dx2[(i*N+j)*P+k] = -1 * dd_dx2[(i*N+j)*P+k] * zD[i*N+j];
                for (unsigned int q=0; q<3; q++){
                    kef_[((i*N+j)*P+k)*3+q] = -1 * dD_dx2[(i*N+j)*P+k] * dx2dr[(j*P+k)*3+q]; 
                }
            }
        }
    }

    // Sum M and P
    for (unsigned int j=0;j<N;j++){
        for (unsigned int q=0; q<3; q++){
            double res1 = 0.;
            for (unsigned int i=0;i<M;i++){
                double res2 = 0.;
                for (unsigned int k=0;k<P;k++){
                    res2 += kef_[((i*N+j)*P+k)*3+q];
                }
                res1 += res2 * dk_dD[i*N+j];
            }
            kef_j[j*3+q] = res1/M;
        }
    }

    for (unsigned int q=0; q<3; q++){
        int count = 0;
        for (unsigned int i=0; i<n; i++){
            int n_tmp = x2_indices[i];
            double res = 0.;
            for (unsigned int j=0; j<n_tmp; j++){
                res += kef_j[(count+j)*3+q];
            }
            kef[i*3+q] = res;
            count += n_tmp;
        }
    }
    
    return kef;
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
    double* dx2dr = new double[N*D*3];
    
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

    // Fill dx2dr with random numbers
    int count = 0;
    for (unsigned int i=0; i<N; i++){
        for (unsigned int j=0; j<D; j++){
            for (unsigned int q=0; q<3; q++){
                dx2dr[count] = doubleDist(rnd);
                count++;
            }
        }
    }

    double sigma=28.835, l=1., zeta=2.0, eps=1e-8;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    double *kef = kef_single(x1, x2, dx2dr, x2_indices, sigma, l, zeta, M, N, D, n, eps);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

    print_2d(kef, n, 3, 0);

    std::cout << "time in C " << time_span.count() << " sec.";
    std::cout << std::endl;
    return 0;
}
