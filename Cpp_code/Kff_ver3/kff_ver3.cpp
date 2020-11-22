#include <iostream>
#include <ctime>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <cstring>

using namespace std;
using namespace std::chrono;

void print_1di(int* arr,int m){
	int i;

	cout << "[";
	for(i=0;i<m;i++){
		cout << arr[i] << " ";
	}
	cout << "]" << endl;
}

void print_1d(double* arr,int m){
	int i;

	cout << "[";
	for(i=0;i<m;i++){
		cout << arr[i] << " ";
	}
	cout << "]" << endl;
}
void print_2d(double* arr,int m,int n,int cut){
	int i,j;

	for(i=0;i<m;i++){
		if(cut !=0 && i> cut) break;
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
extern "C"
void kff_many(int n, int d, int x2i, double sigma, double* x1, double* dx1dr, int* ele1, int* x1_indices, double* pout){

	double sigma0=1.0,zeta=2.0,eps=1e-8;
//    cout << "start" << endl;
//    return;
	double dval,sigma2,sigma02;
	sigma2=sigma*sigma;
	sigma02=sigma0*sigma0;
    double x1_norm,x1_norm2,x1_norm3,x2_norm,x2_norm2,x2_norm3,
	       x1x2_norm,x1x2_dot,dx,d2,d1,dk_dD,d2d_dx1dx2;
	int i,j,k,l,TI,index,ii,jj;
//	int m,n,d,x2i;
	int _i,_j,_ele1,_ele2;
	int mempoint,memsize,memsizei;
    int *x1_inds;
    double *Cbig,*d2k_dx1dx2,*ans;
//    double *_x1,*_dx1dr,*_x2,*_dx2dr;
    int sizei=sizeof(int);
    int sized=sizeof(double);
    int sizec=sizeof(char);


//    std::cout << std::setprecision(15);
//    cout << n << " " << d << " " << x2i << endl;
//    cout <<x1_indices[0] << " " << x1_indices[x2i-1] << endl;
//   
//    print_1di(x1_indices,x2i);
//    return;
/*
	x1=new double[n*d];
	dx1dr=new double[n*d*3];
	ele1=new int[n];
	x1_indices=new int[x2i];
    mempoint=0;

    memsize=n*d*sized;
    memcpy(x1,memdata+mempoint,memsize);
    mempoint+=memsize;
    
    memsize=n*d*3*sized;
    memcpy(dx1dr,memdata+mempoint,memsize);
    mempoint+=memsize;    
    
    memsize=n*sizei;
    memcpy(ele1,memdata+mempoint,memsize);
    mempoint+=memsize; 
    
    memsize=x2i*sizei;
    memcpy(x1_indices,memdata+mempoint,memsize);
    mempoint+=memsize;     
*/       
	TI=0;
	for(i=0;i<x2i;i++) TI += x1_indices[i];
	x1_inds=new int[TI];
	index=0;
	for(i=0;i<x2i;i++){
		for(j=0;j<x1_indices[i];j++){
			x1_inds[index]=i;
			index++;
		}
	}
//	TI=x2i*3*x2i*3;
//    Cbig=new double[TI];
//    for(i=0;i<TI;i++) Cbig[i]=0;
//    _x1=new double[d];
//    _dx1dr=new double[d*3];
//    _x2=new double[d];
//    _dx2dr=new double[d*3];
    d2k_dx1dx2=new double[d*d];
    ans=new double[3*3];
    int dsized=d*sized;
    int d3sized=d*3*sized;
    int d3=d*3;

	for(ii=0;ii<n;ii++){
//		memcpy(_x1,x1+d*ii,dsized);
//		memcpy(_dx1dr,dx1dr+d3*ii,d3sized);
		_ele1=ele1[ii];
		_i=x1_inds[ii];

		dval=0.0;
		for(i=0;i<d;i++){
			dval+=x1[ii*d+i]*x1[ii*d+i];
		}
		x1_norm=sqrt(dval);
		if(x1_norm > eps){
			x1_norm2=x1_norm*x1_norm;
			x1_norm3=x1_norm2*x1_norm;

			for(jj=0;jj<n;jj++){
//				memcpy(_x2,x1+d*jj,dsized);
//				memcpy(_dx2dr,dx1dr+d3*jj,d3sized);
				_ele2=ele1[jj];
				_j=x1_inds[jj];
				dval=0.0;
				for(i=0;i<d;i++){
					dval+=x1[jj*d+i]*x1[jj*d+i];
				}
				x2_norm=sqrt(dval);
				if(_ele1==_ele2 && x2_norm > eps){
					x2_norm2=x2_norm*x2_norm;
					x2_norm3=x2_norm2*x2_norm;
					x1x2_norm = x1_norm*x2_norm;
					x1x2_dot=0;
					for(i=0;i<d;i++){
						x1x2_dot+=x1[ii*d+i]*x1[jj*d+i];
					}
					dx = x1x2_dot/x1x2_norm;
				    d2 = pow(dx,zeta-2);
				    d1 = dx*d2;
				    dk_dD = sigma2;

				    for(i=0;i<d;i++){
				    	for(j=0;j<d;j++){
				    	   dval=0;
				           if(i==j) dval=1.0;
				           d2d_dx1dx2=(dval-x1[jj*d+i]*x1[jj*d+j]/x2_norm2)/x1x2_norm+
				        		   (x1[ii*d+i]*x1[jj*d+j]*x1x2_dot/x2_norm2-x1[ii*d+i]*x1[ii*d+j])/x1_norm3/x2_norm;

				           d2k_dx1dx2[i*d+j]=(((x1[jj*d+i]/x1x2_norm-x1[ii*d+i]*x1x2_dot/x1_norm3/x2_norm)*
				        		               (x1[ii*d+j]/x1x2_norm-x1[jj*d+j]*x1x2_dot/x1_norm/x2_norm3) )*d2*(zeta-1) +
						                     d1*d2d_dx1dx2 )*zeta*dk_dD;
				    	}
				    }

				    for(k=0;k<3;k++){
				    	for(l=0;l<3;l++){
				    		dval=0;
				            for(i=0;i<d;i++){
				    	        for(j=0;j<d;j++){
				    				//dval+=_dx1dr[i*3+k]*d2k_dx1dx2[i*d+j]*_dx2dr[j*3+l];
				    				dval+=dx1dr[(ii*d+i)*3+k]*d2k_dx1dx2[i*d+j]*dx1dr[(jj*d+j)*3+l];
				    			}
				    		}
				            ans[k*3+l]=dval;
				    	}
				    }


				    for(i=_i*3;i<(_i+1)*3;i++){
				    	for(j=_j*3;j<(_j+1)*3;j++){
				    	//	Cbig[i*x2i*3+j]+=ans[(i-(_i*3))*3+j-(_j*3)];
				    		pout[i*x2i*3+j]+=ans[(i-(_i*3))*3+j-(_j*3)];
				    	
				    	}
				    }



				} //if(_ele1==_ele2 && x2_norm > eps)
			}//for(jj=0;jj<n;jj++)
		}//if(x1_norm > eps){
	}//for(ii=0;ii<n;ii++)
//	std::cout << std::scientific;
//	print_2d(Cbig,x2i*3,x2i*3,5);
//    return;

//	TI=x2i*3*x2i*3;
//    for(i=0;i<TI;i++) memcpy(pout+i*sized,Cbig+i,sized);
//    for(i=0;i<TI;i++) pout[i]=Cbig[i];
     
//    cout << "before end" << endl;
//    delete Cbig;
    delete d2k_dx1dx2;
    delete ans;
    delete x1_inds;
    
};



int main() {
    double *ptemp1,*ptemp2,*ptemp3;
    int *itemp1,*itemp2;
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	kff_many(1,2,3,4,ptemp1,ptemp2,itemp1,itemp2,ptemp3);
//	kff_many(int n, int d, int x2i, double sigma, double* x1, double* dx1dr, int* ele1, int* x1_indices, double* pout)
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

    std::cout << "time in C " << time_span.count() << " sec.";
    std::cout << std::endl;
    return 0;
}

