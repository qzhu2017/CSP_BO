#include <iostream>
#include <ctime>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <iomanip>

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

void kff_many(){

	double sigma=28.835,sigma0=0.01,zeta=2.0,eps=1e-8;

	double dval,sigma2,sigma02;
	sigma2=sigma*sigma;
	sigma02=sigma0*sigma0;

	int i,j,k,l,ID,TI,ii,l2;
	int m,n,d,x2i,m1;
    int *mi,*ele_all,*x2_indices,*ele1,*x2_inds;
    double *x2,*dxdr_all,*x1,*dx1dr,*Cbig,
	       *dk_dD,*D1,*D2,*x1_x1_norm3,*d2d_dx1dx2,*tmp33,*tmp,
		   *dd_dx1,*dd_dx2,*d2k_dx1dx2,*_kff1,*kff,*_C;
    double *x1_norm,*x1_norm2,*x1_norm3,*x2_norm,*x2_norm2,*x2_norm3,*x1x2_dot;
    int sizei=sizeof(int);
    int sized=sizeof(double);

    std::cout << std::setprecision(15);

	FILE * filp = fopen("c.dat", "rb");
	if(filp==NULL){
		cout << "file open error" << endl;
		fclose(filp);
		return;
	}
	int bytes_read = fread(&n, sizeof(int), 1, filp);
	if(bytes_read <=0 ){
		cout << "file read error" << endl;
		fclose(filp);
		return;
	}
	fread(&d, sizei, 1, filp);
	fread(&x2i, sizei, 1, filp);
	fread(&m1, sizei, 1, filp);
	mi=new int[m1];
	fread(mi, sizei, m1, filp);

    kff=new double[n*3*3];
	x2=new double[n*d];
	dxdr_all=new double[n*d*3];
	ele_all=new int[n];
	x2_indices=new int[x2i];
	fread(x2, sized, n*d, filp);
	fread(dxdr_all, sized, n*d*3, filp);
	fread(ele_all, sizei, n, filp);
	fread(x2_indices, sizei, x2i, filp);

	x2_inds=new int[x2i*2];
	x2_inds[0]=0,x2_inds[1]=x2_indices[0];

	for(i=1;i<x2i;i++){
		ID=x2_inds[(i-1)*2+1];
		x2_inds[i*2]=ID,x2_inds[i*2+1]=ID+x2_indices[i];
	}

    TI=m1*n*3*3;
	Cbig=new double[TI];
	for(i=0;i<TI;i++) Cbig[i]=0;


    x2_norm= new double[n];
    x2_norm2= new double[n];
    x2_norm3= new double[n];
	for(i=0;i<n;i++){
		dval=0.0;
		for(j=0;j<d;j++){
			dval+=x2[i*d+j]*x2[i*d+j];
		}
		x2_norm[i]=sqrt(dval)+eps;
		x2_norm2[i]=pow(x2_norm[i],2);
		x2_norm3[i]=pow(x2_norm[i],3);
	}
	tmp33=new double[n*d*d];
	for(i=0;i<n;i++){
		for(j=0;j<d;j++){
			for(k=0;k<d;k++){
				if(j==k)
				  tmp33[(i*d+j)*d+k]=1.0-x2[i*d+j]*x2[i*d+k]/x2_norm2[i];
				else
				  tmp33[(i*d+j)*d+k]=-x2[i*d+j]*x2[i*d+k]/x2_norm2[i];
			}
		}
	}
	int max_m=31;
	x1=new double[max_m*d];
	dx1dr=new double[max_m*d*3];
	ele1=new int[max_m];
    dk_dD=new double[max_m*n];
    x1_norm= new double[max_m];
    x1_norm2= new double[max_m];
    x1_norm3= new double[max_m];
	x1x2_dot=new double[max_m*n];
	D1=new double[max_m*n];
	D2=new double[max_m*n];
	x1_x1_norm3=new double[max_m*d];
	dd_dx1=new double[max_m*n*d];
	dd_dx2=new double[max_m*n*d];
	d2d_dx1dx2=new double[max_m*n*d*d];
	d2k_dx1dx2=new double[max_m*n*d*d];
	_kff1=new double[n*d*3];

	for(ii=0;ii<m1;ii++){
		m=mi[ii];

		fread(x1, sized, m*d, filp);
		fread(dx1dr, sized, m*d*3, filp);
		fread(ele1, sizei, m, filp);


        for(i=0;i<m;i++){
        	for(j=0;j<n;j++){
        	  if(ele1[i]-ele_all[j] !=0) dk_dD[i*n+j]=0;
        	  else dk_dD[i*n+j]=sigma2;
        	}
        }

///////////////kff///////////

		for(i=0;i<m;i++){
			dval=0.0;
			for(j=0;j<d;j++){
				dval+=x1[i*d+j]*x1[i*d+j];
			}
			x1_norm[i]=sqrt(dval);
			x1_norm2[i]=pow(x1_norm[i],2);
			x1_norm3[i]=pow(x1_norm[i],3);
		}


		for(i=0;i<m;i++){
			for(j=0;j<n;j++){
				dval=0.0;
				for(k=0;k<d;k++){
					dval+=x1[i*d+k]*x2[j*d+k];
				}
				x1x2_dot[i*n+j]=dval;
				dval=dval/(eps+x1_norm[i]*x2_norm[j]);
				D2[i*n+j]=pow(dval,zeta-2);
				D1[i*n+j]=dval*D2[i*n+j];
			}
		}

		for(i=0;i<m;i++){
			for(j=0;j<d;j++){
				x1_x1_norm3[i*d+j]=x1[i*d+j]/x1_norm3[i];
			}
		}

		for(i=0;i<m;i++){
			for(j=0;j<n;j++){
				for(k=0;k<d;k++){
					dd_dx2[(i*n+j)*d+k]=
						(x1[i*d+k]*x2_norm[j]-x1x2_dot[i*n+j]*x2[j*d+k]/x2_norm[j])
					    /x1_norm[i]/x2_norm2[j];
					dd_dx1[(i*n+j)*d+k]=
						(x2[j*d+k]*x1_norm[i]-x1x2_dot[i*n+j]*x1[i*d+k]/x1_norm[i])
						/x1_norm2[i]/x2_norm[j];
				}
			}
		}

		for(i=0;i<m;i++){
			for(j=0;j<n;j++){
				for(k=0;k<d;k++){
					for(l=0;l<d;l++){
						d2d_dx1dx2[((i*n+j)*d+k)*d+l]=
                                tmp33[(j*d+k)*d+l]/x1_norm[i]/x2_norm[j]
								-x1[i*d+l]/x2_norm[j]*x1_x1_norm3[i*d+k]
							   +x1_x1_norm3[i*d+k]*x2[j*d+l]/x2_norm3[j]*x1x2_dot[i*n+j];
/// d2k_dx1dx2=d2k_dx1dx2*dk_dD
						d2k_dx1dx2[((i*n+j)*d+k)*d+l]=
							(dd_dx1[(i*n+j)*d+k]*dd_dx2[(i*n+j)*d+l]*D2[i*n+j]*(zeta-1)
							  +D1[i*n+j]*d2d_dx1dx2[((i*n+j)*d+k)*d+l])*zeta*dk_dD[i*n+j];


					}
				}
			}
		}

		for(i=0;i<n;i++){
			for(j=0;j<d;j++){
				for(k=0;k<3;k++){
					dval=0;
					for(l=0;l<m;l++){
						for(l2=0;l2<d;l2++){
							dval+= dx1dr[(l*d+l2)*3+k]*d2k_dx1dx2[((l*n+i)*d+l2)*d+j];
						}
					}
					_kff1[(i*d+j)*3+k]=dval;
				}
			}
		}

		for(i=0;i<n;i++){
			for(j=0;j<3;j++){
				for(k=0;k<3;k++){
					dval=0;
					for(l=0;l<d;l++){
						dval+= _kff1[(i*d+l)*3+j]*dxdr_all[(i*d+l)*3+k];
					}
					kff[(i*3+j)*3+k]=dval;
					Cbig[((ii*n+i)*3+j)*3+k]=dval;
				}
			}
		}

///////////////end kff////////

	}

	delete x1;
	delete dx1dr;
	delete ele1;
	delete dk_dD;
	delete x1x2_dot;
	delete D1;
	delete D2;
	delete d2d_dx1dx2;
	delete d2k_dx1dx2;
	delete _kff1;
	delete dd_dx1;
	delete dd_dx2;
	delete x1_norm;
	delete x1_norm2;
	delete x1_norm3;
	delete x1_x1_norm3;

	 TI=m1*3*x2i*3;
	_C=new double[m1*3*x2i*3];
	for(i=0;i<TI;i++) _C[i]=0;
	int ind0,ind1;
	for(ii=0;ii<x2i;ii++){
		ind0=x2_inds[ii*2];
		ind1=x2_inds[ii*2+1];
		tmp=new double[m1*3*3];
		for(i=0;i<m1;i++){
			for(j=0;j<3;j++){
				for(k=0;k<3;k++){
					dval=0;
					for(l=ind0;l<ind1;l++){
						dval+= Cbig[((i*n+l)*3+j)*3+k];
					}
					tmp[(i*3+j)*3+k]=dval;
				}
			}
		}

		for(i=0;i<m1;i++){
			for(j=i*3;j<(i+1)*3;j++){
				for(k=ii*3;k<(ii+1)*3;k++){
					_C[j*(x2i*3)+k]=tmp[(i*3+j-i*3)*3+k-ii*3];
				}
			}
		}

		delete tmp;
	}
	fclose(filp);

	std::cout << std::scientific ;
	print_2d(_C,m1*3,x2i*3,8);

	delete x2,x2_norm,x2_norm2,x2_norm3,x2_inds,
	     mi,Cbig,dxdr_all,ele_all,x2_indices,tmp33,kff;

};



int main() {

	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	kff_many();
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

    std::cout << "time in C " << time_span.count() << " sec.";
    std::cout << std::endl;
    return 0;
}

