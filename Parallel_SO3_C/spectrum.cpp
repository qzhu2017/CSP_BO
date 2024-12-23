
//#include "f2c.h"
//#include "clapack.h"
#include <cmath>
#include <iostream>
#include <complex.h>
#include <tgmath.h>
#include <string.h>

using namespace std;

double* Wigner_data;

double fac_arr[]={
		  1,
		  1,
		  2,
		  6,
		  24,
		  120,
		  720,
		  5040,
		  40320,
		  362880,
		  3628800,
		  39916800,
		  479001600,
		  6227020800,
		  87178291200,
		  1307674368000,
		  20922789888000,
		  355687428096000,
		  6.402373705728e+15,
		  1.21645100408832e+17,
		  2.43290200817664e+18,
		  5.10909421717094e+19,
		  1.12400072777761e+21,
		  2.5852016738885e+22,
		  6.20448401733239e+23,
		  1.5511210043331e+25,
		  4.03291461126606e+26,
		  1.08888694504184e+28,
		  3.04888344611714e+29,
		  8.8417619937397e+30,
		  2.65252859812191e+32,
		  8.22283865417792e+33,
		  2.63130836933694e+35,
		  8.68331761881189e+36,
		  2.95232799039604e+38,
		  1.03331479663861e+40,
		  3.71993326789901e+41,
		  1.37637530912263e+43,
		  5.23022617466601e+44,
		  2.03978820811974e+46,
		  8.15915283247898e+47,
		  3.34525266131638e+49,
		  1.40500611775288e+51,
		  6.04152630633738e+52,
		  2.65827157478845e+54,
		  1.1962222086548e+56,
		  5.50262215981209e+57,
		  2.58623241511168e+59,
		  1.24139155925361e+61,
		  6.08281864034268e+62,
		  3.04140932017134e+64,
		  1.55111875328738e+66,
		  8.06581751709439e+67,
		  4.27488328406003e+69,
		  2.30843697339241e+71,
		  1.26964033536583e+73,
		  7.10998587804863e+74,
		  4.05269195048772e+76,
		  2.35056133128288e+78,
		  1.3868311854569e+80,
		  8.32098711274139e+81,
		  5.07580213877225e+83,
		  3.14699732603879e+85,
		  1.98260831540444e+87,
		  1.26886932185884e+89,
		  8.24765059208247e+90,
		  5.44344939077443e+92,
		  3.64711109181887e+94,
		  2.48003554243683e+96,
		  1.71122452428141e+98,
		  1.19785716699699e+100,
		  8.50478588567862e+101,
		  6.12344583768861e+103,
		  4.47011546151268e+105,
		  3.30788544151939e+107,
		  2.48091408113954e+109,
		  1.88549470166605e+111,
		  1.45183092028286e+113,
		  1.13242811782063e+115,
		  8.94618213078297e+116,
		  7.15694570462638e+118,
		  5.79712602074737e+120,
		  4.75364333701284e+122,
		  3.94552396972066e+124,
		  3.31424013456535e+126,
		  2.81710411438055e+128,
		  2.42270953836727e+130,
		  2.10775729837953e+132,
		  1.85482642257398e+134,
		  1.65079551609085e+136,
		  1.48571596448176e+138,
		  1.3520015276784e+140,
		  1.24384140546413e+142,
		  1.15677250708164e+144,
		  1.08736615665674e+146,
		  1.03299784882391e+148,
		  9.91677934870949e+149,
		  9.61927596824821e+151,
		  9.42689044888324e+153,
		  9.33262154439441e+155,
		  9.33262154439441e+157,
		  9.42594775983835e+159,
		  9.61446671503512e+161,
		  9.90290071648618e+163,
		  1.02990167451456e+166,
		  1.08139675824029e+168,
		  1.14628056373471e+170,
		  1.22652020319614e+172,
		  1.32464181945183e+174,
		  1.44385958320249e+176,
		  1.58824554152274e+178,
		  1.76295255109024e+180,
		  1.97450685722107e+182,
		  2.23119274865981e+184,
		  2.54355973347219e+186,
		  2.92509369349301e+188,
		  3.3931086844519e+190,
		  3.96993716080872e+192,
		  4.68452584975429e+194,
		  5.5745857612076e+196,
		  6.68950291344912e+198,
		  8.09429852527344e+200,
		  9.8750442008336e+202,
		  1.21463043670253e+205,
		  1.50614174151114e+207,
		  1.88267717688893e+209,
		  2.37217324288005e+211,
		  3.01266001845766e+213,
		  3.8562048236258e+215,
		  4.97450422247729e+217,
		  6.46685548922047e+219,
		  8.47158069087882e+221,
		  1.118248651196e+224,
		  1.48727070609069e+226,
		  1.99294274616152e+228,
		  2.69047270731805e+230,
		  3.65904288195255e+232,
		  5.01288874827499e+234,
		  6.91778647261949e+236,
		  9.61572319694109e+238,
		  1.34620124757175e+241,
		  1.89814375907617e+243,
		  2.69536413788816e+245,
		  3.85437071718007e+247,
		  5.5502938327393e+249,
		  8.04792605747199e+251,
		  1.17499720439091e+254,
		  1.72724589045464e+256,
		  2.55632391787286e+258,
		  3.80892263763057e+260,
		  5.71338395644585e+262,
		  8.62720977423323e+264,
		  1.31133588568345e+267,
		  2.00634390509568e+269,
		  3.08976961384735e+271,
		  4.78914290146339e+273,
		  7.47106292628289e+275,
		  1.17295687942641e+278,
		  1.85327186949373e+280,
		  2.94670227249504e+282,
		  4.71472363599206e+284,
		  7.59070505394721e+286,
		  1.22969421873945e+289,
		  2.0044015765453e+291,
		  3.28721858553429e+293,
		  5.42391066613159e+295,
		  9.00369170577843e+297,
		  1.503616514865e+300
};


void print_1di(int* arr,int m){
	int i;

	cout << "[";
	for(i=0;i<m;i++){
		cout << arr[i] << " ";
	}
	cout << "]" << endl;
}
void print_2d(string cstr,double* arr,int n){
	int i,j;

    cout << endl;
    cout << cstr << endl;
    for ( i=0;i<n;i++){
    	for ( j=0;j<n;j++){
          cout << arr[n*i+j] << " " ;
    	}
    	cout << endl;
    }
}
void print_2d(string cstr,complex<double> *arr,int n, int n2){
	int i,j;

	cout.precision(5);
	cout <<std::scientific;
    cout << endl;
    cout << cstr << endl;
    for ( i=0;i<n;i++){
    	for ( j=0;j<n2;j++){
          cout << arr[n2*i+j] << " " ;
          if(j%2==1) cout << endl;
    	}
    	cout << endl;
    }
}
void print_1c(complex<double> *arr,int m){
	int i;

	cout << "[";
	for(i=0;i<m;i++){
		cout << arr[i] << " ";
	}
	cout << "]" << endl;
}
void swap(double* a, double* b)
{
    double t = *a;
    *a = *b;
    *b = t;
}
int partition (double arr[], int low, int high, double arrv[],int n)
{
    double pivot = arr[high];    // pivot
    int i = (low - 1);  // Index of smaller element
    int vi;
    for (int j = low; j <= high- 1; j++)
    {
        // If current element is smaller than or
        // equal to pivot
        if (arr[j] >= pivot)
        {
            i++;    // increment index of smaller element
            swap(&arr[i], &arr[j]);
            for(vi=0;vi<n;vi++){
            	swap(&arrv[vi*n+i], &arrv[vi*n+j]);
            }
        }
    }
    swap(&arr[i + 1], &arr[high]);
    for(vi=0;vi<n;vi++){
    	swap(&arrv[vi*n+i+1], &arrv[vi*n+high]);
    }
    return (i + 1);
}
void quickSort(double arr[], int low, int high, double arrv[], int n)
{
    if (low < high)
    {
        /* pi is partitioning index, arr[p] is now
           at right place */
        int pi = partition(arr, low, high, arrv,n);

        // Separately sort elements before
        // partition and after partition
        quickSort(arr, low, pi - 1, arrv,n);
        quickSort(arr, pi + 1, high, arrv,n);
    }
}

extern "C" {
    extern int dgeev_(char*,char*,int*,double*,int*,double*, double*, double*, int*, double*, int*, double*, int*, int*);
    // LU decomoposition of a general matrix
    void dgetrf_(int* M, int *N, double* A, int* lda, int* IPIV, int* INFO);

    // generate inverse of a matrix given its LU decomposition
    void dgetri_(int* N, double* A, int* lda, int* IPIV, double* WORK, int* lwork, int* INFO);
    void dgemm_(char *transa, char *transb,
                int *m, int *n, int *k, double *alpha, double *a,
				int *lda, double *b, int *ldb,
                double *beta, double *c, int *ldc );
}


extern "C"
void stest(double* center_atoms){

    char Nchar='V';
    char charN='N';
    char charT='T';
    int i,j,k,l,totaln,n=3;
    double *outeig= new double[n];
    double *outeigvec= new double[n*n];

    double *sqrtD=new double[n*n];
    double *tempM=new double[n*n];

    int *IPIV = new int[n];
    double *eigReal=new double[n];
    double *eigImag=new double[n];

    double *data=new double[n*n];
    data[0]=1;data[1]=0.98107084;data[2]=0.93937439;
    data[3]=0.98107084;data[4]=1;data[5]=0.98742088;
    data[6]=0.93937439;data[7]=0.98742088;data[8]=1;

    int lwork=6*n;
    double *vl=new double[n*n];
    double *vr=new double[n*n];
    double *work=new double[lwork];
    int info;

    cout << "after inverse" << endl;
    dgetrf_(&n,&n,data,&n,IPIV,&info);
    dgetri_(&n,data,&n,IPIV,work,&lwork,&info);
    cout << endl;
    cout << "sinv" << endl;
    for (i=0;i<n;i++){
    	for ( j=0;j<n;j++){
          cout << data[n*i+j] << " " ;
    	}
    	cout << endl;
    }

    // calculate eigenvalues using the DGEEV subroutine
    dgeev_(&Nchar,&Nchar,&n,data,&n,outeig,eigImag,
          vl,&n,vr,&n,
          work,&lwork,&info);
    // check for errors
    if (info!=0){
      cout << "Error: dgeev returned error code " << info << endl;
      return ;
    }
/*
    cout << endl;
    cout << "vl" << endl;
    for(int i=0;i<n*n;i++){
    	cout << vl[i] << " ";
    }
    cout << endl;
    cout << "vr" << endl;
    for(int i=0;i<n*n;i++){
    	cout << vr[i] << " ";
    }
*/
    for( i=0;i<n;i++){
       for( j=0;j<n;j++){
    	   outeigvec[i*n+j]=vl[j*n+i];
       }
    }
    // output eigenvalues to stdout

    quickSort(outeig,0,n-1,outeigvec,n);
    cout << "--- Eigenvalues ---" << endl;
    for ( i=0;i<n;i++){
      cout << outeig[i] << " ";
    }
    print_2d("eigen vectors",outeigvec,n);

    for ( i=0;i<n;i++){
    	for ( j=0;j<n;j++){
          if(i==j) sqrtD[i*n+j]=sqrt(outeig[i]);
          else sqrtD[i*n+j]=0.0;
    	}
    }
    print_2d("sqrtD",sqrtD,n);

    totaln=n*n;
    for(i=0;i<totaln;i++) tempM[i]=0.0;
// ** to avoid row major issue, implemented direct matrix dot.
//    double dpone = 1.0, dmone = 0.0;
//    dgemm_ (&charN, &charT, &n, &n, &n, &dpone, outeigvec, &n, sqrtD, &n, &dmone,
//    tempM, &n);
    double dtemp;
    for(i=0;i<n;i++){
    	for(j=0;j<n;j++){
    		dtemp=0;
    		for(k=0;k<n;k++) dtemp+=outeigvec[i*n+k]*sqrtD[k*n+j];

	        tempM[i*n+j]=dtemp;
    	}
    }

    print_2d("V*sqrtD",tempM,n);

    dgetrf_(&n,&n,outeigvec,&n,IPIV,&info);
    dgetri_(&n,outeigvec,&n,IPIV,work,&lwork,&info);

    print_2d("inv V",outeigvec,n);
//    dgemm_ (&charN, &charN, &n, &n, &n, &dpone, tempM, &n, outeigvec, &n, &dmone,
//    data, &n);

    for(i=0;i<n;i++){
    	for(j=0;j<n;j++){
    		dtemp=0;
    		for(k=0;k<n;k++) dtemp+=tempM[i*n+k]*outeigvec[k*n+j];

	        data[i*n+j]=dtemp;
    	}
    }
    print_2d("out data",data,n);

}

void W(int nmax, double *arr){

	int alpha,beta,temp1,temp2;

	for(alpha=1;alpha<nmax+1;alpha++){
		temp1=(2*alpha+5)*(2*alpha+6)*(2*alpha+7);
		for(beta=1;beta<alpha+1;beta++){
            temp2 = (2*beta+5)*(2*beta+6)*(2*beta+7);
            arr[(alpha-1)*nmax+beta-1] = sqrt(temp1*temp2)/(5+alpha+beta)/(6+alpha+beta)/(7+alpha+beta);
            arr[(beta-1)*nmax+alpha-1] = arr[(alpha-1)*nmax+beta-1]	;
		}
	}
//	print_2d("arr",arr,nmax);

    char Nchar='V';
    char charN='N';
    char charT='T';
    int i,j,k,l,totaln,n=nmax;
    double *outeig= new double[n];
    double *outeigvec= new double[n*n];

    double *sqrtD=new double[n*n];
    double *tempM=new double[n*n];

    int *IPIV = new int[n];
    double *eigReal=new double[n];
    double *eigImag=new double[n];

    int lwork=6*n;
    double *vl=new double[n*n];
    double *vr=new double[n*n];
    double *work=new double[lwork];
    int info;

    //cout << "after inverse" << endl;
    dgetrf_(&n,&n,arr,&n,IPIV,&info);
    dgetri_(&n,arr,&n,IPIV,work,&lwork,&info);


    // calculate eigenvalues using the DGEEV subroutine
    dgeev_(&Nchar,&Nchar,&n,arr,&n,outeig,eigImag,
          vl,&n,vr,&n,
          work,&lwork,&info);
    // check for errors
    if (info!=0){
      cout << "Error: dgeev returned error code " << info << endl;
      return ;
    }

    for( i=0;i<n;i++){
       for( j=0;j<n;j++){
    	   outeigvec[i*n+j]=vl[j*n+i];
       }
    }
    // output eigenvalues to stdout

    quickSort(outeig,0,n-1,outeigvec,n);
//    cout << "--- Eigenvalues ---" << endl;
//    for ( i=0;i<n;i++){
//      cout << outeig[i] << " ";
//    }
//    print_2d("eigen vectors",outeigvec,n);

    for ( i=0;i<n;i++){
    	for ( j=0;j<n;j++){
          if(i==j) sqrtD[i*n+j]=sqrt(outeig[i]);
          else sqrtD[i*n+j]=0.0;
    	}
    }
//    print_2d("sqrtD",sqrtD,n);


// ** to avoid row major issue, implemented direct matrix dot.
//    double dpone = 1.0, dmone = 0.0;
//    dgemm_ (&charN, &charT, &n, &n, &n, &dpone, outeigvec, &n, sqrtD, &n, &dmone,
//    tempM, &n);
    double dtemp;
    for(i=0;i<n;i++){
    	for(j=0;j<n;j++){
    		dtemp=0;
    		for(k=0;k<n;k++) dtemp+=outeigvec[i*n+k]*sqrtD[k*n+j];

	        tempM[i*n+j]=dtemp;
    	}
    }

//    print_2d("V*sqrtD",tempM,n);

    dgetrf_(&n,&n,outeigvec,&n,IPIV,&info);
    dgetri_(&n,outeigvec,&n,IPIV,work,&lwork,&info);

//    print_2d("inv V",outeigvec,n);

    for(i=0;i<n;i++){
    	for(j=0;j<n;j++){
    		dtemp=0;
    		for(k=0;k<n;k++) dtemp+=tempM[i*n+k]*outeigvec[k*n+j];

	        arr[i*n+j]=dtemp;
    	}
    }
//    print_2d("out data",arr,n);

    delete outeig;
    delete outeigvec;

    delete sqrtD;
    delete tempM;

    delete IPIV;
    delete eigReal;
    delete eigImag;

    delete vl;
    delete vr;
    delete work;

}
void compute_pi(int nmax,int lmax, complex<double> *clisttot, int lcl1,int lcl2,
		complex<double> *plist, int lpl1,int lpl2,int indpl){

	//cout << "pi "<< M_PI << endl;
	int n1,n2,j,l,m,i=0;
	double norm;
	for(n1=0;n1<nmax;n1++){
		for(n2=0;n2<n1+1;n2++){
			j=0;
			for(l=0;l<lmax+1;l++){
				 norm = 2.0*sqrt(2.0)*M_PI/sqrt(2.0*l+1.0);
				 //cout<<"norm l "<<norm << " "<<l<<endl;

			     for(m=-l;m<l+1;m++){
				      //cout << "n1 n2 m "<<n1<<" "<<n2<<" "<<m<<endl;
				      //cout << "clist conj(clist) "<< clisttot[lcl2*n2+j] << " "<<conj(clisttot[lcl2*n2+j])<<endl;
				      plist[lpl2*indpl+i] += clisttot[lcl2*n1+j] * conj(clisttot[lcl2*n2+j])*norm;
			          j += 1;
			     }
			     i += 1;
			}
		}
	}
}
double phi(double r,int alpha,double rcut){

	return pow((rcut-r),(alpha+2))/sqrt(2*pow(rcut,(2*alpha+7))/(2*alpha+5)/(2*alpha+6)/(2*alpha+7));
}
complex<double> g(double r,int n,int nmax,double rcut,double *w,int lw1, int lw2){

	complex<double> Sum;
	Sum={0.0,0.0};
	int alpha;
	for(alpha=1;alpha<nmax+1;alpha++){
		Sum += w[(n-1)*lw1+alpha-1]*phi(r, alpha, rcut);
	}
	return Sum;
}
double modifiedSphericalBessel1(double r, int n, int derivative){

	double *temp_arr;
	double dval;
	int i;

	if(derivative==0){
		if(n==0){
			return sinh(r)/r;
		}else if(n==1){
			return (r*cosh(r)-sinh(r))/(r*r);
		}else{
			temp_arr=new double[n+1];
			for(i=0;i<n+1;i++) temp_arr[i]=0;
			temp_arr[0] = sinh(r)/r;
			temp_arr[1] = (r*cosh(r)-sinh(r))/(r*r);
			for(i=2;i<n+1;i++){
				temp_arr[i] = temp_arr[i-2] - (2*i-1)/r*temp_arr[i-1];
			}
			dval=temp_arr[n];
			delete temp_arr;
			return dval;
		}
	}else{
		if(n==0){
			return (r*cosh(r)-sinh(r))/(r*r);
		}else{
			temp_arr=new double[n+2];
			for(i=0;i<n+2;i++) temp_arr[i]=0;
			temp_arr[0] = sinh(r)/r;
			temp_arr[1] = (r*cosh(r)-sinh(r))/(r*r);
			for(i=2;i<n+2;i++){
				temp_arr[i] = temp_arr[i-2] - (2*i-1)/r*temp_arr[i-1];
			}
			dval=(n*temp_arr[n-1] + (n+1)*temp_arr[n+1]) / (2*n+1);
			delete temp_arr;
			return dval;
		}

	}
}

complex<double> integrand(double r, double ri, double alpha, double rcut,
		int n, int l, int nmax, double *w,int lw1,int lw2, int derivative){

    if (derivative == 0)
        return r*r*g(r, n, nmax, rcut, w,lw1,lw2)*exp(-alpha*r*r)*modifiedSphericalBessel1(2*alpha*r*ri, l, 0);
    else
        return r*r*r*g(r, n, nmax, rcut, w,lw1,lw2)*exp(-alpha*r*r)*modifiedSphericalBessel1(2*alpha*r*ri, l, 1);
}


complex<double>  get_radial_inner_product(double ri, double alpha, double rcut,
		int n, int l, int nmax, double *w,int lw1,int lw2,int derivative){
	double x,xi;
	complex<double> integral={0.0,0.0};
	int i,BigN;
	BigN=(n+l+1)*10;
	for(i=1;i<BigN+1;i++){
		x = cos((2*i-1)*M_PI/2/BigN);
		xi = rcut/2*(x+1);
		integral += sqrt(1-x*x)*integrand(xi, ri, alpha, rcut, n, l, nmax, w,lw1,lw2, derivative);
	}
	integral *= rcut/2*M_PI/BigN;
	return integral;
}
int _Wigner_index(int twoj, int twomp, int twom){
    return int(twoj*((2*twoj + 3) * twoj + 1) / 6) + int((twoj + twomp)/2) * (twoj + 1) + int((twoj + twom) /2);
}

double _Wigner_coefficient(int twoj, int twomp, int twom){
	//cout <<"_Wigner_coefficient twoj twomp twom "<< twoj <<" " << twomp << " " <<twom << endl;
	//cout <<"_coeff " << Wigner_data[_Wigner_index(twoj, twomp, twom)] << endl;
    return Wigner_data[_Wigner_index(twoj, twomp, twom)];
}

double Wigner_coefficient(int j, int mp, int m){
    return _Wigner_coefficient(int(round(2*j)), int(round(2*mp)), int(round(2*m)));

}


complex<double> Wigner_D(complex<double> vRa, complex<double> vRb, int twol, int twomp, int twom){

	double ra, phia, rb,phib;
	ra=abs(vRa);
	phia=arg(vRa);

	rb=abs(vRb);
	phib=arg(vRb);

	double epsilon = pow(10,-15);
	complex<double> czero={0.0,0.0};

    //cout <<"Wigner_D"<<endl;
    //cout <<vRa<<" "<<vRb<<" "<<twol<<" "<<twomp<<" "<<twom<<" "<<ra
    //		<<" "<<phia<<" "<<rb<<" "<<phib<<" "<<epsilon<<endl;

    if(ra <= epsilon){
        if(twomp != -twom || abs(twomp) > twol || abs(twom) > twol){
            return czero;
        }else{
            if((twol - twom) % 4 == 0){
                return pow(vRb,twom);
            }else{
                return pow(-vRb,twom);
            }
        }
    }else if(rb <= epsilon){
        if(twomp != twom || abs(twomp) > twol || abs(twom) > twol){
            return czero;
        }else{
            return pow(vRa,twom);
        }
    }else if(ra < rb){
    	double x = - ra*ra / rb / rb;
    	if (abs(twomp) > twol || abs(twom) > twol){
    	    return czero;
    	}else{
    		complex<double> Prefactor = polar(
    				 _Wigner_coefficient(twol, -twomp, twom)
    		                * pow(rb,(twol - (twom+twomp)/2))
    		                * pow(ra, ((twom+twomp)/2)),
    		                phib * (twom - twomp)/2 + phia * (twom + twomp)/2);
    		if(Prefactor==czero){
    			return czero;
    		}else{
    			int  l = twol/2;
    			int  mp = twomp/2;
    			int  m = twom/2;
    			int kmax = round(min(l-mp, l-m));
    			int kmin = round(max(0, -mp-m));
    			if ((twol - twom) %4 != 0){
    			     Prefactor *= -1;
    			}
    			double Sum = 1/fac_arr[int(round(kmax))]/fac_arr[int(round(l-m-kmax))]
				              /fac_arr[int(round(mp+m+kmax))]/fac_arr[int(round(l-mp-kmax))];
    			for(int k=kmax-1;k>kmin-1;k--){
    				Sum *=x;
    				Sum += 1/fac_arr[int(round(k))]/fac_arr[int(round(l-m-k))]
							/fac_arr[int(round(mp+m+k))]/fac_arr[int(round(l-mp-k))];
    			}
    			Sum *= pow(x,kmin);
    			return Prefactor * Sum;
    		}
    	}
    }else{
    	//cout <<"ra>rb"<<endl;
    	double x = - rb*rb / (ra * ra);
    	//cout <<"x"<<x<<endl;
    	//cout<<abs(twomp)<<" "<<twol<<" "<<abs(twom)<<" "<<twol<<endl;
    	if (abs(twomp) > twol || abs(twom) > twol){
    	    return czero;
    	}else{
    		complex<double> Prefactor = polar(
    				 _Wigner_coefficient(twol, twomp, twom)
    		                * pow(ra,(twol - twom/2 + twomp/2))
    		                * pow(rb, (twom/2 - twomp/2)),
							phia * (twom + twomp)/2 + phib * (twom - twomp)/2);
    		//cout<<"Prefactor "<<Prefactor<<endl;
    		if(Prefactor==czero){
    			return czero;
    		}else{
    			int  l = twol/2;
    			int  mp = twomp/2;
    			int  m = twom/2;
    			int kmax = round(min(l + mp, l - m));
    			int kmin = round(max(0, mp-m));
    			//cout<<l<<" "<<mp<<" "<<m<<" "<<kmax<<" "<<kmin<<endl;
    			double Sum = 1/fac_arr[int(round(kmax))]/fac_arr[int(round(l+mp-kmax))]
				              /fac_arr[int(round(l-m-kmax))]/fac_arr[int(round(-mp+m+kmax))];
    			//cout <<"Sum 1 "<<Sum << endl;
    			for(int k=kmax-1;k>kmin-1;k--){
    				Sum *=x;
    				//cout << "k Sum_1 "<< Sum << endl;
    				Sum += 1/fac_arr[int(round(k))]/fac_arr[int(round(l+mp-k))]
							/fac_arr[int(round(l-m-k))]/fac_arr[int(round(-mp+m+k))];
    				//cout << "k Sum_2 "<< Sum << endl;
    			}
    			//cout <<"Sum 2 "<<Sum<<endl;
    			Sum *= pow(x,kmin);
    			//cout <<"Sum"<<" "<<Sum<<endl;
    			return Prefactor * Sum;
    		}
    	}
    }

}

complex<double> sph_harm(complex<double> Ra, complex<double> Rb, int l, int m){
	//cout <<"sph_harm" << endl;
	//cout <<Ra<<" "<<Rb <<" "<<l << " "<<m<<endl;

	complex<double> ctemp;
//	ctemp=Wigner_D(Ra, Rb, 2*l, 0, -2*m);
//	cout << "ctemp "<<ctemp<<endl;
//	ctemp=conj(ctemp);
//	cout << "conj(ctemp) "<<ctemp<<endl;
	if(m%2==0){
		ctemp=Wigner_D(Ra, Rb, 2*l, 0, -2*m);
		//cout <<"Wigner_D return "<<ctemp<<endl;
		ctemp=conj(ctemp);
		//cout <<"mid "<<sqrt((2.0*l+1.0)/4.0/M_PI)<<endl;
		//cout <<"sph_harm return "<< ctemp * sqrt((2*l+1)/4/M_PI) * pow(-1.0,m)<<endl;
		return ctemp * sqrt((2.0*l+1.0)/4.0/M_PI) * pow(-1.0,m);
	}else{
		ctemp=Wigner_D(Ra, Rb, 2*l, 0, -2*m);
		ctemp=conj(ctemp);
		return ctemp * sqrt((2.0*l+1.0)/4.0/M_PI) * pow(-1.0,m+1);
	}
}
void compute_carray(double x, double y, double z, double ri, double alpha, double rcut,
		int nmax, int lmax, double *w, int lw1, int lw2,
		complex<double> *clist,int lcli1,int lcli2){

	double theta,phi,atheta,btheta,expfac;
	complex<double> aphi,Ra,Rb,r_int,Ylm;
	int n,i,l,m;
	theta=acos(z/ri);
	phi=atan2(y,x);

	atheta = cos(theta/2);
	btheta = sin(theta/2);

	aphi = {cos(phi/2), sin(phi/2)};
    Ra = atheta*aphi;
    Rb = btheta*aphi;

    expfac = 4*M_PI*exp(-alpha*ri*ri);

    //cout << "compute_carray" << endl;
    //cout <<"x y z "<<x<<" "<<y<<" "<<z<<endl;
    //cout << theta<<" "<<phi<<" "<<atheta<<" "<<btheta<<" "
    //		<<aphi<<" "<<Ra<<" "<<Rb<<" "<<expfac<<endl;
    for(n=1;n<nmax+1;n++){
    	i=0;
    	for(l=0;l<lmax+1;l++){
    		r_int = get_radial_inner_product(ri, alpha, rcut, n, l, nmax, w,lw1,lw2, 0);
    		//cout <<"r_int"<< " "<<r_int<<endl;
    		for(m=-l;m<l+1;m++){
    			 //cout <<Ra << " "<<Rb<<" "<<l<<" "<<m<<endl;
    			 Ylm = sph_harm(Ra, Rb, l, m);
    			 //cout << "Ylm" << " "<<Ylm<<endl;
    			 clist[(n-1)*lcli2+i] += r_int*Ylm*expfac;
//    			 if(n==1 && i==1) return;
    			 i+=1;
    		}
    	}
    }
}
int read_Wigner_data(){

        FILE * filp = fopen("Wigner_coefficients.dat", "rb");
        if(filp==NULL){
                cout << "file open error" << endl;
                fclose(filp);
                return 0;
        }
        int sizei=sizeof(int);
        int sized=sizeof(double);
        int total;
        int bytes_read = fread(&total, sizeof(int), 1, filp);
        if(bytes_read <=0 ){
                cout << "file read error" << endl;
                fclose(filp);
                return 0;
        }
        Wigner_data = new double[total];
        int i;
        for(i=0;i<total;i++){
          fread(&Wigner_data[i], sized, 1, filp);
//        cout << i << " " << Wigner_data[i] << endl;
        }
        fclose(filp);
//        cout << endl;
//      cout <<  wdata[_Wigner_index(2,1,0)] << endl;
 //       delete wdata;
        return 0;
}
extern "C"
void spectrum(int lcen1,int lcen2, double* center_atoms,
		int lne1,int lne2, int lne3, double* neighborlist,
		int lseq1,int lseq2, int* seq,
		int lans1,int lans2,int* neighbor_ANs,
		int nmax, int lmax,double rcut, double alpha,int derivative,int stress,
		int lpli1,int lpli2,double* plist_r, double* plist_i,
		int ldpl1,int ldpl2,int ldpl3,double* dplist_r, double* dplist_i,
		int lpst1,int lpst2,int lpst3,int lpst4,double* pstress_r,double* pstress_i){


	complex<double> *plist,*dplist,*pstress,*clisttot,*clist,*dclist;
	plist=new complex<double>[lpli1*lpli2];
	dplist=new complex<double>[ldpl1*ldpl2*ldpl3];
	pstress=new complex<double>[lpst1*lpst2*lpst3*lpst4];
	double *w;

	read_Wigner_data();

	int i,j,k,l,ti;
	int totali;
	totali=lpli1*lpli2;
    for(i=0;i<totali;i++){
         plist[i]={plist_r[i],plist_i[i]};
    }
	totali=ldpl1*ldpl2*ldpl3;
    for(i=0;i<totali;i++){
    	dplist[i]={dplist_r[i],dplist_i[i]};
    }
	totali=lpst1*lpst2*lpst3*lpst4;
    for(i=0;i<totali;i++){
    	pstress[i]={pstress_r[i],pstress_i[i]};
    }

//    print_1c(plist,lpli1*lpli2);
    int npairs=lne1;
    int nneighbors=lne2;
    int numYlms=(lmax+1)*(lmax+1);
    //cout << npairs << " " << nneighbors << " " << numYlms << endl;
    clisttot=new complex<double>[nmax*numYlms];
    clist=new complex<double>[nmax*numYlms];
    dclist=new complex<double>[nneighbors*nmax*numYlms*3];
    totali=nmax*numYlms;
    for(i=0;i<totali;i++){
    	clisttot[i]={0.0,0.0};
    	clist[i]={0.0,0.0};
    }
    totali=nneighbors*nmax*numYlms;
    for(i=0;i<totali;i++){
    	dclist[i]={0.0,0.0};
    }
    w=new double[nmax*nmax];
    totali=nmax*nmax;
    for(i=0;i<totali;i++) w[i]=0.0;

    W(nmax,w);
    //print_2d("w",w,nmax);
    //print_2d("o before clisttot",clisttot,nmax,numYlms);

    int numps,nstart,nsite,n,weight,neighbor;
    complex<double> *tempdp;
    double isite;
    double x,y,z,r;

    if(derivative==1){
		cout << "Error: not implemented" << endl;
		return;
    	if(stress==1){

    	}else{

    	}

    }else{
		isite=seq[0];
		nstart = 0;
		nsite = 0;
		for(n=0;n<npairs;n++){
		  //cout << lseq1 << " " << lseq2 << endl;
          i=seq[n*lseq2];
          j=seq[n*lseq2+1];
//          cout << i << " " << j << endl;
          if(i==j) nsite=n;
          weight=neighbor_ANs[n*lans2];
//          cout << weight << endl;
          if(i!=isite){
        	  compute_pi(nmax,lmax,clisttot,nmax,numYlms,plist,lpli1,lpli2,isite);
//        	  cout <<"len clisttot " << nmax << " " << numYlms << endl;
//        	  cout <<"len plist " << lpli1 << " " << lpli2 << endl;
//              print_2d("plist",plist,lpli1,lpli2);
              isite=i;
              nstart=n;
              totali=nmax*numYlms;
              for(ti=0;ti<totali;ti++) clisttot[ti]={0.0,0.0};

          }
          for(neighbor=0;neighbor<nneighbors;neighbor++){
//              x = neighborlist[(n*lne1+neighbor)*lne2+0];
//              y = neighborlist[(n*lne1+neighbor)*lne2+1];
//              z = neighborlist[(n*lne1+neighbor)*lne2+2];
        	  x = neighborlist[(n*lne2+neighbor)*lne3+0];
        	  y = neighborlist[(n*lne2+neighbor)*lne3+1];
        	  z = neighborlist[(n*lne2+neighbor)*lne3+2];

        	  //cout<<"neighborlist lne1 lne2 lne3 "<<lne1<<" "<<lne2<<" "<<lne3<< endl;
        	  //cout<<"n neighbor "<<n<<" "<<neighbor<< endl;
        	  //cout<<"x y z "<<x<<" "<<y<<" "<<z<<endl;

              r = sqrt(x*x + y*y + z*z);
              if(r<pow(10,-8)) continue;
              totali=nmax*numYlms;
              for(ti=0;ti<totali;ti++) clist[ti]={0.0,0.0};
              //print_2d("before clist",clist,nmax,numYlms);
              compute_carray(x, y, z, r, alpha, rcut, nmax, lmax, w,nmax,nmax,
                                                    clist,nmax,numYlms);
              //cout << x<< " " << y << " " << z << " " << r << " " << alpha << " "
              //     << " " << rcut << " "
            	//   << nmax <<" "<<lmax<<" "<<numYlms<< endl;
              //print_2d("clist",clist,nmax,numYlms);
//              return;
//              if(neighbor==1) return;

              totali=nmax*numYlms;
              for(int tn=0;tn<totali;tn++){
            	  clist[tn] = clist[tn]*double(weight);
              }

              for(int tn=0;tn<totali;tn++){
            	  clisttot[tn] += clist[tn];
              }
              //cout<<"weight "<<double(weight)<<endl;
              //print_2d("clisttot",clisttot,nmax,numYlms);
          }

		}
		//print_2d("before clisttot",clisttot,nmax,numYlms);
//		return;
		compute_pi(nmax,lmax,clisttot,nmax,numYlms,plist,lpli1,lpli2,isite);
		//print_2d("after plist",plist,lpli1,lpli2);

		totali=lpli1*lpli2;
	    for(i=0;i<totali;i++){
	         plist_r[i]=real(plist[i]);
	         plist_i[i]=imag(plist[i]);
	    }
		return;

    }
};

/*
int main(){
	double center_atom[10];
	complex<double> ctest[1];
	ctest[0]=complex<double>(1.2,3.4);
//	spectrum(8,center_atom,ctest);
	return 1;
}
*/

