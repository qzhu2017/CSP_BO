#include <cmath>

extern "C"
void kff_many(int n1, int n2, int d, int x2i, double sigma, double zeta,
            double* x1, double* dx1dr, int* ele1, int* x1_inds, 
            double* x2, double* dx2dr, int* ele2, int* x2_inds, double* pout){

    double eps=1e-8;
    double dval;
    double x1_norm, x2_norm, _x2_norm2, _x1x2_norm, x1x2_dot; 
    double _x1x2_norm31, x1x2_dot_norm02, x1x2_dot_norm13, x1x2_dot_norm31;
    double  dx,d2, d1, dk_dD, d2d_dx1dx2;
    int i,j,k,l,ii,jj,_i,_j,_ele1,_ele2;
    double *d2k_dx1dx2;

    d2k_dx1dx2=new double[d*d];
    dk_dD = sigma*sigma;

    for(ii=0; ii<n1; ii++){
    	_ele1 = ele1[ii];
    	_i = x1_inds[ii];
    
    	dval=0.0;
    	for(i=0;i<d;i++){
            dval+=x1[ii*d+i]*x1[ii*d+i];
    	}
    	x1_norm=sqrt(dval);
    
    	if(x1_norm > eps){
    
    	    for(jj=0; jj<n2; jj++){
    	    	_ele2 = ele2[jj];
    	    	_j = x2_inds[jj];
    
    	    	dval=0.0;
    	    	for(i=0;i<d;i++){
                    dval+=x1[jj*d+i]*x1[jj*d+i];
    	    	}
    	    	x2_norm=sqrt(dval);
    
    	    	if(_ele1==_ele2 && x2_norm > eps){
    	    	    _x2_norm2 = 1.0/(x2_norm*x2_norm);
                    _x1x2_norm = 1.0/(x1_norm*x2_norm);
                    _x1x2_norm31 = _x1x2_norm/(x1_norm*x1_norm);
   
    	    	    x1x2_dot=0;
    	    	    for(i=0;i<d;i++){
    	    		    x1x2_dot+=x1[ii*d+i]*x1[jj*d+i];
    	    		}
    
                    x1x2_dot_norm31 = x1x2_dot*_x1x2_norm31;
                    x1x2_dot_norm13 = x1x2_dot*_x1x2_norm*_x2_norm2;
                    x1x2_dot_norm02 = x1x2_dot*_x2_norm2;
 
                    dx = x1x2_dot*_x1x2_norm;
    	    	    d2 = pow(dx,zeta-2);
    	    	    d1 = dx*d2;
                    
    	    	    for(i=0;i<d;i++){
    	    	    	for(j=0;j<d;j++){
    	    	    	   dval=0;
    	    	           if(i==j) dval=1.0;
    	    	           d2d_dx1dx2=(dval-x1[jj*d+i]*x1[jj*d+j]*_x2_norm2)*_x1x2_norm+
    	    	        		   (x1[ii*d+i]*x1[jj*d+j]*x1x2_dot_norm02-x1[ii*d+i]*x1[ii*d+j])*_x1x2_norm31;    

    	    	           d2k_dx1dx2[i*d+j]=(((x1[jj*d+i]*_x1x2_norm-x1[ii*d+i]*x1x2_dot_norm31)*
    	    	        		               (x1[ii*d+j]*_x1x2_norm-x1[jj*d+j]*x1x2_dot_norm13))*d2*(zeta-1) +
    	    			                     d1*d2d_dx1dx2)*zeta*dk_dD;
    	    	    	}
    	    	    }
    
    	    	    for(k=0;k<3;k++){
    	    	    	for(l=0;l<3;l++){
    	    	            for(i=0;i<d;i++){
    	    	    	        for(j=0;j<d;j++){
    	    	                    pout[(k+_i*3)*x2i*3+l+_j*3] += dx1dr[(ii*d+i)*3+k] * d2k_dx1dx2[i*d+j] * dx2dr[(jj*d+j)*3+l];
                                }
                            }
    	    	    	}
    	    	    }
    	    	} //if(_ele1==_ele2 && x2_norm > eps)
    	    }//for(jj=0;jj<n;jj++)
    	    }//if(x1_norm > eps){
    }//for(ii=0;ii<n;ii++)

    delete d2k_dx1dx2;
    
};

