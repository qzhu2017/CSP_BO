#include <cmath>

extern "C"
void kff_many(int n1, int n2, int d, int x2i, double zeta,
            double* x1, double* dx1dr, int* ele1, int* x1_inds, 
            double* x2, double* dx2dr, int* ele2, int* x2_inds, double* pout){

    double eps=1e-8;
    double dval;
    double x1_norm, x2_norm, _x2_norm2, _x1x2_norm, x1x2_dot; 
    double _x1x2_norm31, x1x2_dot_norm02, x1x2_dot_norm13, x1x2_dot_norm31;
    double  dx, d2, d1, d2d_dx1dx2;
    double  C1, C2, C3, C4, C5, C6, C7, C8, C9;
    int i, j, ii, jj, _i, _j, _ele1, _ele2;
    double *d2k_dx1dx2;

    d2k_dx1dx2=new double[d*d];
    //dk_dD = sigma*sigma*zeta;

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
    	    	    d2 = d2*(zeta-1);
                    
                    C1 = C2 = C3 = C4 = C5 = C6 = C7 = C8 = C9 = 0;
    	    	    for(i=0;i<d;i++){
    	    	        for(j=0;j<d;j++){
    	    	            if(i==j){
                                dval=1.0;
                            } else {
    	    	    	        dval=0;
                            }

    	    	            d2d_dx1dx2 = (dval-x1[jj*d+i]*x1[jj*d+j]*_x2_norm2)*_x1x2_norm +
    	    	            (x1[ii*d+i]*x1[jj*d+j]*x1x2_dot_norm02-x1[ii*d+i]*x1[ii*d+j])*_x1x2_norm31;    

    	    	            d2k_dx1dx2[i*d+j]=((x1[jj*d+i]*_x1x2_norm-x1[ii*d+i]*x1x2_dot_norm31)*
    	    	        	(x1[ii*d+j]*_x1x2_norm-x1[jj*d+j]*x1x2_dot_norm13))*d2 + d1*d2d_dx1dx2;
    	    	    	}
    	    	    }

                    // switch the sequence of for loop can be helpful   
                    C1 = C2 = C3 = C4 = C5 = C6 = C7 = C8 = C9 = 0;
    	    	    for(i=0;i<d;i++){
    	    	    	for(j=0;j<d;j++){
                            dval = d2k_dx1dx2[i*d+j];
                            C1 += dx1dr[(ii*d+i)*3 + 0] * dval * dx2dr[(jj*d+j)*3 + 0];
                            C2 += dx1dr[(ii*d+i)*3 + 1] * dval * dx2dr[(jj*d+j)*3 + 0];
                            C3 += dx1dr[(ii*d+i)*3 + 2] * dval * dx2dr[(jj*d+j)*3 + 0];
                            C4 += dx1dr[(ii*d+i)*3 + 0] * dval * dx2dr[(jj*d+j)*3 + 1];
                            C5 += dx1dr[(ii*d+i)*3 + 1] * dval * dx2dr[(jj*d+j)*3 + 1];
                            C6 += dx1dr[(ii*d+i)*3 + 2] * dval * dx2dr[(jj*d+j)*3 + 1];
                            C7 += dx1dr[(ii*d+i)*3 + 0] * dval * dx2dr[(jj*d+j)*3 + 2];
                            C8 += dx1dr[(ii*d+i)*3 + 1] * dval * dx2dr[(jj*d+j)*3 + 2];
                            C9 += dx1dr[(ii*d+i)*3 + 2] * dval * dx2dr[(jj*d+j)*3 + 2];
    	    	            /*for(k=0;k<3;k++){
    	    	    	        for(l=0;l<3;l++){
    	    	                    pout[(k+_i*3)*x2i*3+l+_j*3] += dx1dr[(ii*d+i)*3+k] * dval * dx2dr[(jj*d+j)*3+l];
                                }
                            }*/
    	    	    	}
    	    	    }

    	    	    pout[(0+_i*3)*x2i*3+0+_j*3] += C1;
    	    	    pout[(1+_i*3)*x2i*3+0+_j*3] += C2;
    	    	    pout[(2+_i*3)*x2i*3+0+_j*3] += C3;
    	    	    pout[(0+_i*3)*x2i*3+1+_j*3] += C4;
    	    	    pout[(1+_i*3)*x2i*3+1+_j*3] += C5;
    	    	    pout[(2+_i*3)*x2i*3+1+_j*3] += C6;
    	    	    pout[(0+_i*3)*x2i*3+2+_j*3] += C7;
    	    	    pout[(1+_i*3)*x2i*3+2+_j*3] += C8;
    	    	    pout[(2+_i*3)*x2i*3+2+_j*3] += C9;

    	    	} //if(_ele1==_ele2 && x2_norm > eps)
    	    }//for(jj=0;jj<n;jj++)
    	    }//if(x1_norm > eps){
    }//for(ii=0;ii<n;ii++)

    delete d2k_dx1dx2;
    
};

