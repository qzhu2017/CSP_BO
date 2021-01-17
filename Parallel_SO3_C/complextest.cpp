#include <cmath>
#include <iostream>
#include <complex.h>
#include <tgmath.h>
#include <string.h>

using namespace std;
int main(){


	complex<double> vRa;
	vRa={2,3};
	double ra, phia;
	ra=abs(vRa);
	phia=arg(vRa);
	cout <<ra<<" "<<phia<< endl;
}
