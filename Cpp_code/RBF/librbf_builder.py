import cffi

builder = cffi.FFI()
builder.cdef("""
        void kee_many(int n1, int n2, int d, int x2i, double zeta, double sigma2, double l2, double* x1, int* ele1, int* x1_inds, double* x2, int* ele2, int* x2_inds, double* pout);
        void kef_many(int n1, int n2, int d, int x2i, double zeta, double sigma2, double l2, double* x1, int* ele1, int* x1_inds, double* x2, double* dx2dr, int* ele2, int* x2_inds, double* pout);
        void kef_many_stress(int n1, int n2, int d, int x2i, double zeta, double sigma2, double l2, double* x1, int* ele1, int* x1_inds, double* x2, double* dx2dr, int* ele2, int* x2_inds, double* pout);
        void kff_many(int n1, int n2, int n2_start, int n2_end, int d, int x2i, double zeta, double sigma2, double l2, double* x1, double* dx1dr, int* ele1, int* x1_inds, double* x2, double* dx2dr, int* ele2, int* x2_inds, double* pout);
        void kff_many_stress(int n1, int n2, int n2_start, int n2_end, int d, int x2i, double zeta, double sigma2, double l2, double* x1, double* dx1dr, int* ele1, int* x1_inds, double* x2, double* dx2dr, int* ele2, int* x2_inds, double* pout);
        """)

builder.set_source("_rbf_kernel", "",
                   sources=["rbf_kernel.cpp"],)

if __name__ == "__main__":
    builder.compile(verbose=True)
