import cffi

ffibuilder = cffi.FFI()
ffibuilder.cdef("""
    void kee_many(int n1, int n2, int d, int x2i, double zeta, double sigma, double sigma02, double* x1, int* ele1, int* x1_inds, double* x2, int* ele2, int* x2_inds, double* pout);
    void kef_many(int n1, int n2, int d, int x2i, double zeta, double* x1, int* ele1, int* x1_inds, double* x2, double* dx2dr, int* ele2, int* x2_inds, double* pout);
    void kff_many(int n1, int n2, int n2_start, int n2_end, int d, int x2i, double zeta, double* x1, double* dx1dr, int* ele1, int* x1_inds, double* x2, double* dx2dr, int* ele2, int* x2_inds, double* pout);
""")
ffibuilder.set_source("_dot_kernel", #lib name
    "", 
    sources=["dot_kernel.cpp"],
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
