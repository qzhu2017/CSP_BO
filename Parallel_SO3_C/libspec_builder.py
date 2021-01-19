import cffi

ffibuilder = cffi.FFI()
ffibuilder.cdef("""
    void stest(double* center_atoms);
    void spectrum(int lcen1,int lcen2, double* center_atoms,int lne1,int lne2, int lne3, double* neighborlist,int lseq1,int lseq2, int* seq,int lans1,int lans2,int* neighbor_ANs,int nmax, int lmax,double rcut, double alpha,int derivative,int stress,int lpli1,int lpli2,double* plist_r, double* plist_i,int ldpl1,int ldpl2,int ldpl3,double* dplist_r, double* dplist_i,int lpst1,int lpst2,int lpst3,int lpst4,double* pstress_r,double* pstress_i);
""")
ffibuilder.set_source("_spectrum", #lib name
    "", 
    sources=["spectrum.cpp"],
    library_dirs=["./"],
    libraries=["lapack_LINUX","blas_LINUX","libf2c"],
    extra_compile_args=["-lstdc++"]   
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
