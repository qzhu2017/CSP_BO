import cffi

ffibuilder = cffi.FFI()
ffibuilder.cdef("""
    void kff_many(int n1, int n2, int d, int x2i, double zeta, double* x1, double* dx1dr, int* ele1, int* x1_inds, double* x2, double* dx2dr, int* ele2, int* x2_inds, double* pout);
""")
ffibuilder.set_source("_kff", #lib name
    "",
    sources=["kff_v3_openblas.cpp"],
    include_dirs=["../../../OpenBLAS"],
    library_dirs=["../../../OpenBLAS"],
    libraries=["openblas"]
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
