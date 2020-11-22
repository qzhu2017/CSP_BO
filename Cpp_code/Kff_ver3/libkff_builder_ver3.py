import cffi

ffibuilder = cffi.FFI()
ffibuilder.cdef("""
    void kff_many(int n, int d, int x2i, double sigma, double* x1, double* dx1dr, int* ele1, int* x1_indices, double* pout);
""")
ffibuilder.set_source("_kff", #lib name
    "", 
    #sources=["test.c"],
#    sources=["kff.c"],
    sources=["kff_ver3.cpp"],
    #source_extension=".cpp",
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)