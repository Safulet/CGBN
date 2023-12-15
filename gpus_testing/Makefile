INC := -I/root/ck.w/zts-hpcc/libs/gmp/include
LIB := -L/root/ck.w/zts-hpcc/libs/gmp/lib

clean:
	rm -f libcalc.so
	rm -f main
	rm -f a.out

build_cu: 
	nvcc --ptxas-options=-v --compiler-options '-fPIC' $(INC) $(LIB) -I../include -arch=sm_75 -lgmp -Xcompiler -fopenmp calc.cu

build_cuda: 
	nvcc --ptxas-options=-v --compiler-options '-fPIC' $(INC) $(LIB) -I../include -arch=sm_75 -lgmp -Xcompiler -fopenmp -o libcalc.so --shared calc.cu

build_go:
	go build -o main main.go

