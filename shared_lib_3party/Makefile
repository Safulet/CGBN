ifdef GMP_HOME
  INC := -I/root/ck.w/gmp-6.2.1/include
  LIB := -L/root/ck.w/gmp-6.2.1/lib
endif
ifndef GMP_HOME
  INC :=
  LIB :=
endif

clean:
	rm -f libcalc.so
	rm -f main
	rm -f main_s

build_cuda: 
	nvcc --ptxas-options=-v --compiler-options '-fPIC' $(INC) $(LIB) -I../include -arch=sm_70 -lgmp -Xcompiler -fopenmp -o libcalc.so --shared calc.cu

build_go:
	go build -o main main.go

build_go_s:
	go build -o main_s main_s.go
