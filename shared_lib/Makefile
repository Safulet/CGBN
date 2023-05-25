ifdef GMP_HOME
  INC := -I$(GMP_HOME)/include
  LIB := -L$(GMP_HOME)/lib
endif
ifndef GMP_HOME
  INC :=
  LIB :=
endif

clean:
	rm -f libcalc.so

build: 
	nvcc --ptxas-options=-v --compiler-options '-fPIC' $(INC) $(LIB) -I../include -arch=sm_75 -lgmp -Xcompiler -fopenmp -o libcalc.so --shared calc.cu

