package main

/*
#cgo LDFLAGS: -L. -L./ -lcalc
void cgbn_power(int size);
void cgbn_power_uma(int size);
void cgbn_power_zc(int size);
void powm_fixed_window(int size);
void powm_sliding_window(int size);
void powm_sliding_window_uma(int size);
void powm_sliding_window_zc(int size);
void omp_gmp_power(int size);
void gmp_power(int size);
*/
import "C"

import (
	"fmt"
	"time"
	"os"
	"strconv"
)

func run_cuda(size int) {
	C.cgbn_power(C.int(size))
}

func run_cuda_uma(size int) {
	C.cgbn_power_uma(C.int(size))
}

func run_cuda_zc(size int) {
	C.cgbn_power_zc(C.int(size))
}

func run_powm_fixed_window(size int) {
	C.powm_fixed_window(C.int(size))
}

func run_powm_sliding_window(size int) {
	C.powm_sliding_window(C.int(size))
}

func run_powm_sliding_window_uma(size int) {
	C.powm_sliding_window_uma(C.int(size))
}

func run_powm_sliding_window_zc(size int) {
	C.powm_sliding_window_zc(C.int(size))
}

func run_omp_gmp(size int) {
	C.omp_gmp_power(C.int(size))
}

func run_gmp(size int) {
	C.gmp_power(C.int(size))
}

func main() {
	size, _ := strconv.Atoi(os.Args[1])

	//size := 5
	num := 3
/*
	for i := 0; i < num; i++ {
		startTime := time.Now()
		run_cuda(size)
		fmt.Println(fmt.Sprintf("run_cuda cost %v", time.Now().Sub(startTime)))
	}

	fmt.Println(fmt.Sprintf("========================="))
	for i := 0; i < num; i++ {
		startTime := time.Now()
		run_cuda_uma(size)
		fmt.Println(fmt.Sprintf("run_cuda_uma cost %v", time.Now().Sub(startTime)))
	}

	fmt.Println(fmt.Sprintf("========================="))
	for i := 0; i < num; i++ {
		startTime := time.Now()
		run_cuda_zc(size)
		fmt.Println(fmt.Sprintf("run_cuda_zc cost %v", time.Now().Sub(startTime)))
	}

*/

/*
	fmt.Println(fmt.Sprintf("========================="))
	for i := 0; i < num; i++ {
		startTime := time.Now()
		run_powm_fixed_window(size)
		fmt.Println(fmt.Sprintf("run_powm_fixed_window cost %v", time.Now().Sub(startTime)))
	}

	fmt.Println(fmt.Sprintf("========================="))
	for i := 0; i < num; i++ {
		startTime := time.Now()
		run_powm_sliding_window(size)
		fmt.Println(fmt.Sprintf("run_powm_sliding_window cost %v", time.Now().Sub(startTime)))
	}

	fmt.Println(fmt.Sprintf("========================="))
	for i := 0; i < num; i++ {
		startTime := time.Now()
		run_powm_sliding_window_uma(size)
		fmt.Println(fmt.Sprintf("run_powm_sliding_window_uma cost %v", time.Now().Sub(startTime)))
	}
*/

	fmt.Println(fmt.Sprintf("========================="))
	for i := 0; i < num; i++ {
		startTime := time.Now()
		run_powm_sliding_window_zc(size)
		fmt.Println(fmt.Sprintf("run_powm_sliding_window_zc cost %v", time.Now().Sub(startTime)))
	}
	
/*
*/
	/*
	fmt.Println(fmt.Sprintf("========================="))
	for i := 0; i < num; i++ {
		startTime := time.Now()
		run_gmp(size)
		fmt.Println(fmt.Sprintf("run_gmp cost %v", time.Now().Sub(startTime)))
	}

	fmt.Println(fmt.Sprintf("========================="))
	for i := 0; i < num; i++ {
		startTime := time.Now()
		run_omp_gmp(size)
		fmt.Println(fmt.Sprintf("run_omp_gmp cost %v", time.Now().Sub(startTime)))
	}
	*/

}
