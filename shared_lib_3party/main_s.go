package main

/*
#cgo LDFLAGS: -L. -L./ -lcalc
void cgbn_power(int size);
void cgbn_power_uma(int size);
void cgbn_power_zc(int size);
void powm_fixed_window(int size);
void tss_powm_fixed_window(int size);
void powm_sliding_window(int size);
void omp_gmp_power(int size);
void gmp_power(int size);
*/
import "C"

import (
	"fmt"
	"time"
	"os"
	"strconv"
	"sync"
	//"net/http"
        //glog "log"
        //_ "net/http/pprof"
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

func run_tss_powm_fixed_window(size int) {
	C.tss_powm_fixed_window(C.int(size))
}

func run_powm_sliding_window(size int) {
	C.powm_sliding_window(C.int(size))
}

func run_omp_gmp(size int) {
	C.omp_gmp_power(C.int(size))
}

func run_gmp(size int) {
	C.gmp_power(C.int(size))
}

func testFun(size int) {
	//run_tss_powm_fixed_window(size);
	//run_powm_fixed_window(size);
	//run_powm_sliding_window(size);
	//run_cuda(size);
	run_gmp(size);
}

func main() {
	size, _ := strconv.Atoi(os.Args[1])
        runTimes := 1000 * 3
        wg := sync.WaitGroup{}
        wg.Add(runTimes)

	//go func() {
        //glog.Println(http.ListenAndServe("localhost:6060", nil))
        //}()

	startTime := time.Now()
        go func() {
                for i := 0; i < 1000; i++ {
                        testFun(size)
                        wg.Done()
                }
        }()
        go func() {
                for i := 0; i < 1000; i++ {
                        testFun(size)
                        wg.Done()
                }
        }()
        go func() {
                for i := 0; i < 1000; i++ {
                        testFun(size)
                        wg.Done()
                }
        }()

        wg.Wait()
	fmt.Println(fmt.Sprintf("main_s cost %v", time.Now().Sub(startTime)))
}

