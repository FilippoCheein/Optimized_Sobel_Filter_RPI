final0: final.cpp
	g++ -o final_O0 lab6.cpp -std=c++11 -I /usr/include/opencv -I /usr/local/include/perfmon -I /usr/local/include/perf -I /usr/local/lib/perf/include -L/usr/local/lib -lperf -Wall -O0 -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio -lpthread -lpfm -mfpu=neon -mcpu=Cortex-A8 -flax-vector-conversions -fpermissive

final1: final.cpp
	g++ -o final_O1 final.cpp -std=c++11 -I /usr/include/opencv -I /usr/local/include/perfmon -I /usr/local/include/perf -I /usr/local/lib/perf/include -L/usr/local/lib -lperf -Wall -Wextra -O -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio -lpthread -lpfm -mfpu=neon -mcpu=Cortex-A8 -flax-vector-conversions -fpermissive

final3: final.cpp
	g++ -o final_O3 final.cpp -std=c++11 -I /usr/include/opencv -I /usr/local/include/perfmon -I /usr/local/include/perf -I /usr/local/lib/perf/include -L/usr/local/lib -lperf -Wall -O3 -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio -lpthread -lpfm -mfpu=neon -mcpu=Cortex-A8 -flax-vector-conversions -fpermissive
