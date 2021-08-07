/*******************
* Final Project
*
* Filippo Cheein and Lewis Pietropaoli
*
* November 24, 2020
*
* CPE 442-01
* 
* Prof. Danowitz
*
* Final Optimizations:
* 1. Convert 3-channel Frames to 1-channel
* 2. Attempt to vectorize Matrix math in sobel_helper()
* 3. Look into -O3 compile flag
* 
*******************/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <pthread.h>
#include <arm_neon.h>
#include <pfmlib.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "libperf.h"
#include <atomic>

using namespace cv;

/* FUNCTION PROTOTYPES */
Mat to442_grayscale(Mat frame, Mat frame_C1);
void grayscale_helper(Mat *frame, Mat *frame_C1, uint16_t i);

Mat to442_sobel(Mat frame);
void sobel_helper(Mat *frame, uint16_t i);

void *fnForThread(void *threadArgs);

/* END FUNCTION PROTOTYPES */

/* THREADING GLOBALS */
void *threadStatus0;
void *threadStatus1;
void *threadStatus2;
void *threadStatus3;

pthread_t thread[4];

struct threadArgs {
    Mat frame; //3 channel frame
    Mat simple_frame; //1 channel frame
    std::atomic_bool complete;
    std::atomic_bool flag;
};
/* THREADING GLOBALS END */

/* MAIN GLOBALS */
Mat frame_main;
Mat frame_done;

uint64_t total_cache_misses = 0;
uint64_t total_l1d_cache_misses = 0;
uint64_t total_cpu_cycles = 0;
uint64_t total_time = 0;
uint64_t total_num_frames = 0;
/* MAIN GLOBALS END */

int main(int argc, char **argv){
   pthread_attr_t attr;
   pthread_attr_init (&attr);
   pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_JOINABLE);
 
   /* init counter */
   struct libperf_data* pd = libperf_initialize(-1,-1); /* init lib */

   /* define args for each thread */
   struct threadArgs threadArgs0 = { .frame = frame_main, 
	                             .complete = {false},
                                     .flag = {false}};
   struct threadArgs threadArgs1 = { .frame = frame_main, 
	                             .complete = {false},
                                     .flag = {false}};
   struct threadArgs threadArgs2 = { .frame = frame_main, 
	                             .complete = {false},
                                     .flag = {false}};
   struct threadArgs threadArgs3 = { .frame = frame_main, 
	                             .complete = {false},
                                     .flag = {false}};
    
   /* open video file */
   VideoCapture vid(argv[1]);

   /* check if able to open */
   if(!vid.isOpened()){
      printf("Error opening %s\n", argv[1]);
      return -1;
   }
   printf("Successfully opened: %s\n", argv[1]);

   /* spawn threads */
   pthread_create(&thread[0], NULL, fnForThread, (void *)&threadArgs0);
   pthread_create(&thread[1], NULL, fnForThread, (void *)&threadArgs1);
   pthread_create(&thread[2], NULL, fnForThread, (void *)&threadArgs2);
   pthread_create(&thread[3], NULL, fnForThread, (void *)&threadArgs3);
   
   /* enable counters */
   libperf_enablecounter(pd, LIBPERF_COUNT_HW_CPU_CYCLES);
   libperf_enablecounter(pd, LIBPERF_COUNT_HW_CACHE_MISSES);
   libperf_enablecounter(pd, LIBPERF_COUNT_HW_CACHE_L1D_LOADS_MISSES);
   libperf_enablecounter(pd, LIBPERF_COUNT_SW_CPU_CLOCK);  

   /* frame dimensions */
   uint16_t rows;
   uint16_t cols;
   uint16_t qrows;
   uint16_t qcols;
   Mat frame1, frame2;

   /* main thread routine: iterate thru frames */
   while(1){
      /* get frame */
      vid >> frame_main;

      /* check if empty */
      if(frame_main.empty()){
         threadArgs0.complete = {true};
         threadArgs1.complete = {true};
         threadArgs2.complete = {true};
         threadArgs3.complete = {true};
	 threadArgs0.flag = {true};
         threadArgs1.flag = {true};
         threadArgs2.flag = {true};
         threadArgs3.flag = {true};
         break;
      }
      
      /* split frame into quadrants */
      rows = frame_main.rows;
      cols = frame_main.cols;
      qrows = rows/2 + 1;
      qcols = cols/2 + 1;

      /* init simple_frame */
      threadArgs0.simple_frame = Mat(rows, cols, CV_8UC1);
      threadArgs1.simple_frame = Mat(rows, cols, CV_8UC1);
      threadArgs2.simple_frame = Mat(rows, cols, CV_8UC1);
      threadArgs3.simple_frame = Mat(rows, cols, CV_8UC1);

      /* assign quadrants to each image thread */
      frame_main(Rect( 0,        0,        qcols, qrows)).copyTo(threadArgs0.frame);
      frame_main(Rect( cols/2-1, 0,        qcols, qrows)).copyTo(threadArgs1.frame);
      frame_main(Rect( 0,        rows/2-1, qcols, qrows)).copyTo(threadArgs2.frame);
      frame_main(Rect( cols/2-1, rows/2-1, qcols, qrows)).copyTo(threadArgs3.frame);

      /* set flags true */
      threadArgs0.flag = {true};
      threadArgs1.flag = {true};
      threadArgs2.flag = {true};
      threadArgs3.flag = {true};

      /* wait for all 4 threads to finish processing before proceeding */
      while (threadArgs0.flag);
      while (threadArgs1.flag);
      while (threadArgs2.flag);
      while (threadArgs3.flag);

      /* reassemble frame */
      hconcat(threadArgs0.simple_frame(Rect(0, 0, qcols-2, qrows-2)), 
              threadArgs1.simple_frame(Rect(0, 0, qcols-2, qrows-2)),
              frame1),
      hconcat(threadArgs2.simple_frame(Rect(0, 0, qcols-2, qrows-2)), 
              threadArgs3.simple_frame(Rect(0, 0, qcols-2, qrows-2)), 
              frame2); 
      vconcat(frame1, frame2, frame_done);

      /* display frame */
      imshow("Frame", frame_done);
      waitKey(12); 

      total_num_frames++;
   }

   /* disable and finalize counter */
   libperf_disablecounter(pd, LIBPERF_COUNT_HW_CPU_CYCLES);
   libperf_disablecounter(pd, LIBPERF_COUNT_HW_CACHE_MISSES);
   libperf_disablecounter(pd, LIBPERF_COUNT_SW_CPU_CLOCK);
   libperf_disablecounter(pd, LIBPERF_COUNT_HW_CACHE_L1D_LOADS_MISSES);

   /* join all threads */
   pthread_join(thread[0], &threadStatus0);
   pthread_join(thread[1], &threadStatus1);
   pthread_join(thread[2], &threadStatus2);
   pthread_join(thread[3], &threadStatus3);
    
   /* retrieve final stats */
   total_cpu_cycles = libperf_readcounter(pd, LIBPERF_COUNT_HW_CPU_CYCLES);
   total_cache_misses = libperf_readcounter(pd, LIBPERF_COUNT_HW_CACHE_MISSES);
   total_time = libperf_readcounter(pd, LIBPERF_COUNT_SW_CPU_CLOCK);
   total_l1d_cache_misses = libperf_readcounter(pd, LIBPERF_COUNT_HW_CACHE_L1D_LOADS_MISSES);
   libperf_finalize(pd, 0);

   /* print final stats */
   printf("TOTAL FRAMES %" PRIu64"\n", total_num_frames);
   printf("TOTAL TIME [ns]: %.2f\n", float(total_time >> 2));
   printf("AVG TIME PER FRAME [ns/frame]: %.2f\n", float(total_time >> 2)/total_num_frames);
   printf("AVG CACHE MISSES PER FRAME PER CORE: %.2f (TOTAL: %" PRIu64")\n",
             float(total_cache_misses >> 2)/total_num_frames,
	     total_cache_misses); 
   printf("AVG L1D CACHE MISSES PER FRAME PER CORE: %.2f (TOTAL: %" PRIu64")\n",
             float(total_l1d_cache_misses >> 2)/total_num_frames,
	     total_l1d_cache_misses); 

   printf("AVG CPU CYCLES PER FRAME PER CORE: %.2f (TOTAL: %" PRIu64")\n",
	     float(total_cpu_cycles >> 2)/total_num_frames,
	     total_cpu_cycles);

   /* close window */
   vid.release();
   destroyAllWindows();

   return 0;
}

/* image thread routine */
void *fnForThread(void *threadArgs){
   struct threadArgs *args = (struct threadArgs *)threadArgs; 
  
   while(1) {
      /* wait for new frame */
      while (args->flag == false);

      /* return when complete flag set */
      if(args->complete)
         return 0;

      /* process frame and assign to global struct */
      args->simple_frame = to442_sobel(to442_grayscale(args->frame, args->simple_frame));
      
      /* set flag to false */    
      args->flag = {false};
   }

   return 0;
}

Mat to442_grayscale(Mat frame, Mat frame_C1){
   uint16_t i;
   uint16_t cols = frame.cols;
    
   /* iterate thru each col */
   for (i = 0; i < cols; i++){   
      grayscale_helper(&frame, &frame_C1, i);
   }

   /* return simple frame */
   return frame_C1;
}

void grayscale_helper(Mat *frame, Mat *frame_C1, uint16_t i){
   uint16_t j, k;
   uint16_t rows = frame->rows;
   uint8_t r, g, b, gray;

   uint8x8_t v_r = vdup_n_u8(0);
   uint8x8_t v_g = vdup_n_u8(0);
   uint8x8_t v_b = vdup_n_u8(0);

   uint8x8_t w_r = vdup_n_u8(77);
   uint8x8_t w_g = vdup_n_u8(150);
   uint8x8_t w_b = vdup_n_u8(29);

   uint8x16_t temp_r, temp_g, temp_b;

   uint8x8_t v_gray;

   /* iterate thru each row
    * vectorizable loop */
   for (j = 0; j < (rows & ~0xf); j += 8){
      /* vectorize each color separately */
        for (k = 0; k < 8; k++){
         /* get color values at pixel */	   
         v_b[k] = frame->at<cv::Vec3b>(j+k,i)[0]; //blue
         v_g[k] = frame->at<cv::Vec3b>(j+k,i)[1]; //green
         v_r[k] = frame->at<cv::Vec3b>(j+k,i)[2]; //red
      }

      /* apply weight to each color */
      temp_b = vmull_u8(v_b, w_b);
      temp_g = vmull_u8(v_g, w_g);
      temp_r = vmull_u8(v_r, w_r);

      /* shift to narrow back to 8-bits */
      v_b = vshrn_n_u16(temp_b, 8);
      v_g = vshrn_n_u16(temp_g, 8);
      v_r = vshrn_n_u16(temp_r, 8);

      /* sum weighted color vectors */
      v_gray = vadd_u8(v_b, v_g);
      v_gray = vadd_u8(v_r, v_gray);

      /* assign back to Mat */
      for (k = 0; k < 8; k++){
	 frame_C1->at<uint8_t>(j+k,i) = v_gray[k];     
      }      
   }
   
   /* cleanup loop */
   for (j = (rows & ~0xf); j < rows; j++){
      /* get color values at pixel */	   
      b = frame->at<cv::Vec3b>(j,i)[0]; //blue
      g = frame->at<cv::Vec3b>(j,i)[1]; //green
      r = frame->at<cv::Vec3b>(j,i)[2]; //red

      /* calc gray using eq */
      gray = (77*r + 150*g + 29*b) >> 8;

      /* set colors to gray */
      frame_C1->at<uint8_t>(j,i) = gray;
   }
}

Mat to442_sobel(Mat frame){ //1 channel frame
   uint16_t i;
   uint16_t cols = frame.cols - 1; //ignoring borders
   
   /* iterate thru each col */
   for (i = 1; i < cols; i++){
      sobel_helper(&frame, i);
   }

   return frame;
}

void sobel_helper(Mat *frame, uint16_t i){
   uint16_t j, k;
   uint16_t rows = frame->rows - 1;
   uint16_t gx, gy, gtot;

   uint16x8_t v_gx, v_gy;
   uint8x8_t v_gx8, v_gy8;
   uint8x8_t v_gtot;

   /* iterate thru rows */
   /* vectorizable loop */
   for(j = 1; j < (rows & ~0xf)-7; j+=8){ 
      for(k = 0; k < 8; k++){

      /* VECTORIZABLE? */	      
      /*compute gy*/
      v_gy[k] = (-2*frame->at<uint8_t>(j+k,i+1) - frame->at<uint8_t>(j+1+k,i+1) -
                 frame->at<uint8_t>(j-1+k,i+1) + 2*frame->at<uint8_t>(j+k,i-1) +
                 frame->at<uint8_t>(j+1+k,i-1) + frame->at<uint8_t>(j-1+k,i-1) );    

      /* compute gx */
      v_gx[k] = (2*frame->at<uint8_t>(j+1+k,i) + frame->at<uint8_t>(j+1+k,i+1) +
                 frame->at<uint8_t>(j+1+k,i-1) - 2*frame->at<uint8_t>(j-1+k,i) -
                 frame->at<uint8_t>(j-1+k,i+1) - frame->at<uint8_t>(j-1+k,i-1) );
      }
     
      /* absolute value of vector gx and vector gy  */ 
      v_gy = vabsq_s16(v_gy);
      v_gx = vabsq_s16(v_gx);

      /* convert and saturate 16bit values into 8bit */
      v_gy8 = vqmovn_u16(v_gy);
      v_gx8 = vqmovn_u16(v_gx);

      /* get total gradient with saturation if above 255*/
      v_gtot = vqadd_u8(v_gx8, v_gy8);

      
      /* store back the values  */
      for(k = 0; k < 8; k++){
	 frame->at<uint8_t>(j-1+k,i-1) = v_gtot[k];
      }
   }

   /* cleanup loop */
   for(j = (rows & ~0xf)-7; j < rows; j++){

      /* compute gy */
      gy = abs(-2*frame->at<uint8_t>(j,i+1) - frame->at<uint8_t>(j+1,i+1) -
                 frame->at<uint8_t>(j-1,i+1) + 2*frame->at<uint8_t>(j,i-1) +
                 frame->at<uint8_t>(j+1,i-1) + frame->at<uint8_t>(j-1,i-1) );    

      /* compute gx */
      gx = abs(2*frame->at<uint8_t>(j+1,i) + frame->at<uint8_t>(j+1,i+1) +
                 frame->at<uint8_t>(j+1,i-1) - 2*frame->at<uint8_t>(j-1,i) -
                 frame->at<uint8_t>(j-1,i+1) - frame->at<uint8_t>(j-1,i-1) );
      
      /* get total gradient */
      gtot = gx + gy;

      /* if gtot > 255, assign white */
      if (gtot > 255) {
         frame->at<uint8_t>(j-1,i-1) = 255;

      /* else assign gtot */
      } else {
         frame->at<uint8_t>(j-1,i-1) = gtot;
      }
   }
}
