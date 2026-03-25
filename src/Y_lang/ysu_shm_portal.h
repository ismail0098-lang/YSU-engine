#ifndef YSU_SHM_PORTAL_H
#define YSU_SHM_PORTAL_H

#include <stdint.h>

// The core data structure for YSU audio/data streaming
typedef struct {
    float *buffer;
    int head;
    int tail;
    int size;
} RingBuffer;

// Function prototype for the producer
void ysu_produce_safe(RingBuffer *rb, float sample);

#endif
