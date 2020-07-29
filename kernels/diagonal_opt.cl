// Authors: Tobias Kenter, Arjun Ramaswami

typedef struct {
   float2 i0;
   float2 i1;
   float2 i2;
   float2 i3;
   float2 i4;
   float2 i5;
   float2 i6;
   float2 i7;
} float2x8;

float2x8 readBuf(float2 bufA[DEPTH][POINTS], unsigned step){
  const unsigned N = (1 << LOGN);
  unsigned base = (step & (N / POINTS - 1)) << LOGN; // 0, N, 2N, ...
  unsigned offset = (step >> LOGN) & ((N / 8) - 1);  // 0, .. N / POINTS
  float2 rotate_out[POINTS];
  float2x8 data;

  #pragma unroll POINTS
  for(unsigned i = 0; i < POINTS; i++){
    unsigned rot = ((POINTS + i - (step >> (LOGN - LOGPOINTS))) << (LOGN - LOGPOINTS)) & (N - 1);
    unsigned row_rotate = base + offset + rot;
    rotate_out[i] = bufA[row_rotate][i];
  }

  unsigned rot_out = (step >> (LOGN - LOGPOINTS)) & (POINTS - 1);
  data.i0 = rotate_out[(0 + rot_out) & (POINTS - 1)];
  data.i1 = rotate_out[(1 + rot_out) & (POINTS - 1)];
  data.i2 = rotate_out[(2 + rot_out) & (POINTS - 1)];
  data.i3 = rotate_out[(3 + rot_out) & (POINTS - 1)];
  data.i4 = rotate_out[(4 + rot_out) & (POINTS - 1)];
  data.i5 = rotate_out[(5 + rot_out) & (POINTS - 1)];
  data.i6 = rotate_out[(6 + rot_out) & (POINTS - 1)];
  data.i7 = rotate_out[(7 + rot_out) & (POINTS - 1)];

  return data;
}

void writeBuf(float2 data[POINTS], float2 bufA[DEPTH][POINTS], unsigned step){
  const unsigned N = (1 << LOGN);

  unsigned row = step & (DEPTH - 1);
  unsigned rot = (row >> (LOGN - LOGPOINTS)) & (POINTS - 1);

  #pragma unroll POINTS
  for(unsigned i = 0; i < POINTS; i++){
    bufA[row][i] = data[((i + POINTS) - rot) & (POINTS -1)];
  }
}