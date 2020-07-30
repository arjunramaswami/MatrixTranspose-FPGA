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

void bitreverse_out(float2 bitrev_outA[N], float2 bitrev_outB[N], float2 rotate_in[POINTS], float2 rotate_out[POINTS], unsigned row){
  const unsigned STEPS = (1 << (LOGN - LOGPOINTS));

  unsigned index = (row & (STEPS - 1)) * 8;
  unsigned rot = (row >> (LOGN - LOGPOINTS)) & (POINTS - 1);

  bitrev_outA[index] = rotate_in[(0 + rot) & (POINTS - 1)];
  bitrev_outA[index + 1] = rotate_in[(1 + rot) & (POINTS - 1)];
  bitrev_outA[index + 2] = rotate_in[(2 + rot) & (POINTS - 1)];
  bitrev_outA[index + 3] = rotate_in[(3 + rot) & (POINTS - 1)];
  bitrev_outA[index + 4] = rotate_in[(4 + rot) & (POINTS - 1)];
  bitrev_outA[index + 5] = rotate_in[(5 + rot) & (POINTS - 1)];
  bitrev_outA[index + 6] = rotate_in[(6 + rot) & (POINTS - 1)];
  bitrev_outA[index + 7] = rotate_in[(7 + rot) & (POINTS - 1)];

  unsigned index_out = (row & (STEPS - 1));
  rotate_out[0] = bitrev_outB[index_out]; 
  rotate_out[1] = bitrev_outB[(4 * N / 8) + index_out];
  rotate_out[2] = bitrev_outB[(2 * N / 8) + index_out];
  rotate_out[3] = bitrev_outB[(6 * N / 8) + index_out];
  rotate_out[4] = bitrev_outB[(N / 8) + index_out];
  rotate_out[5] = bitrev_outB[(5 * N / 8) + index_out];
  rotate_out[6] = bitrev_outB[(3 * N / 8) + index_out];
  rotate_out[7] = bitrev_outB[(7 * N / 8) + index_out];
}

float2x8 readBuf(float2 buf[DEPTH][POINTS], float2 bitrev_outA[N], float2 bitrev_outB[N], unsigned step){
  const unsigned DELAY = (1 << (LOGN - LOGPOINTS)); // N / 8

  unsigned rows = (step + DELAY);
  unsigned base = (rows & (N / POINTS - 1)) << LOGN; // 0, N, 2N, ...
  unsigned offset = (rows >> LOGN) & ((N / 8) - 1);  // 0, .. N / POINTS

  float2 rotate_out[POINTS], rot_bitrev_out[POINTS];
  float2x8 data;

  #pragma unroll POINTS
  for(unsigned i = 0; i < POINTS; i++){
    unsigned rot = ((POINTS + i - (rows >> (LOGN - LOGPOINTS))) << (LOGN - LOGPOINTS)) & (N - 1);
    unsigned row_rotate = (base + offset + rot);
    rotate_out[i] = buf[row_rotate][i];
  }

  unsigned start_row = (step + DELAY) & (DEPTH -1);
  bitreverse_out(bitrev_outA, bitrev_outB, rotate_out, rot_bitrev_out, start_row);

  data.i0 = rot_bitrev_out[0];
  data.i1 = rot_bitrev_out[1];
  data.i2 = rot_bitrev_out[2];
  data.i3 = rot_bitrev_out[3];
  data.i4 = rot_bitrev_out[4];
  data.i5 = rot_bitrev_out[5];
  data.i6 = rot_bitrev_out[6];
  data.i7 = rot_bitrev_out[7];

  return data;
}

void bitreverse_in(float2 bitrev_inA[N], float2 bitrev_inB[N], float2x8 rotate_in, float2 rotate_out[POINTS], unsigned row){

  const unsigned STEPS = (N / 8);
  int index = row & (STEPS - 1); // [0, N/8 - 1]

  bitrev_inA[index] = rotate_in.i0; // 0
  bitrev_inA[(4 * (N / 8)) + index] = rotate_in.i1; // 32
  bitrev_inA[(2 * (N / 8)) + index] = rotate_in.i2; // 16
  bitrev_inA[(6 * (N / 8)) + index] = rotate_in.i3; // 48
  bitrev_inA[(N / 8) + index] = rotate_in.i4; // 8
  bitrev_inA[(5 * (N / 8)) + index] = rotate_in.i5; // 40
  bitrev_inA[(3 * (N / 8)) + index] = rotate_in.i6; // 24
  bitrev_inA[(7 * (N / 8)) + index] = rotate_in.i7; // 5

  int index_out = index * 8;

  rotate_out[0] = bitrev_inB[index_out + 0];
  rotate_out[1] = bitrev_inB[index_out + 1];
  rotate_out[2] = bitrev_inB[index_out + 2];
  rotate_out[3] = bitrev_inB[index_out + 3];
  rotate_out[4] = bitrev_inB[index_out + 4];
  rotate_out[5] = bitrev_inB[index_out + 5];
  rotate_out[6] = bitrev_inB[index_out + 6];
  rotate_out[7] = bitrev_inB[index_out + 7];
}

void writeBuf(float2x8 data, float2 buf[DEPTH][POINTS], float2 bitrev_inA[N], float2 bitrev_inB[N], int step){

  const unsigned DELAY = (1 << (LOGN - LOGPOINTS)); // N / 8
  unsigned row = step & (DEPTH - 1);
  float2 rot_bitrev_in[POINTS];

  bitreverse_in(bitrev_inA, bitrev_inB, data, rot_bitrev_in, row);

  unsigned rot = ((step + DELAY) >> (LOGN - LOGPOINTS)) & (POINTS - 1);
  unsigned row_in = (step + DELAY) & (DEPTH - 1); 

  #pragma unroll POINTS
  for(unsigned i = 0; i < POINTS; i++){
    buf[row_in][i] = rot_bitrev_in[((i + POINTS) - rot) & (POINTS -1)];
  }
}