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

float2x8 bitreverse_out(float2 bitrev_outA[N], float2 bitrev_outB[N], float2x8 data, unsigned row){
  float2 rotate_in[POINTS];

  rotate_in[0] = data.i0;
  rotate_in[1] = data.i1;
  rotate_in[2] = data.i2;
  rotate_in[3] = data.i3;
  rotate_in[4] = data.i4;
  rotate_in[5] = data.i5;
  rotate_in[6] = data.i6;
  rotate_in[7] = data.i7;

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
  float2x8 rotate_out;
  rotate_out.i0 = bitrev_outB[index_out]; 
  rotate_out.i1 = bitrev_outB[(4 * N / 8) + index_out];
  rotate_out.i2 = bitrev_outB[(2 * N / 8) + index_out];
  rotate_out.i3 = bitrev_outB[(6 * N / 8) + index_out];
  rotate_out.i4 = bitrev_outB[(N / 8) + index_out];
  rotate_out.i5 = bitrev_outB[(5 * N / 8) + index_out];
  rotate_out.i6 = bitrev_outB[(3 * N / 8) + index_out];
  rotate_out.i7 = bitrev_outB[(7 * N / 8) + index_out];

  return rotate_out;
}

float2x8 readBuf(float2 buf[DEPTH][POINTS], unsigned step){
  const unsigned DELAY = (1 << (LOGN - LOGPOINTS)); // N / 8

  unsigned rows = (step + DELAY);
  unsigned base = (rows & (N / POINTS - 1)) << LOGN; // 0, N, 2N, ...
  unsigned offset = (rows >> LOGN) & ((N / 8) - 1);  // 0, .. N / POINTS

  float2 rotate_out[POINTS];
  float2x8 data;

  #pragma unroll POINTS
  for(unsigned i = 0; i < POINTS; i++){
    unsigned rot = ((POINTS + i - (rows >> (LOGN - LOGPOINTS))) << (LOGN - LOGPOINTS)) & (N - 1);
    unsigned row_rotate = (base + offset + rot);
    rotate_out[i] = buf[row_rotate][i];
  }

  data.i0 = rotate_out[0];
  data.i1 = rotate_out[1];
  data.i2 = rotate_out[2];
  data.i3 = rotate_out[3];
  data.i4 = rotate_out[4];
  data.i5 = rotate_out[5];
  data.i6 = rotate_out[6];
  data.i7 = rotate_out[7];

  return data;
}

int bit_reversed(int x, int bits) {
  int y = 0;
  #pragma unroll 
  for (int i = 0; i < bits; i++) {
    y <<= 1;
    y |= x & 1;
    x >>= 1;
  }
  return y;
}

float2x8 bitreverse_in(float2x8 rotate_in, float2 bitrev_inA[N], float2 bitrev_inB[N], unsigned row){

  const unsigned STEPS = (N / 8);
  int index = row & (STEPS - 1); // [0, N/8 - 1]
  int index_in = index * 8;

  bitrev_inA[index_in + 0] = rotate_in.i0; // 0
  bitrev_inA[index_in + 1] = rotate_in.i1; // 32
  bitrev_inA[index_in + 2] = rotate_in.i2; // 16
  bitrev_inA[index_in + 3] = rotate_in.i3; // 48
  bitrev_inA[index_in + 4] = rotate_in.i4; // 8
  bitrev_inA[index_in + 5] = rotate_in.i5; // 40
  bitrev_inA[index_in + 6] = rotate_in.i6; // 24
  bitrev_inA[index_in + 7] = rotate_in.i7; // 5

  float2x8 rotate_out;
  int index_out = index * 8;
  int index0 = bit_reversed(index_out + 0, LOGN);
  int index1 = bit_reversed(index_out + 1, LOGN);
  int index2 = bit_reversed(index_out + 2, LOGN);
  int index3 = bit_reversed(index_out + 3, LOGN);
  int index4 = bit_reversed(index_out + 4, LOGN);
  int index5 = bit_reversed(index_out + 5, LOGN);
  int index6 = bit_reversed(index_out + 6, LOGN);
  int index7 = bit_reversed(index_out + 7, LOGN);

  rotate_out.i0 = bitrev_inB[index0];
  rotate_out.i1 = bitrev_inB[index1];
  rotate_out.i2 = bitrev_inB[index2];
  rotate_out.i3 = bitrev_inB[index3];
  rotate_out.i4 = bitrev_inB[index4];
  rotate_out.i5 = bitrev_inB[index5];
  rotate_out.i6 = bitrev_inB[index6];
  rotate_out.i7 = bitrev_inB[index7];

  return rotate_out;
}

void writeBuf(float2x8 data, float2 buf[DEPTH][POINTS], int step){

  const unsigned DELAY = (1 << (LOGN - LOGPOINTS)); // N / 8
  float2 rot_bitrev_in[POINTS];

  rot_bitrev_in[0] = data.i0;
  rot_bitrev_in[1] = data.i1;
  rot_bitrev_in[2] = data.i2;
  rot_bitrev_in[3] = data.i3;
  rot_bitrev_in[4] = data.i4;
  rot_bitrev_in[5] = data.i5;
  rot_bitrev_in[6] = data.i6;
  rot_bitrev_in[7] = data.i7;

  unsigned rot = ((step + DELAY) >> (LOGN - LOGPOINTS)) & (POINTS - 1);
  unsigned row_in = (step + DELAY) & (DEPTH - 1); 

  #pragma unroll POINTS
  for(unsigned i = 0; i < POINTS; i++){
    buf[row_in][i] = rot_bitrev_in[((i + POINTS) - rot) & (POINTS -1)];
  }
}