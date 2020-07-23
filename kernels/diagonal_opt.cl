void transpose_step(float2 *data, float2 bufA[][POINTS], float2 bufB[][POINTS], unsigned step, unsigned LOGN) {

  const unsigned N = (1 << LOGN);
  const unsigned DEPTH = (1 << (LOGN + LOGN - LOGPOINTS));

  unsigned row = step & (DEPTH - 1);
  unsigned rot = (row >> (LOGN - LOGPOINTS)) & (POINTS - 1);

  #pragma unroll POINTS
  for(unsigned i = 0; i < POINTS; i++){
      bufA[row][i] = data[((i + POINTS) - rot) & (POINTS -1)];
  }

  unsigned base = (step & (N / POINTS - 1)) << LOGN; // 0, N, 2N, ...
  unsigned offset = (step >> LOGN) & ((N / 8) - 1);  // 0, .. N / POINTS
  float2 rotate_out[POINTS];

  #pragma unroll POINTS
  for(unsigned i = 0; i < POINTS; i++){
    unsigned rot = ((POINTS + i - (step >> (LOGN - LOGPOINTS))) << (LOGN - LOGPOINTS)) & (N - 1);
    unsigned row_rotate = base + offset + rot;
    rotate_out[i] = bufB[row_rotate][i];
  }

  unsigned rot_out = (step >> (LOGN - LOGPOINTS)) & (POINTS - 1);
  data[0] = rotate_out[(0 + rot_out) & (POINTS - 1)];
  data[1] = rotate_out[(1 + rot_out) & (POINTS - 1)];
  data[2] = rotate_out[(2 + rot_out) & (POINTS - 1)];
  data[3] = rotate_out[(3 + rot_out) & (POINTS - 1)];
  data[4] = rotate_out[(4 + rot_out) & (POINTS - 1)];
  data[5] = rotate_out[(5 + rot_out) & (POINTS - 1)];
  data[6] = rotate_out[(6 + rot_out) & (POINTS - 1)];
  data[7] = rotate_out[(7 + rot_out) & (POINTS - 1)];
}
