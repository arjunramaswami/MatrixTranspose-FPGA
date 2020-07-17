
void bitreverse(float2 * bitrevA, float2 * bitrevB, float2 * rotate_in, float2 * rotate_out, unsigned N, unsigned row){

  const unsigned STEPS = N / 8;
  unsigned index = row & (STEPS - 1); // [0, N/8 - 1]

  bitrevA[index] = rotate_in[0]; // 0
  bitrevA[(4 * (N / 8)) + index] = rotate_in[1]; // 32
  bitrevA[(2 * (N / 8)) + index] = rotate_in[2]; // 16
  bitrevA[(6 * (N / 8)) + index] = rotate_in[3]; // 48
  bitrevA[(N / 8) + index] = rotate_in[4]; // 8
  bitrevA[(5 * (N / 8)) + index] = rotate_in[5]; // 40
  bitrevA[(3 * (N / 8)) + index] = rotate_in[6]; // 24
  bitrevA[(7 * (N / 8)) + index] = rotate_in[7]; // 5

  unsigned index_out = index * 8;

  rotate_out[0] = bitrevB[index_out + 0];
  rotate_out[1] = bitrevB[index_out + 1];
  rotate_out[2] = bitrevB[index_out + 2];
  rotate_out[3] = bitrevB[index_out + 3];
  rotate_out[4] = bitrevB[index_out + 4];
  rotate_out[5] = bitrevB[index_out + 5];
  rotate_out[6] = bitrevB[index_out + 6];
  rotate_out[7] = bitrevB[index_out + 7];
}

void transpose_step(float2 *data, float2 bufA[][POINTS], float2 bufB[][POINTS], float2 * bitrev_inA, float2 * bitrev_inB, float2 * bitrev_outA, float2 * bitrev_outB, int step, unsigned row, unsigned LOGN) {

  const unsigned N = (1 << LOGN);
  const unsigned DEPTH = (1 << (LOGN + LOGN - LOGPOINTS));
  const unsigned DELAY = (1 << (LOGN - LOGPOINTS)); // N / 8

  float2 rot_bitrev_in[POINTS], rot_bitrev_out[POINTS];
  float2 rotate_out[POINTS];

  //printf("Input bitreversal \n");
  bitreverse(bitrev_inA, bitrev_inB, data, rot_bitrev_in, N, row);

  unsigned rot = (step - DELAY) >> (LOGN - LOGPOINTS) & (POINTS - 1);
  //unsigned rot = row >> (LOGN - LOGPOINTS) & (POINTS - 1);
  // (row -1) to take into account first N elements being garbage
  unsigned row_in = (step - DELAY) & (DEPTH - 1);

  //printf("Feed to buffer\n");
  #pragma unroll 8
  for(unsigned i = 0; i < POINTS; i++){
      bufA[row_in][i] = rot_bitrev_in[((i + POINTS) - rot) & (POINTS -1)];
  }

  /*
  printf("Buf A\n");
  for(unsigned i = 0; i < DEPTH; i++){
    for(unsigned j = 0; j < POINTS; j++){
      printf("(%f, %f) ", bufA[i][j].x, bufA[i][j].y);
    }
    printf("\n");
  }
  printf("\n\n");
  */
  /*
  printf("Buf B\n");
  for(unsigned i = 0; i < DEPTH; i++){
    for(unsigned j = 0; j < POINTS; j++){
      printf("(%f, %f) ", bufA[i][j].x, bufA[i][j].y);
    }
    printf("\n");
  }
  printf("\n\n");
  */

  unsigned base = ((step - DELAY) & (N / POINTS - 1)) << LOGN; // 0, N, 2N, ...
  // unsigned base = (row & (N / POINTS - 1)) << LOGN; // 0, N, 2N, ...
  // unsigned offset = row >> LOGN;                    // 0, .. N / POINTS
  unsigned offset = (step - DELAY) >> LOGN & ((N / 8) - 1);  // 0, .. N / POINTS

  //printf("Drain from buffer \n");
  // store data into temp buffer
  #pragma unroll 8
  for(unsigned i = 0; i < POINTS; i++){
    unsigned rot = ((POINTS + i - ((step - DELAY) >> (LOGN - LOGPOINTS))) << (LOGN - LOGPOINTS)) & (N - 1);
    //unsigned rot = ((POINTS + i - (row >> (LOGN - LOGPOINTS))) << (LOGN - LOGPOINTS)) & (N - 1);
    unsigned row_rotate  = base + offset + rot;
    rotate_out[i] = bufB[row_rotate][i];
  }

  /*
  if(step >= DEPTH + DELAY){
    printf("%d: ", step);
    for(unsigned j = 0; j < POINTS; j++){
      printf("(%f, %f) ", rotate_out[j].x, rotate_out[j].y);
    }
    printf("\n\n");
  }
  */

  //printf("Output bitreversal \n");
  bitreverse(bitrev_outA, bitrev_outB, rotate_out, rot_bitrev_out, N, row);

  if(step >= DEPTH + DELAY){
    printf("%d: ", step);
    for(unsigned j = 0; j < N; j++){
      printf("(%f, %f) ", bitrev_outA[j].x, bitrev_outA[j].y);
    }
    printf("\n\n");
  }
  /*
  if(step >= DEPTH + DELAY){
    printf("%d: ", step);
    for(unsigned j = 0; j < POINTS; j++){
      printf("(%f, %f) ", rot_bitrev_out[j].x, rot_bitrev_out[j].y);
    }
    printf("\n\n");
  }
  */
  unsigned rot_out = (step - DELAY) >> (LOGN - LOGPOINTS) & (POINTS - 1);
  //unsigned rot_out = row >> (LOGN - LOGPOINTS) & (POINTS - 1);

  #pragma unroll 8
  for(unsigned i = 0; i < POINTS; i++){
    data[i] = rot_bitrev_out[(i + rot_out) & (POINTS - 1)];
  }

}
