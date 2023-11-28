#include <omp.h>
#include <x86intrin.h>

#include "compute.h"

// Computes the convolution of two matrices
int convolve(matrix_t *a_matrix, matrix_t *b_matrix, matrix_t **output_matrix) {
  // TODO: convolve matrix a and matrix b, and store the resulting matrix in
  // output_matrix
  if (output_matrix == NULL) return -1;
  __m256i reverse_order = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);

  int rows_bound = a_matrix->rows - b_matrix->rows+1;
  int cols_bound = a_matrix->cols - b_matrix->cols+1;
  int32_t a_cols = a_matrix->cols;
  int32_t b_cols = b_matrix->cols;
  int32_t b_rows = b_matrix->rows;
  *output_matrix = (matrix_t*)malloc(sizeof(matrix_t));
  if (*output_matrix == NULL) return -1;
  (*output_matrix)->rows = rows_bound;
  (*output_matrix)->cols = cols_bound;
  (*output_matrix)->data = (int*)malloc(rows_bound*cols_bound*sizeof(int));
  if ((*output_matrix)->data == NULL) {
      free(*output_matrix);
      return -1;
  }
  int sum;
  int32_t* a_data = a_matrix->data;
  int32_t* b_data = b_matrix->data;
  #pragma omp parallel for collapse(2) reduction(+:sum)
  for (int i=0; i < rows_bound; i++) {
      for (int j = 0; j < cols_bound; j++) {
          sum = 0;
          __m256i sum_vec = _mm256_setzero_si256();
          int ai = i;
          int aj = j;
          for (int bi = b_rows -1; bi >= 0; bi--) {
              int bj;
              for (bj = b_cols -1; bj >= 31; bj -= 32) {
                  __m256i a_vec = _mm256_loadu_si256((const __m256i *)&(a_data[ai*a_cols + aj]));
                  __m256i b_vec = _mm256_loadu_si256((const __m256i *)&(b_data[bi*b_cols + bj - 7]));
                  b_vec = _mm256_permutevar8x32_epi32(b_vec, reverse_order);

                  sum_vec = _mm256_add_epi32(sum_vec, _mm256_mullo_epi32(a_vec, b_vec));
                  aj += 8;

                  a_vec = _mm256_loadu_si256((const __m256i *)&(a_data[ai*a_cols + aj]));
                  b_vec = _mm256_loadu_si256((const __m256i *)&(b_data[bi*b_cols + bj - 15]));
                  b_vec = _mm256_permutevar8x32_epi32(b_vec, reverse_order);
                  sum_vec = _mm256_add_epi32(sum_vec, _mm256_mullo_epi32(a_vec, b_vec));
                  aj += 8;

                  a_vec = _mm256_loadu_si256((const __m256i *)&(a_data[ai*a_cols + aj]));
                  b_vec = _mm256_loadu_si256((const __m256i *)&(b_data[bi*b_cols + bj - 23]));
                  b_vec = _mm256_permutevar8x32_epi32(b_vec, reverse_order);
                  sum_vec = _mm256_add_epi32(sum_vec, _mm256_mullo_epi32(a_vec, b_vec));
                  aj += 8;

                  a_vec = _mm256_loadu_si256((const __m256i *)&(a_data[ai*a_cols + aj]));
                  b_vec = _mm256_loadu_si256((const __m256i *)&(b_data[bi*b_cols + bj - 31]));
                  b_vec = _mm256_permutevar8x32_epi32(b_vec, reverse_order);
                  sum_vec = _mm256_add_epi32(sum_vec, _mm256_mullo_epi32(a_vec, b_vec));
                  aj += 8;
              }
              for (; bj >= 7; bj -= 8) {
                  __m256i a_vec = _mm256_loadu_si256((const __m256i *)&(a_data[ai*a_cols + aj]));
                  __m256i b_vec = _mm256_loadu_si256((const __m256i *)&(b_data[bi*b_cols + bj - 7]));
                  b_vec = _mm256_permutevar8x32_epi32(b_vec, reverse_order);
                  sum_vec = _mm256_add_epi32(sum_vec, _mm256_mullo_epi32(a_vec, b_vec));
                  aj += 8;
              }
              for (; bj >= 0; bj--) {
                  sum += (a_data[(ai)*a_cols + aj] * b_data[bi*b_cols +bj]);
                  aj += 1;
              }
              ai += 1;
              aj = j;
          }
          int temp_arr[8];
          _mm256_storeu_si256((__m256i *) temp_arr, sum_vec);
          sum += temp_arr[0] + temp_arr[1] + temp_arr[2] + temp_arr[3] + temp_arr[4] + 
              temp_arr[5] + temp_arr[6] + temp_arr[7];
          (*output_matrix)->data[i*cols_bound+j] = sum;
      }
  }

  return 0;
}

// Executes a task
int execute_task(task_t *task) {
  matrix_t *a_matrix, *b_matrix, *output_matrix;

  char *a_matrix_path = get_a_matrix_path(task);
  if (read_matrix(a_matrix_path, &a_matrix)) {
    printf("Error reading matrix from %s\n", a_matrix_path);
    return -1;
  }
  free(a_matrix_path);

  char *b_matrix_path = get_b_matrix_path(task);
  if (read_matrix(b_matrix_path, &b_matrix)) {
    printf("Error reading matrix from %s\n", b_matrix_path);
    return -1;
  }
  free(b_matrix_path);

  if (convolve(a_matrix, b_matrix, &output_matrix)) {
    printf("convolve returned a non-zero integer\n");
    return -1;
  }

  char *output_matrix_path = get_output_matrix_path(task);
  if (write_matrix(output_matrix_path, output_matrix)) {
    printf("Error writing matrix to %s\n", output_matrix_path);
    return -1;
  }
  free(output_matrix_path);

  free(a_matrix->data);
  free(b_matrix->data);
  free(output_matrix->data);
  free(a_matrix);
  free(b_matrix);
  free(output_matrix);
  return 0;
}
