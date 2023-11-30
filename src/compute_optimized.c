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
  int32_t* out_data = (*output_matrix)->data;

  int x = b_cols / 8;

  __m256i* b_vecs = calloc(b_rows*x, sizeof(__m256i));
  if (b_vecs == NULL) {
      free((*output_matrix)->data);
      free(*output_matrix);
      return -1;
  }

  for (int i = 0; i < b_rows; i++) {
      for (int j = 0; j < x; j ++) {
          __m256i b_vec = _mm256_loadu_si256((const __m256i *)&(b_data[(b_rows-i-1)*b_cols + (b_cols-j*8-1) - 7]));
          b_vecs[i*x+j] = _mm256_permutevar8x32_epi32(b_vec, reverse_order);
      }
  }

  //#pragma omp parallel for collapse(2) reduction(+:sum) schedule(static, 7)
  for (int i=0; i < rows_bound; i++) {
      for (int j = 0; j < cols_bound; j++) {
          sum = 0;
          __m256i sum_vec = _mm256_setzero_si256();
          for (int bi = 0; bi < b_rows; bi++) {
              int bj;
              for (bj = 0; bj < b_cols / 32 * 32; bj += 32) {
                  __m256i a_vec = _mm256_loadu_si256((const __m256i *)&(a_data[(i+bi)*a_cols + (j+bj)]));
                  sum_vec = _mm256_add_epi32(sum_vec, _mm256_mullo_epi32(a_vec, b_vecs[bi*x+bj/8]));

                  a_vec = _mm256_loadu_si256((const __m256i *)&(a_data[(i+bi)*a_cols + (j+bj)+8]));
                  sum_vec = _mm256_add_epi32(sum_vec, _mm256_mullo_epi32(a_vec, b_vecs[bi*x+bj/8+1]));

                  a_vec = _mm256_loadu_si256((const __m256i *)&(a_data[(i+bi)*a_cols + (j+bj)+16]));
                  sum_vec = _mm256_add_epi32(sum_vec, _mm256_mullo_epi32(a_vec, b_vecs[bi*x+bj/8+2]));

                  a_vec = _mm256_loadu_si256((const __m256i *)&(a_data[(i+bi)*a_cols + (j+bj)+24]));
                  sum_vec = _mm256_add_epi32(sum_vec, _mm256_mullo_epi32(a_vec, b_vecs[bi*x+bj/8+3]));
              }
              for (; bj < x*8; bj += 8) {
                  __m256i a_vec = _mm256_loadu_si256((const __m256i *)&(a_data[(i+bi)*a_cols + j+bj]));
                  sum_vec = _mm256_add_epi32(sum_vec, _mm256_mullo_epi32(a_vec, b_vecs[bi*x+bj/8]));
              }
              for (; bj < b_cols; bj++) {
                  sum += (a_data[(i+bi)*a_cols + j + bj] * b_data[(b_rows-bi-1)*b_cols+b_cols-bj-1]);
              }
          }
          int temp_arr[8];
          _mm256_storeu_si256((__m256i *) temp_arr, sum_vec);
          sum += temp_arr[0] + temp_arr[1] + temp_arr[2] + temp_arr[3] + temp_arr[4] + 
              temp_arr[5] + temp_arr[6] + temp_arr[7];
          out_data[i*cols_bound+j] = sum;
      }
  }
  free(b_vecs);

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
