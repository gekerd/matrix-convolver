#include "compute.h"

// Computes the convolution of two matrices
int convolve(matrix_t *a_matrix, matrix_t *b_matrix, matrix_t **output_matrix) {

  if (output_matrix == NULL) return -1;

  // Steps up size of output matirx
  int rows_bound = a_matrix->rows - b_matrix->rows + 1;
  int cols_bound = a_matrix->cols - b_matrix->cols + 1;
  *output_matrix = (matrix_t*)malloc(sizeof(matrix_t));
  if (*output_matrix == NULL) {
      return -1;
  }
  (*output_matrix)->rows = rows_bound;
  (*output_matrix)->cols = cols_bound;
  (*output_matrix)->data = (int*)malloc(rows_bound * cols_bound * sizeof(int));

  if ((*output_matrix)->data == NULL) {
      free(*output_matrix);
      return -1;
  }

  // Outer Loops to represent row and column of output matrix
  for (int i = 0; i < rows_bound; i++) {
      for (int j = 0; j < cols_bound; j++) {
          int sum = 0;
          int ai = 0;
          int aj = 0;
	  // Inner loops does the computations for the output matrix
          for (int bi = b_matrix->rows - 1; bi >= 0; bi--) {
              for (int bj = b_matrix->cols - 1; bj >= 0; bj--) {
                  sum += (a_matrix->data[(i+ai)*a_matrix->cols + j+aj] * b_matrix->data[bi*b_matrix->cols + bj]);
                  aj += 1;
              }
              ai += 1;
              aj = 0;
          }
          (*output_matrix)->data[i*cols_bound+j] = sum;
      }
  }

  return 0;
}

// Tests the convolve function
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
