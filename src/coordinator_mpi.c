#include <mpi.h>

#include "coordinator.h"

#define READY 0
#define NEW_TASK 1
#define TERMINATE -1

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Error: not enough arguments\n");
    printf("Usage: %s [path_to_task_list]\n", argv[0]);
    return -1;
  }

  // TODO: implement Open MPI coordinator
  MPI_Init(&argc, &argv);
  int rank, size, num_tasks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int tasks_per_proc = num_tasks / size;
  int remainder = num_tasks % size;
  int start_task = rank * tasks_per_proc + (rank < remainder ? rank : remainder);
  int end_task = start_task + tasks_per_proc + (rank < remainder ? 1 : 0);

  for (int i = start_task; i < end_task; i++) {
      if (execute_task(i) != 0) fprintf(stderr, "Task %d failed on process %d\n", i, rank);
  }
  MPI_Finalize();
  return 0;
}
