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

    int rank, size, num_tasks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    task_t **tasks = NULL;

    if (rank == 0) {
        if (read_tasks(argv[1], &num_tasks, &tasks) != 0) {
            fprintf(stderr, "Error reading task list from %s\n", argv[1]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    MPI_Bcast(&num_tasks, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 1; i < num_tasks; ++i) {
            MPI_Send(tasks[i]->path, strlen(tasks[i]->path) + 1, MPI_CHAR, i % (size - 1) + 1, 0, MPI_COMM_WORLD);
        }
    }

    if (rank != 0) {
        char *path = malloc(256 * sizeof(char));
        MPI_Recv(path, 256, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        task_t task = { .path = path };

        if (execute_task(&task) != 0) {
            fprintf(stderr, "Task execution failed on process %d\n", rank);
        }

        free(path);
    }

    if (rank == 0) {
        for (int i = 0; i < num_tasks; ++i) {
            free(tasks[i]->path);
            free(tasks[i]);
        }
        free(tasks);
    }

    MPI_Finalize();
    return 0;
}
