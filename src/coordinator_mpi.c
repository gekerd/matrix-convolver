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

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) { 
        int num_tasks;
        task_t **tasks;
        if (read_tasks(argv[1], &num_tasks, &tasks) != 0) {
            fprintf(stderr, "Error reading task list from %s\n", argv[1]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        for (int i = 0; i < num_tasks; ++i) {
            int worker_rank = i % size;
            if (worker_rank == 0) { 
                if (execute_task(tasks[i]) != 0) {
                    fprintf(stderr, "Task %d failed\n", i);
                }
            } else {
                int path_length = strlen(tasks[i]->path) + 1;
                MPI_Send(&path_length, 1, MPI_INT, worker_rank, 0, MPI_COMM_WORLD);
                MPI_Send(tasks[i]->path, path_length, MPI_CHAR, worker_rank, 0, MPI_COMM_WORLD);
            }
        }

        for (int i = 1; i < size; ++i) {
            int terminate_signal = 0;
            MPI_Send(&terminate_signal, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        for (int i = 0; i < num_tasks; ++i) {
            free(tasks[i]->path);
            free(tasks[i]);
        }
        free(tasks);
    } else { 
        while (1) {
            int path_length;
            MPI_Recv(&path_length, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (path_length == 0) {
                break;
            }

            char *path = (char *)malloc(path_length * sizeof(char));
            MPI_Recv(path, path_length, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            task_t task = {.path = path};
            if (execute_task(&task) != 0) {
                fprintf(stderr, "Task execution failed on process %d\n", rank);
            }
            free(path);
        }
    }

    MPI_Finalize();
    return 0;
}
