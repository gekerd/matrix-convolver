#include <mpi.h>

#include "coordinator.h"

#define READY 0
#define NEW_TASK 1
#define TERMINATE -1

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    task_t *task;
    int num_tasks;

    if (world_rank == 0) {
        if (argc < 2) {
            fprintf(stderr, "Error: not enough arguments\n");
            fprintf(stderr, "Usage: %s [path_to_task_list]\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return -1;
        }

        task_t **tasks = NULL;
        if (!read_tasks(argv[1], &num_tasks, &tasks)) {
            fprintf(stderr, "Error reading task list from %s\n", argv[1]);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return -1;
        }

        for (int i = 0; i < num_tasks; i++) {
            int worker_rank = (i % (world_size - 1)) + 1; // Simple round-robin
            MPI_Send(tasks[i], sizeof(task_t), MPI_BYTE, worker_rank, 0, MPI_COMM_WORLD);
        }

        for (int i = 0; i < num_tasks; i++) {
            int worker_rank = (i % (world_size - 1)) + 1;
            int task_result;
            MPI_Recv(&task_result, 1, MPI_INT, worker_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (task_result != 0) {
                fprintf(stderr, "Task %d failed\n", i);
            }
        }

        for (int i = 0; i < num_tasks; i++) {
            free(tasks[i]->path);
            free(tasks[i]);
        }
        free(tasks);

    } else {
        while (1) {
            task = (task_t *)malloc(sizeof(task_t));
            MPI_Recv(task, sizeof(task_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int task_result = execute_task(task);
            MPI_Send(&task_result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            
            free(task->path);
            free(task);
        }
    }

    MPI_Finalize();
    return 0;
}
