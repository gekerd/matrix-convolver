from framework import Test, TestSpec, Task, Matrix
import filters


@Test(seed=100)
def test_example(test: TestSpec):
    test.add_task(Task(Matrix.random(100, 99), Matrix.random(10, 9)))
    test.add_task(Task(Matrix.random(2, 10, min_value=1, max_value=10), 
                       Matrix.random(1, 9, min_value=1, max_value=10)))
