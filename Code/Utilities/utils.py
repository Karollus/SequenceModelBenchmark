import pickle
import os
import functools

def batcherator(sample_generator, total_samples, num_of_jobs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base, rest = divmod(total_samples, num_of_jobs)
    job_sizes = rest * [base+1] + (num_of_jobs-rest) * [base]

    cumsum_job_sizes = functools.reduce(lambda x, y: x+y, job_sizes)
    assert cumsum_job_sizes == total_samples, \
        f"job_sizes should reduce to {total_samples}, got {cumsum_job_sizes}"

    job_idx = 0
    for job_size in job_sizes:
        job = []
        for i in range(job_size):
            sample = next(sample_generator)
            job.append(sample)
        assert len(job) == job_size
        with open(os.path.join(output_dir, f"{job_idx}.pkl"), "wb") as f:
            pickle.dump(job, f, protocol=5)
        job_idx += 1

    sentinel = object()
    assert next(sample_generator, sentinel) is sentinel, \
        "sample_generator still contains samples but should be empty"
