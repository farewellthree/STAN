import numpy as np
a = np.random.rand(180000,768)
b = np.random.rand(180000,768)
def batched_dot_product(a, b, batch_size):
    m, n = a.shape
    q, p = b.shape

    result = np.zeros((m, q))

    num_batches = (q + batch_size - 1) // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, q)

        b_batch = b[start_idx:end_idx, :]
        result[:, start_idx:end_idx] = np.dot(a, b_batch.T)

    return result 
print (a@b.T)
print ('------------')
print (batched_dot_product(a,b,100))