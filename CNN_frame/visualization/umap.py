import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm
import scipy.sparse
import numba
import matplotlib.pyplot as plt

# x 维度 [N,D]
def cal_pairwise_dist(x):
    print("compute distance")
    N,D = np.shape(x)
    
    dist = np.zeros([N,N])
    
    for i in tqdm(range(N)):
        for j in range(N):
            dist[i,j] = np.sqrt(np.dot((x[i]-x[j]),(x[i]-x[j]).T))

    #返回任意两个点之间距离
    return dist

# 获取每个样本点的 n_neighbors个临近点的位置以及距离
def get_n_neighbors(data, n_neighbors = 15):
    dist = cal_pairwise_dist(data)
    dist[dist < 0] = 0
    N = dist.shape[0]
    NN_index = np.argsort(dist,axis=1)[:,0:n_neighbors]
    NN_dist = np.sort(dist,axis=1)[:,0:n_neighbors]
    return NN_index,NN_dist
    
# 计算每个样本点的参数 sigmas 以及 rhos
def compute_sigmas_and_rhos(distances,k,
                            local_connectivity=1,n_iter=64,
                            tol = 1.0e-5,min_k_dis_scale=1e-3):
    print("computing sigmas and rhos")
    # 获取样本数目
    N= distances.shape[0]
    
    # 定义变量存储每个样本的 sigma 和 rho
    rhos = np.zeros(N, dtype=np.float32)
    sigmas = np.zeros(N, dtype=np.float32)
    
    mean_distances = np.mean(distances)
    
    target = np.log2(k)
    
    for i in tqdm(range(N)):
        lo = 0.0
        hi = np.inf
        mid = 1.0
        
        # rho_i 为距离第i个样本最近的第local_connectivity个距离
        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        rhos[i] = non_zero_dists[local_connectivity - 1]
        
        # 通过2值搜索的方法计算sigma_i
        for n in range(n_iter):

            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = distances[i, j] - rhos[i]
                if d > 0:
                    psum += np.exp(-(d / mid))
                else:
                    psum += 1.0

            if np.fabs(psum - target) < tol:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == np.inf:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0
        
        sigmas[i] = mid
        
        # 进一步处理 防止 sigma_i 过小
        if rhos[i] > 0.0:
            mean_ith_distances = np.mean(ith_distances)
            if sigmas[i] < min_k_dis_scale * mean_ith_distances:
                sigmas[i] = min_k_dis_scale * mean_ith_distances
        # rhos[i]<=0 N个近邻点距离过近
        else:
            if sigmas[i] < min_k_dis_scale * mean_distances:
                sigmas[i] = min_k_dis_scale * mean_distances
    
    return sigmas,rhos

# 计算两点间的连接强度
def compute_membership_strengths(NN_index,NN_dists,sigmas,rhos):
    
    print("compute membership strengths")
    n_samples, n_neighbors =  np.shape(NN_index) 
    
    rows = np.zeros(n_samples*n_neighbors, dtype=np.int32)
    cols = np.zeros(n_samples*n_neighbors, dtype=np.int32)
    vals = np.zeros(n_samples*n_neighbors, dtype=np.float32)
    
    for i in tqdm(range(n_samples)):
       for j in range(n_neighbors):
           
            if NN_index[i, j] == i:
                val = 0.0
            elif NN_dists[i, j] - rhos[i] <= 0.0 or sigmas[i] == 0.0:
                val = 1.0
            else:
                val = np.exp(-((NN_dists[i, j] - rhos[i]) / (sigmas[i])))

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = NN_index[i, j]
            vals[i * n_neighbors + j] = val
            
    return rows, cols, vals

def get_graph_Inputs(X,n_neighbors,local_connectivity=1):
    n_samples = X.shape[0]
    
    # 计算每个样本点的N个临近点的位置和距离
    NN_index, NN_dists = get_n_neighbors(X,n_neighbors)
    
    # 计算每个样本的 sigm 与 rho 为后边的图计算提供参数
    sigmas,rhos = compute_sigmas_and_rhos(NN_dists,n_neighbors,local_connectivity)
    
    # 计算两点间的 连接强度 即计算条件概率 Pj|i
    rows, cols, vals = compute_membership_strengths(NN_index,NN_dists,sigmas,rhos)
    
    # 构造稀疏矩阵
    result = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(X.shape[0], X.shape[0]))
    result.eliminate_zeros() # 去掉0
    
    # 计算联合概率 Pij
    transpose = result.transpose()
    prod_matrix = result.multiply(transpose)
    # Pij  =  Pj|i + Pi|j - Pj|i* Pi|j
    result = result + transpose - prod_matrix
    return result

# 谱分析法进行初始化
def init_embedding_spectral(graph,dim):
    n_samples = graph.shape[0]
    k = dim
    diag_data = np.asarray(graph.sum(axis=0))
        
    # Normalized Laplacian
    I = scipy.sparse.identity(graph.shape[0], dtype=np.float64)
    D = scipy.sparse.spdiags(
        1.0 / np.sqrt(diag_data), 0, graph.shape[0], graph.shape[0]
    )
    L = I - D * graph * D
    num_lanczos_vectors = max(2 * k + 1, int(np.sqrt(graph.shape[0])))
    try:
        if L.shape[0] < 2000000:
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
                L,
                k+1,
                which="SM",
                ncv=num_lanczos_vectors,
                tol=1e-4,
                v0=np.ones(L.shape[0]),
                maxiter=graph.shape[0] * 5,
            )
        else:
            print("---------------eigenvalues-------------------")
            eigenvalues, eigenvectors = scipy.sparse.linalg.lobpcg(
                L, np.random.normal(size=(L.shape[0], k+1)), largest=False, tol=1e-8
            )
        order = np.argsort(eigenvalues)[1:k+1]
        return eigenvectors[:, order]
    except scipy.sparse.linalg.ArpackError:
        warn(
            "WARNING: spectral initialisation failed! The eigenvector solver\n"
            "failed. This is likely due to too small an eigengap. Consider\n"
            "adding some noise or jitter to your data.\n\n"
            "Falling back to random initialisation!"
        )
        return np.random.uniform(low=-10.0, high=10.0, size=(graph.shape[0], dim))

def make_epochs_per_sample(weights, n_epochs):
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    # 边的权重越大在整个训练过程中更新的次数越多，更新间隔越小
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / n_samples[n_samples > 0] # 更新间隔
    return result

# 梯度裁剪 -4，+4之间
@numba.njit()
def clip(val):
   
    if val > 4.0:
        return 4.0
    elif val < -4.0:
        return -4.0
    else:
        return val
        
        
def train_one_epoch(head_embedding,
                    tail_embedding,
                    head,
                    tail,
                    n_vertices,
                    epochs_per_sample,
                    epochs_per_negative_sample,
                    epoch_of_next_sample,
                    epoch_of_next_negative_sample,
                    a,
                    b,
                    alpha,
                    n,
                    dim): 
                    
    for i in range(epochs_per_sample.shape[0]):
        #对正样本进行采样
        if epoch_of_next_sample[i] <= n:
            
            j = head[i]
            k = tail[i]

            current = head_embedding[j]
            other = tail_embedding[k]
            
            # 计算两点间距离
            dist_squared = np.dot((current-other),(current-other))
            
            # 计算正样本梯度
            if dist_squared > 0.0:
                grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                grad_coeff /= a * pow(dist_squared, b) + 1.0
            else:
                grad_coeff = 0.0
                
            # 进行更新
            for d in range(dim):
                # 梯度裁剪
                grad_d = clip(grad_coeff * (current[d] - other[d]))
                # 梯度
                current[d] += grad_d * alpha
            
            # 下次更新的轮次
            epoch_of_next_sample[i] += epochs_per_sample[i]
            
            # 计算负样本的数目
            n_neg_samples = int(
                (n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
            )
            
            # 进行负样本采样
            for p in range(n_neg_samples):
                k = np.random.randint(n_vertices)

                other = tail_embedding[k]

                dist_squared = np.dot((current-other),(current-other))

                if dist_squared > 0.0:
                    grad_coeff = 2.0 * b
                    grad_coeff /= (0.001 + dist_squared) * (
                        a * pow(dist_squared, b) + 1
                    )
                elif j == k:
                    continue
                else:
                    grad_coeff = 0.0

                for d in range(dim):
                    if grad_coeff > 0.0:
                        grad_d = clip(grad_coeff * (current[d] - other[d]))
                    else:
                        grad_d = 4.0
                    current[d] += grad_d * alpha
            
            # 计算下次负样本更新轮次
            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )
    
# 通过训练获得embedding
def train_embedding(head_embedding, #头结点 向量
                    tail_embedding, #尾结点 向量
                    head, # 头结点 编号
                    tail, # 尾结点 编号
                    epochs_per_sample, # 正样本采样控制
                    epochs_per_negative_sample, # 负样本采样控制
                    a,b, # 
                    initial_alpha, #  初始化学习率
                    n_epochs, # 训练轮次
                    n_vertices # 顶点数目
                    ):
    dim = head_embedding.shape[1]
    alpha = initial_alpha
    
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()
    
    optimize_fn = numba.njit(train_one_epoch, fastmath=True, parallel=False)
    
    for n in tqdm(range(n_epochs)):
        
        # 进行1轮更新
        optimize_fn(head_embedding,
                    tail_embedding,
                    head,
                    tail,
                    n_vertices,
                    epochs_per_sample,
                    epochs_per_negative_sample,
                    epoch_of_next_sample,
                    epoch_of_next_negative_sample,
                    a,
                    b,
                    alpha,
                    n,
                    dim)
        
        # 更新学习率
        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))
    
    return head_embedding
        
 
def get_embedding(graph,
                  dim,a,b,
                  negative_sample_rate,
                  n_epochs=None,
                  initial_alpha=1.0):
                  
    # 行列交换
    graph = graph.tocoo()
    graph.sum_duplicates()
    # 顶点数目
    n_vertices = graph.shape[1]
    
    # 计算迭代轮次 数据越少迭代轮次越多
    if n_epochs is None:
        if graph.shape[0] <= 10000:
            n_epochs = 500
        else:
            n_epochs = 200
    
    # 边的权重过低，无法采样，将权重设置为0
    if n_epochs > 10:
        graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
       
    graph.eliminate_zeros()
    
    # 利用谱分析的方法，借助graph，对低维数据进行初始化
    initialisation = init_embedding_spectral(graph,dim)
    
    # 加入一些随机数据增加随机性
    expansion = 10.0 / np.abs(initialisation).max()
    embedding = (initialisation * expansion).astype(
        np.float32
    ) + np.random.normal(
        scale=0.0001, size=[graph.shape[0], dim]
    ).astype(
        np.float32
    )
    
    # 计算图中每条边，每隔多少个epoch 更新一次
    epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)
    # 负样本，每隔多少个epoch 更新一次
    epochs_per_negative_sample = epochs_per_sample/negative_sample_rate
    
    # 开始进行训练，获取 embedding
    head = graph.row
    tail = graph.col
    
    # 训练获取降维数据
    embedding =train_embedding(embedding,embedding,
                              head,tail,
                              epochs_per_sample,epochs_per_negative_sample,
                              a,b,initial_alpha,n_epochs,n_vertices)
    return embedding
    
    
def find_ab_params(min_dist,spread):
    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]


def UAMP(X,
        dim=2, # 降维后的维度
        n_neighbors=15, # N近邻
        min_dist = 0.1, # 控制投影后，相似点的聚拢程度
        spread = 1,
        negative_sample_rate=5, # 负样本采样是正样本采样的多少倍
        n_epochs=None, # 训练轮次
        initial_alpha= 1.0 # 初始化学习率
        ):
    
    # 估算参数 a,b
    a,b = find_ab_params(min_dist,spread)
    
    # 根据高维数据 计算点与点之间的连接关系
    graph = get_graph_Inputs(X,n_neighbors,local_connectivity=1)
    print(graph)
    # 
    embedding = get_embedding(graph,dim,a,b,negative_sample_rate,n_epochs,initial_alpha)
    return embedding
    
    
def draw_pic(datas,labs,name = '1.jpg'):
    plt.cla()
    unque_labs = np.unique(labs)
    colors = [plt.cm.Spectral(each)
      for each in np.linspace(0, 1,len(unque_labs))]
    p=[]
    legends = []
    for i in range(len(unque_labs)):
        index = np.where(labs==unque_labs[i])
        pi = plt.scatter(datas[index, 0], datas[index, 1], c =[colors[i]] )
        p.append(pi)
        legends.append(unque_labs[i])

    plt.legend(p, legends)
    plt.savefig(name)
def use(feature_path,labels_path,save_picture_path):
    mnist_datas = np.loadtxt(feature_path)
    mnist_labs = np.loadtxt(labels_path)
    print(mnist_datas.shape)
    embedding = UAMP(mnist_datas, dim=2, min_dist=0.01, spread=1, n_neighbors=30)
    print(embedding.shape)

    draw_pic(embedding, mnist_labs, name=save_picture_path)
    plt.show()


if __name__ == "__main__":
    mnist_datas = np.loadtxt("C:/Users/DELL/Desktop/DLRS_sfy2.0/DLRS/features.txt")
    mnist_labs = np.loadtxt("C:/Users/DELL/Desktop/DLRS_sfy2.0/DLRS/labels.txt")
    print(mnist_datas.shape)
    # mnist_datas = mnist_datas[:500,:]
    embedding = UAMP(mnist_datas,dim=2,min_dist=0.01,spread=1,n_neighbors=30)
    print(embedding.shape)
    
    draw_pic(embedding,mnist_labs,name = "final-d0.01n-30_new.jpg")
    plt.show()

    
    
    
    
