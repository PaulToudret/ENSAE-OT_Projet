
import jax
import jax.numpy as jnp

from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.initializers.linear.initializers_lr import Rank2Initializer
from ott.solvers.linear.sinkhorn_lr import LRSinkhorn

def low_rank_partition(
        x_block: jnp.ndarray,
        y_block: jnp.ndarray,
        rank: int = 2
        ):
    """
    Renvoie pour chaque point de x_block et y_block 
    son indice de cluster {0,1} à rang=2, 
    en initialisant avec Rank2Initializer.
    """
    # 1) Problème OT
    geom = pointcloud.PointCloud(x_block, y_block)
    ot_prob = linear_problem.LinearProblem(geom)

    # 2) Initialiser avec Rank2Initializer (déterministe)
    init = Rank2Initializer(rank=rank)

    # 3) Créer le solver low‑rank Sinkhorn
    solver = jax.jit(LRSinkhorn(rank=rank, initializer=init, gamma=50000)) # Very important that gamma value is high (>100) 

    # 4) Lancer la résolution (pas besoin de jit pour du debug)
    out = solver(ot_prob)

    # 5) Les deux soft‑clustering Q et R → argmax pour labels {0,1}
    Q, R, G = out.q, out.r, out.g

    partition_x = jnp.argmax(Q, axis=1)
    partition_y = jnp.argmax(R, axis=1)

    return partition_x, partition_y, (Q,R,G)


def check_partition(Q, R, G):
    """
    Vérifier si la décomposition est valables vis-à-vis de la décomposition matricielle en lowrank
    """
    pass