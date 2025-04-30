import jax
import jax.numpy as jnp
from ott.geometry import pointcloud
from ott.solvers.linear.sinkhorn_lr import LRSinkhorn
from ott.problems.linear import linear_problem
from ott.initializers.linear import initializers_lr

from src.rank_annealing import optimal_rank_schedule

def hierarchical_refinement_iterative(
        X, 
        Y,
        a = None,
        b = None,
        strategy: str = "rank_annealing",
        rank_schedule: list = None,
        ):
    """
    HiRef - version itérative utilisant ott-jax et low-rank Sinkhorn.
    
    Args:
        X (jax array): Points source (n, d).
        Y (jax array): Points cible (n, d).
        rank_schedule (list of int): Liste des rangs à utiliser à chaque profondeur.

    Returns:
        list of tuples (xi, yj): Les correspondances finales.
    """
    n = X.shape[0]
    assert n == Y.shape[0], "X et Y doivent avoir la même taille."

    if strategy == "rank_annealing":
        if rank_schedule is None:
            rank_schedule = optimal_rank_schedule(n=len(X))
            print(f"Optimized rank-annealing schedule: { rank_schedule }")
    else:
        return "Error! Strategy not implemented."
        
    correspondances = []

    # Chaque élément de la pile est (x_block, y_block, martingal X, martingal Y depth)
    stack = [(X, Y, 0)]

    while stack:
        ns = len(stack)
        print(f"Stack size: {ns}")

        x_block, y_block, depth = stack.pop()

        # Condition de sortie de notre boucle while. On veut que tous les éléments finissent à cette étape
        if x_block.shape[0] == 1:
            correspondances.append((x_block[0], y_block[0]))
            continue

        # Définir le rang à utiliser
        rank = rank_schedule[depth] if depth < len(rank_schedule) else 2

        print(f"\n[Depth {depth}] Processing block of size {x_block.shape[0]} with rank {rank}")

        ### BLOC DE RESOLUTION DES SOUS CLUSTERS
        # Géométrie
        geom = pointcloud.PointCloud(x_block, y_block)

        # Problème de transport
        ot_prob = linear_problem.LinearProblem(geom)        # pour l'instant on reste avec des lois uniformes
  

        # Initialiseur + solveur Low-Rank Sinkhorn
        initializer = initializers_lr.Rank2Initializer(rank=rank)
        #init_state = initializer(ot_prob, lse_mode=True)  # <<< correction ici

        solver = jax.jit(LRSinkhorn(
            rank=rank,
            initializer=initializer,
            min_iterations=100,
            threshold=1e-3,
            gamma=50000)
        )
        
        out = solver(ot_prob)

        # Récupérer Q et R
        Q = out.q
        R = out.r

        q_labels = jnp.argmax(Q, axis=1)
        r_labels = jnp.argmax(R, axis=1)

        # Pour chaque cluster
        for label in range(rank):
            idx_x = jnp.where(q_labels == label)[0]
            idx_y = jnp.where(r_labels == label)[0]

            if len(idx_x) == 0 or len(idx_y) == 0:
                continue  # Ignore clusters vides

            x_subblock = x_block[idx_x]
            y_subblock = y_block[idx_y]

            # Ajouter le sous-bloc à la pile
            stack.append((x_subblock, y_subblock, depth + 1))

    return correspondances
