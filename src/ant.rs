use ndarray::Array2;
use rand::{prelude::*, rng};
use rand_distr::weighted::WeightedIndex;
use std::collections::HashSet;

pub fn get_vaporise_matrix(pheramones: &Array2<f64>) -> Array2<f64> {
    let mut result = Array2::zeros(pheramones.dim());
    let dim = pheramones.dim();

    for i in 0..dim.0 {
        for j in 0..dim.1 {
            result[[i, j]] = pheramones[[i, j]] * 0.3;
        }
    }

    result
}

pub fn get_pheramones(nodes: &Array2<f64>, prev_pheramones: &Array2<f64>) -> Array2<f64> {
    let mut pheramones = Array2::zeros(nodes.dim());
    let mut visited_nodes = HashSet::new();
    let mut rng = rng();
    let nodes_count = nodes.dim().0;
    let mut current_node = rng.random_range(0..nodes_count);

    while visited_nodes.len() != nodes_count - 1 {
        visited_nodes.insert(current_node);

        let next_node = WeightedIndex::new((0..nodes_count).map(|index| {
            if visited_nodes.contains(&index) {
                return 0.0;
            }

            let mut distance = *nodes.get((current_node, index)).unwrap_or(&f64::MAX);

            if distance == 0.0 {
                distance = 0.001;
            }

            let pheramones_level = *prev_pheramones.get((current_node, index)).unwrap_or(&0.0);
            let weight = (1.0 / distance).powi(3) + pheramones_level.powi(2);

            if !weight.is_finite() || weight < 0.0 {
                0.0
            } else {
                weight
            }
        }))
        .unwrap()
        .sample(&mut rng);

        let mut distance_to_next_node = *nodes.get((current_node, next_node)).unwrap_or(&f64::MAX);
        if distance_to_next_node == 0.0 {
            distance_to_next_node = 0.001;
        }

        let new_value = (1.0 / distance_to_next_node).powi(3);

        pheramones[[current_node, next_node]] = new_value;
        pheramones[[next_node, current_node]] = new_value;

        current_node = next_node;
    }

    return pheramones;
}

pub fn get_best_path(pheramones: &Array2<f64>) -> HashSet<(usize, usize)> {
    let mut result = HashSet::new();
    let mut visited_nodes = HashSet::new();
    let mut current_node = 0;
    let nodes_count = pheramones.dim().0;

    while visited_nodes.len() != nodes_count - 1 {
        visited_nodes.insert(current_node);

        let next_node = (0..nodes_count)
            .fold(None, |acc, el| {
                if visited_nodes.contains(&el) {
                    return acc;
                }

                match (acc, el) {
                    (None, el) => Some(el),
                    (Some(acc), el) => {
                        let pheramon_to_acc_element =
                            *pheramones.get((current_node, acc)).unwrap_or(&0.0);
                        let pheramon_to_next_element =
                            *pheramones.get((current_node, el)).unwrap_or(&0.0);

                        if pheramon_to_acc_element <= pheramon_to_next_element {
                            return Some(el);
                        }

                        Some(acc)
                    }
                }
            })
            .unwrap();

        result.insert((current_node, next_node));
        result.insert((next_node, current_node));
        current_node = next_node;
    }

    return result;
}

pub fn get_path_distance(path: &HashSet<(usize, usize)>, distances: &Array2<f64>) -> f64 {
    path.iter()
        .fold(0.0, |curr, el| curr + distances.get(*el).unwrap_or(&0.0))
}
