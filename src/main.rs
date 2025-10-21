use std::{
    collections::{HashSet, VecDeque},
    sync::{Arc, Mutex},
    thread,
};

use itertools::Itertools;
use ndarray::{Array2, Axis};
use ndarray_linalg::{Eigh, UPLO}; // For eigen decomposition
use raylib::prelude::*;

use crate::{
    aco_file_parser::AcoFileParser,
    ant::{get_best_path, get_path_distance, get_pheramones, get_vaporise_matrix},
};
mod aco_file_parser;
mod ant;

fn main() {
    let file = AcoFileParser::new(String::from("./graphs/test30.aco"))
        .read_file()
        .unwrap();

    let dist = Array2::from_shape_vec((file.size, file.size), file.rows).unwrap();
    let coords = metric_mds(&dist, 2);
    let xs = coords.column(0);
    let ys = coords.column(1);

    // normalize positions
    let minx = xs.iter().cloned().fold(f64::INFINITY, f64::min);
    let maxx = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let miny = ys.iter().cloned().fold(f64::INFINITY, f64::min);
    let maxy = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let scale = 200.0;
    let positions: Vec<(f32, f32)> = xs
        .iter()
        .zip(ys.iter())
        .map(|(&x, &y)| {
            (
                ((x - minx) / (maxx - minx) * 2.0 - 1.0) as f32 * scale + 400.0,
                ((y - miny) / (maxy - miny) * 2.0 - 1.0) as f32 * scale + 300.0,
            )
        })
        .collect();

    let (mut rl, thread) = raylib::init().size(800, 800).title("Ant system").build();

    let best_path = Arc::new(Mutex::new(HashSet::new()));
    let best_path_clone = Arc::clone(&best_path);
    let scores_history: Arc<Mutex<VecDeque<f64>>> = Arc::new(Mutex::new(VecDeque::new()));
    let scores_history_clone = Arc::clone(&scores_history);
    let dist_calculate_thread_copy = dist.clone();

    thread::spawn(move || {
        let dist = dist_calculate_thread_copy;
        let mut pheramones = Array2::<f64>::zeros(dist.dim());

        loop {
            update_pheromones(&mut pheramones, &dist);

            let mut best_path = best_path_clone.lock().unwrap();
            *best_path = get_best_path(&pheramones);

            let mut scores_history = scores_history_clone.lock().unwrap();
            scores_history.push_back(get_path_distance(&best_path, &dist));

            if scores_history.len() > 100 {
                scores_history.pop_front();
            }
        }
    });

    let mut max_recorded = f64::MIN;
    let mut min_recorded = f64::MAX;

    while !rl.window_should_close() {
        let mut d = rl.begin_drawing(&thread);
        let current_best = {
            let best = best_path.lock().unwrap();
            best.clone()
        };

        let scores_history = {
            let histroy = scores_history.lock().unwrap();
            histroy.clone()
        };

        let total_distance = current_best
            .iter()
            .fold(0.0, |curr, el| curr + dist.get(*el).unwrap_or(&0.0));

        d.clear_background(Color::WHITE);
        d.draw_text(
            format!("Total distance {total_distance}").as_str(),
            12,
            12,
            20,
            Color::BLACK,
        );

        d.draw_text(
            format!("Max/MIN {max_recorded}/{min_recorded}").as_str(),
            12,
            32,
            20,
            Color::BLACK,
        );

        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                let (x1, y1) = positions[i];
                let (x2, y2) = positions[j];
                let color = match current_best.contains(&(i, j)) {
                    true => Color::GREEN,
                    false => Color::alpha(&Color::GRAY, 0.01),
                };

                d.draw_line_ex(Vector2::new(x1, y1), Vector2::new(x2, y2), 1.0, color);
            }
        }

        // draw nodes
        for (_, &(x, y)) in positions.iter().enumerate() {
            d.draw_circle(x as i32, y as i32, 2.0, Color::SKYBLUE);
        }

        let start = 50;
        max_recorded = {
            let new_max = *scores_history
                .iter()
                .max_by(|a, b| a.total_cmp(b))
                .unwrap_or(&0.0);

            if new_max > max_recorded {
                new_max
            } else {
                max_recorded
            }
        };

        min_recorded = {
            let new_min = *scores_history
                .iter()
                .min_by(|a, b| a.total_cmp(b))
                .unwrap_or(&f64::MAX);
            if new_min < min_recorded {
                new_min
            } else {
                min_recorded
            }
        };

        let width = 700;
        let x_step = width as f64 / scores_history.len() as f64;

        d.draw_text("Distance grapph", start, 600, 20, Color::BLACK);

        let points: Vec<(i32, i32)> = scores_history
            .iter()
            .enumerate()
            .map(|(index, score)| {
                let x = start as f64 + x_step * index as f64;
                let y = 775.0
                    - ((score - min_recorded) / (max_recorded - min_recorded)).powi(3) * 150.0;
                (x as i32, y as i32)
            })
            .collect();

        for (index, point) in points.iter().enumerate() {
            let next_point = points.get(index + 1);

            if let Some(next_point) = next_point {
                d.draw_line(point.0, point.1, next_point.0, next_point.1, Color::BLACK);
            }
        }
    }
}

pub fn metric_mds(distances: &Array2<f64>, n_components: usize) -> Array2<f64> {
    let n = distances.shape()[0];

    // --- Step 1: Square distances ---
    let mut d2 = distances.mapv(|x| x.powi(2));

    // --- Step 2: Double centering: B = -0.5 * J * D^2 * J ---
    let mean_row = d2.mean_axis(Axis(1)).unwrap();
    let mean_col = d2.mean_axis(Axis(0)).unwrap();
    let mean_total = d2.mean().unwrap();

    for i in 0..n {
        for j in 0..n {
            d2[[i, j]] = -0.5 * (d2[[i, j]] - mean_row[i] - mean_col[j] + mean_total);
        }
    }

    // --- Step 3: Eigen decomposition ---
    let (eigvals, eigvecs) = d2.eigh(UPLO::Upper).unwrap();

    // --- Step 4: Sort eigenvalues/vectors descending ---
    let mut eig_pairs: Vec<_> = eigvals
        .iter()
        .cloned()
        .zip(eigvecs.axis_iter(Axis(1)))
        .collect();
    eig_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // --- Step 5: Take top components ---
    let selected: Vec<_> = eig_pairs.iter().take(n_components).collect();

    // --- Step 6: Compute coordinates ---
    let mut coords = Array2::<f64>::zeros((n, n_components));

    for (k, (val, vec)) in selected.iter().enumerate() {
        let scale = val.sqrt();
        let v = vec.to_owned().into_shape((n, 1)).unwrap();
        for i in 0..n {
            coords[[i, k]] = v[[i, 0]] * scale;
        }
    }

    coords
}

fn update_pheromones(pheramones: &mut Array2<f64>, dist: &Array2<f64>) {
    for _ in 0..100 {
        *pheramones += &get_pheramones(dist, pheramones);
        *pheramones -= &get_vaporise_matrix(pheramones);
    }
}
