use ndarray::{Array2, Axis};
use ndarray_linalg::{Eigh, UPLO}; // For eigen decomposition
use raylib::prelude::*;

use crate::aco_file_parser::AcoFileParser;
mod aco_file_parser;

fn main() {
    let file = AcoFileParser::new(String::from("./graphs/SHP155.aco"))
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

    let scale = 300.0;
    let positions: Vec<(f32, f32)> = xs
        .iter()
        .zip(ys.iter())
        .map(|(&x, &y)| {
            (
                ((x - minx) / (maxx - minx) * 2.0 - 1.0) as f32 * scale + 400.0,
                ((y - miny) / (maxy - miny) * 2.0 - 1.0) as f32 * scale + 400.0,
            )
        })
        .collect();

    let (mut rl, thread) = raylib::init().size(800, 800).title("Ant system").build();

    while !rl.window_should_close() {
        let mut d = rl.begin_drawing(&thread);

        d.clear_background(Color::WHITE);
        // draw edges (weighted)
        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                let (x1, y1) = positions[i];
                let (x2, y2) = positions[j];

                d.draw_line_ex(
                    Vector2::new(x1, y1),
                    Vector2::new(x2, y2),
                    1.0,
                    Color::alpha(&Color::GRAY, 0.2),
                );
            }
        }

        // draw nodes
        for (_, &(x, y)) in positions.iter().enumerate() {
            d.draw_circle(x as i32, y as i32, 2.0, Color::SKYBLUE);
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
