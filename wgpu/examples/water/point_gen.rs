//!
//! This module covers generating points in a hexagonal fashion.
//!

use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;

// The following constants are used in calculations.
// A and B are multiplication factors for x and y.

///
/// X multiplication factor.
/// 1.0 / sqrt(2)
///
const A: f32 = std::f32::consts::FRAC_1_SQRT_2;

///
/// Y multiplication factor.
/// sqrt(3) / sqrt(2) == sqrt(1.5)
///
const B: f32 = SQRT_3 * A;

///
/// `sin(45deg)` is used to rotate the points.
///
const S45: f32 = std::f32::consts::FRAC_1_SQRT_2;
///
/// `cos(45deg)` is used to rotate the points.
///
const C45: f32 = S45;

const SQRT_3: f32 = 1.7320508;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct TerrainVertexAttributes {
    position: [f32; 3],
    normal: [f32; 3],
    colour: [u8; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Pod, Zeroable)]
pub struct WaterVertexAttributes {
    position: [i16; 2],
    offsets: [i8; 4],
}

///
/// Represents the center of a single hexagon.
///
#[derive(Copy, Clone, Debug)]
pub struct TerrainVertex {
    pub position: glam::Vec3,
    pub colour: [u8; 4],
}

///
/// Gets the surrounding hexagonal points from a point.
///
/// +---0---1
/// | / |   |
/// 5---p---2
/// |   | / |
/// 4---3---+
///
fn surrounding_hexagonal_points(x: isize, y: isize) -> [(isize, isize); 6] {
    [
        (x, y - 1),
        (x + 1, y - 1),
        (x + 1, y),
        (x, y + 1),
        (x - 1, y + 1),
        (x - 1, y),
    ]
}

fn surrounding_point_values_iter<T>(
    hashmap: &HashMap<(isize, isize), T>,
    x: isize,
    y: isize,
    for_each: impl FnMut((&T, &T)),
) {
    let points = surrounding_hexagonal_points(x, y);
    let points = [
        points[0], points[1], points[2], points[3], points[4], points[5], points[0],
    ];
    points
        .windows(2)
        .map(|x| (hashmap.get(&x[0]), hashmap.get(&x[1])))
        .flat_map(|(a, b)| a.and_then(|x| b.map(|y| (x, y))))
        .for_each(for_each);
}

///
/// Used in calculating terrain normals.
///
pub fn calculate_normal(a: glam::Vec3, b: glam::Vec3, c: glam::Vec3) -> glam::Vec3 {
    (b - a).normalize().cross((c - a).normalize()).normalize()
}

///
/// Given the radius, how large of a square do we need to make a unit hexagon grid?
///
fn q_given_r(radius: f32) -> usize {
    ((((((4.0 * radius) / SQRT_3) + 1.0).floor() / 2.0).floor() * 2.0) + 1.0) as usize
}

///
/// Represents terrain, however it contains the vertices only once.
///
#[derive(Clone)]
pub struct HexTerrainMesh {
    pub vertices: HashMap<(isize, isize), TerrainVertex>,
    half_size: isize,
}

impl HexTerrainMesh {
    ///
    /// Generates the vertices (or the centers of the hexagons). The colour and height is determined by
    /// a function passed in by the user.
    ///
    pub fn generate(radius: f32, mut gen_vertex: impl FnMut([f32; 2]) -> TerrainVertex) -> Self {
        let width = q_given_r(radius);
        let half_width = (width / 2) as isize;
        let mut map = HashMap::new();
        let mut max = std::f32::NEG_INFINITY;
        for i in -half_width..=half_width {
            let x_o = i as f32;
            for j in -half_width..=half_width {
                let y_o = j as f32;
                let x = A * (x_o * C45 - y_o * S45);
                let z = B * (x_o * S45 + y_o * C45);
                if x.hypot(z) < radius {
                    let vertex = gen_vertex([x, z]);
                    if vertex.position.y > max {
                        max = vertex.position.y;
                    }
                    map.insert((i, j), vertex);
                }
            }
        }
        Self {
            vertices: map,
            half_size: width as isize / 2,
        }
    }

    ///
    /// Creates the points required to render the mesh.
    ///
    pub fn make_buffer_data(&self) -> Vec<TerrainVertexAttributes> {
        let mut vertices = Vec::new();
        fn middle(p1: &TerrainVertex, p2: &TerrainVertex, p: &TerrainVertex) -> glam::Vec3 {
            (p1.position + p2.position + p.position) / 3.0
        }
        fn half(p1: &TerrainVertex, p2: &TerrainVertex) -> glam::Vec3 {
            (p1.position + p2.position) / 2.0
        }
        let mut push_triangle = |p1: &TerrainVertex,
                                 p2: &TerrainVertex,
                                 p: &TerrainVertex,
                                 c: [u8; 4]| {
            let m = middle(p1, p2, p);
            let ap = half(p1, p);
            let bp = half(p2, p);
            let p = p.position;
            let n1 = calculate_normal(ap, m, p);
            let n2 = calculate_normal(m, bp, p);

            vertices.extend(
                [ap, m, p, m, bp, p]
                    .iter()
                    .zip(
                        std::iter::repeat::<[f32; 3]>(n1.into())
                            .chain(std::iter::repeat::<[f32; 3]>(n2.into())),
                    )
                    .zip(std::iter::repeat(c))
                    .map(|((pos, normal), colour)| TerrainVertexAttributes {
                        position: *pos.as_ref(),
                        normal,
                        colour,
                    }),
            );
        };
        for i in -self.half_size..=self.half_size {
            for j in -self.half_size..=self.half_size {
                if let Some(p) = self.vertices.get(&(i, j)) {
                    surrounding_point_values_iter(&self.vertices, i, j, |(a, b)| {
                        push_triangle(a, b, p, p.colour)
                    });
                }
            }
        }
        vertices
    }
}

///
/// Water mesh which contains vertex data for the water mesh.
///
/// It stores the values multiplied and rounded to the
/// nearest whole number to be more efficient with space when
/// sending large meshes to the GPU.
///
pub struct HexWaterMesh {
    pub vertices: HashMap<(isize, isize), [i16; 2]>,
    half_size: isize,
}

impl HexWaterMesh {
    pub fn generate(radius: f32) -> Self {
        let width = q_given_r(radius);
        let half_width = (width / 2) as isize;
        let mut map = HashMap::new();

        for i in -half_width..=half_width {
            let x_o = i as f32;
            for j in -half_width..=half_width {
                let y_o = j as f32;
                let x = A * (x_o * C45 - y_o * S45);
                let z = B * (x_o * S45 + y_o * C45);
                if x.hypot(z) < radius {
                    let x = (x * 2.0).round() as i16;
                    let z = ((z / B) * std::f32::consts::SQRT_2).round() as i16;
                    map.insert((i, j), [x, z]);
                }
            }
        }
        Self {
            vertices: map,
            half_size: half_width,
        }
    }
    ///
    /// Generates the points required to render the mesh.
    ///
    pub fn generate_points(&self) -> Vec<WaterVertexAttributes> {
        let mut vertices = Vec::new();

        fn calculate_differences(a: [i16; 2], b: [i16; 2], c: [i16; 2]) -> [i8; 4] {
            [
                (b[0] - a[0]) as i8,
                (b[1] - a[1]) as i8,
                (c[0] - a[0]) as i8,
                (c[1] - a[1]) as i8,
            ]
        }

        let mut push_triangle = |a: [i16; 2], b: [i16; 2], c: [i16; 2]| {
            let bc = calculate_differences(a, b, c);
            let ca = calculate_differences(b, c, a);
            let ab = calculate_differences(c, a, b);

            vertices.extend(
                [a, b, c]
                    .iter()
                    .zip([bc, ca, ab].iter())
                    .map(|(&position, &offsets)| WaterVertexAttributes { position, offsets }),
            );
        };

        for i in -self.half_size..=self.half_size {
            for j in -self.half_size..=self.half_size {
                if (i - j) % 3 == 0 {
                    if let Some(&p) = self.vertices.get(&(i, j)) {
                        surrounding_point_values_iter(&self.vertices, i, j, |(a, b)| {
                            push_triangle(*a, *b, p)
                        });
                    }
                }
            }
        }

        vertices
    }
}
