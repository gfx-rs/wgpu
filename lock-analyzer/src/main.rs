//! Analyzer for data produced by `wgpu-core`'s `observe_locks` feature.
//!
//! When `wgpu-core`'s `observe_locks` feature is enabled, if the
//! `WGPU_CORE_LOCK_OBSERVE_DIR` environment variable is set to the
//! path of an existing directory, then every thread that acquires a
//! lock in `wgpu-core` will write its own log file to that directory.
//! You can then run this program to read those files and summarize
//! the results.
//!
//! This program also consults the `WGPU_CORE_LOCK_OBSERVE_DIR`
//! environment variable to find the log files written by `wgpu-core`.
//!
//! See `wgpu_core/src/lock/observing.rs` for a general explanation of
//! this analysis.

use std::sync::Arc;
use std::{
    collections::{btree_map::Entry, BTreeMap, BTreeSet, HashMap},
    fmt,
    path::PathBuf,
};

use anyhow::{Context, Result};

// When printing the rank graph, include the source locations where the acquisitions were
// observed. Setting this to `false` produces output that more closely matches `rank.rs`.
const PRINT_ACQUISITION_LOCATIONS: bool = false;

fn main() -> Result<()> {
    let mut ranks: BTreeMap<u32, Rank> = BTreeMap::default();

    let Ok(dir) = std::env::var("WGPU_CORE_LOCK_OBSERVE_DIR") else {
        eprintln!(concat!(
            "Please set the `WGPU_CORE_LOCK_OBSERVE_DIR` environment variable\n",
            "to the path of the directory containing the files written by\n",
            "`wgpu-core`'s `observe_locks` feature."
        ));
        anyhow::bail!("`WGPU_CORE_LOCK_OBSERVE_DIR` environment variable is not set");
    };
    let entries =
        std::fs::read_dir(&dir).with_context(|| format!("failed to read directory {dir}"))?;
    for entry in entries {
        let entry = entry.with_context(|| format!("failed to read directory entry from {dir}"))?;
        let name = PathBuf::from(&entry.file_name());
        let Some(extension) = name.extension() else {
            eprintln!("Ignoring {}", name.display());
            continue;
        };
        if extension != "ron" {
            eprintln!("Ignoring {}", name.display());
            continue;
        }

        let contents = std::fs::read(entry.path())
            .with_context(|| format!("failed to read lock observations from {}", name.display()))?;
        // The addresses of `&'static Location<'static>` values could
        // vary from run to run.
        let mut locations: HashMap<u64, Arc<Location>> = HashMap::default();
        for line in contents.split(|&b| b == b'\n') {
            if line.is_empty() {
                continue;
            }
            let action = ron::de::from_bytes::<Action>(line)
                .with_context(|| format!("Error parsing action from {}", name.display()))?;
            match action {
                Action::Location {
                    address,
                    file,
                    line,
                    column,
                } => {
                    let file = match file.split_once("src/") {
                        Some((_, after)) => after.to_string(),
                        None => file,
                    };
                    assert!(locations
                        .insert(address, Arc::new(Location { file, line, column }))
                        .is_none());
                }
                Action::Rank {
                    bit,
                    member_name,
                    const_name,
                } => match ranks.entry(bit) {
                    Entry::Occupied(occupied) => {
                        let rank = occupied.get();
                        assert_eq!(rank.member_name, member_name);
                        assert_eq!(rank.const_name, const_name);
                    }
                    Entry::Vacant(vacant) => {
                        vacant.insert(Rank {
                            member_name,
                            const_name,
                            acquisitions: BTreeMap::default(),
                        });
                    }
                },
                Action::Acquisition {
                    older_rank,
                    older_location,
                    newer_rank,
                    newer_location,
                } => {
                    let older_location = locations[&older_location].clone();
                    let newer_location = locations[&newer_location].clone();
                    ranks
                        .get_mut(&older_rank)
                        .unwrap()
                        .acquisitions
                        .entry(newer_rank)
                        .or_default()
                        .entry(older_location)
                        .or_default()
                        .insert(newer_location);
                }
            }
        }
    }

    let (order, connected) = order_ranks(&ranks);

    // Assemble the ranks into three groups, in this list:
    // 1. Non-leaf ranks, in the topological order.
    // 2. Leaf ranks that are connected to the graph, alphabetically.
    // 3. Ranks that aren't connected to the graph at all, alphabetically.
    let mut grouped_order = Vec::with_capacity(order.len());

    // Non-leaf ranks in topological order.
    grouped_order.extend(order.iter().filter(|bit| !ranks[bit].is_leaf()));

    // Leaf ranks that are connected to the graph, in alphabetical order.
    grouped_order.extend({
        let mut tmp = order
            .iter()
            .copied()
            .filter(|bit| ranks[bit].is_leaf() && connected & (1 << *bit) != 0)
            .collect::<Vec<_>>();
        tmp.sort_by_key(|bit| &ranks[bit].const_name);
        tmp
    });

    // Ranks that aren't connected to the graph at all, in alphabetical order.
    grouped_order.extend({
        let mut tmp = order
            .iter()
            .copied()
            .filter(|bit| ranks[bit].is_leaf() && connected & (1 << *bit) == 0)
            .collect::<Vec<_>>();
        tmp.sort_by_key(|bit| &ranks[bit].const_name);
        tmp
    });

    // Finally, compute indexes so we can sort the followers of each rank in the same order
    // as the ranks themselves.
    let mut place_in_group_order = vec![0; ranks.len()];
    for (i, &bit) in grouped_order.iter().enumerate() {
        place_in_group_order[bit as usize] = i;
    }

    for bit in &grouped_order {
        let older_rank = &ranks[bit];
        if older_rank.is_leaf() {
            println!(
                "    rank {} {:?} followed by {{ }};",
                older_rank.const_name, older_rank.member_name
            );
            continue;
        }
        println!(
            "    rank {} {:?} followed by {{",
            older_rank.const_name, older_rank.member_name
        );
        let mut acquisitions = older_rank.acquisitions.iter().collect::<Vec<_>>();
        acquisitions.sort_by_key(|(&newer_rank, _)| place_in_group_order[newer_rank as usize]);
        for (newer_rank, locations) in acquisitions {
            if PRINT_ACQUISITION_LOCATIONS {
                for (older_location, newer_locations) in locations {
                    if newer_locations.len() == 1 {
                        for newer_loc in newer_locations {
                            println!(
                                "        // holding {older_location} while locking {newer_loc}"
                            );
                        }
                    } else {
                        println!("        // holding {older_location} while locking:");
                        for newer_loc in newer_locations {
                            println!("        //     {newer_loc}");
                        }
                    }
                }
            }
            println!("        {},", ranks[newer_rank].const_name);
        }

        println!("    }}");
    }

    Ok(())
}

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
enum Action {
    /// A location that we will refer to in later actions.
    Location {
        address: LocationAddress,
        file: String,
        line: u32,
        column: u32,
    },

    /// A lock rank that we will refer to in later actions.
    Rank {
        bit: u32,
        member_name: String,
        const_name: String,
    },

    /// An attempt to acquire a lock while holding another lock.
    Acquisition {
        /// The number of the already acquired lock's rank.
        older_rank: u32,

        /// The source position at which we acquired it. Specifically,
        /// its `Location`'s address, as an integer.
        older_location: LocationAddress,

        /// The number of the rank of the lock we are acquiring.
        newer_rank: u32,

        /// The source position at which we are acquiring it.
        /// Specifically, its `Location`'s address, as an integer.
        newer_location: LocationAddress,
    },
}

/// The memory address at which the `Location` was stored in the
/// observed process.
///
/// This is not `usize` because it does not represent an address in
/// this `lock-analyzer` process. We might generate logs on a 64-bit
/// machine and analyze them on a 32-bit machine. The `u64` type is a
/// reasonable universal type for addresses on any machine.
type LocationAddress = u64;

struct Rank {
    member_name: String,
    const_name: String,
    acquisitions: BTreeMap<u32, LocationSet>,
}

impl Rank {
    fn is_leaf(&self) -> bool {
        self.acquisitions.is_empty()
    }
}

type LocationSet = BTreeMap<Arc<Location>, BTreeSet<Arc<Location>>>;

#[derive(Eq, Ord, PartialEq, PartialOrd)]
struct Location {
    file: String,
    line: u32,
    column: u32,
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.file, self.line)
    }
}

/// Generate a topological order for lock ranks, and find unconnected ranks.
///
/// Return a pair `(ranks, connected)` such that:
///
/// - `ranks` is a vector of rank numbers in which each rank R occurs
///   before all other ranks acquired while R is held.
///
/// - `connected` is a bit mask of all ranks that are acquired while
///   some other rank is held. In other words, this is the set of
///   ranks that have predecessors in the partial order.
fn order_ranks(ranks: &BTreeMap<u32, Rank>) -> (Vec<u32>, u64) {
    let mut order = Vec::with_capacity(ranks.len());
    let mut connected = 0; // mask of ranks that are connected to the graph

    // Generate the ordering in reverse: traverse the graph in
    // depth-first order, adding a rank to end of the list only after
    // all other ranks reachable from it have been appended.

    // A mask of the ranks already visited by the traversal.
    let mut visited = 0;

    fn visit(
        bit: u32,
        visited: &mut u64,
        ranks: &BTreeMap<u32, Rank>,
        order: &mut Vec<u32>,
        connected: &mut u64,
    ) {
        if *visited & (1 << bit) != 0 {
            return;
        }
        *visited |= 1 << bit;

        for &next_rank_bit in ranks[&bit].acquisitions.keys() {
            *connected |= 1 << next_rank_bit;
            visit(next_rank_bit, visited, ranks, order, connected);
        }

        order.push(bit);
    }

    for &bit in ranks.keys() {
        visit(bit, &mut visited, ranks, &mut order, &mut connected);
    }

    // Reverse the ordering, so visitors appear before the visited.
    order.reverse();

    (order, connected)
}
