//! Full-athlete evolution: open-loop periodic Discrete-9 key loops.
//!
//! Authentic QWOP-bot style: the genome is a repeating key sequence
//! (period 12 actions, each 0..8). No network, no gradients — pure
//! evolution on `QwopSim`. Fast enough for several iterations a day
//! on a 2013-era CPU.

use evogait_sim_core::{QwopSim, XorShift64};
use evogait_sim_core::policy::{FbPolicy, FbRunner, LOOP_LEN, rollout_fb};

pub const PERIOD: usize = 12;
pub const EVAL_STEPS: usize = 250; // 10 s at 25 Hz
pub const SAMPLE_SECONDS: f64 = 8.0;

pub type KeyLoop = [u8; PERIOD];

fn random_loop(rng: &mut XorShift64) -> KeyLoop {
    let mut g = [0u8; PERIOD];
    for v in &mut g {
        *v = (rng.next_u64() % 9) as u8;
    }
    g
}

fn mutate(rng: &mut XorShift64, g: &KeyLoop) -> KeyLoop {
    let mut out = *g;
    for v in &mut out {
        if rng.next_f64() < 0.25 {
            *v = (rng.next_u64() % 9) as u8;
        }
    }
    // Include at least one change most of the time.
    if out == *g {
        out[(rng.next_u64() as usize) % PERIOD] = (rng.next_u64() % 9) as u8;
    }
    out
}

/// Score: qwop distance with upright + alternation shaping. Only
/// head/arm contact ends a rollout; low lunges and knee-scoots are legal
/// locomotion. (y grows downward; standing torso is near y=-0.6.)
pub fn evaluate(g: &KeyLoop, steps: usize, hold: usize) -> (f64, bool) {
    let mut sim = QwopSim::new();
    let x0 = sim.torso_x();
    for i in 0..steps {
        if sim.fallen() {
            break;
        }
        sim.step_action(g[(i / hold) % PERIOD] as usize);
    }
    let dist = (sim.torso_x() - x0) / 10.0; // world units -> metres-ish
    let score = dist - if sim.fallen() { 2.0 } else { 0.0 };
    (score, sim.fallen())
}

pub struct FullResult {
    pub best: KeyLoop,
    pub best_score: f64,
    pub history: Vec<(usize, f64)>,
}

pub fn train(gens: usize, pop: usize, seed: u64, hold: usize) -> FullResult {
    let mut rng = XorShift64::new(seed);
    let mut population: Vec<KeyLoop> =
        (0..pop).map(|_| random_loop(&mut rng)).collect();
    let mut history = Vec::new();
    let mut best_overall = population[0];
    let mut best_score_overall = f64::NEG_INFINITY;

    for g in 0..gens {
        let mut scored: Vec<(KeyLoop, f64)> = population
            .iter()
            .map(|genome| {
                let (s, _) = evaluate(genome, EVAL_STEPS, hold);
                (*genome, s)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let elite_n = (pop / 4).max(2).min(pop);
        if scored[0].1 > best_score_overall {
            best_score_overall = scored[0].1;
            best_overall = scored[0].0;
        }
        history.push((g, scored[0].1));
        if g % 10 == 0 || g + 1 == gens {
            eprintln!(
                "full gen {g:4} best {:7.3} overall {:7.3} loop {:?}",
                scored[0].1, best_score_overall, scored[0].0
            );
        }
        // Next generation: elites + mutated copies of elites.
        let mut next = Vec::with_capacity(pop);
        for (genome, _) in scored.iter().take(elite_n) {
            next.push(*genome);
        }
        while next.len() < pop {
            let parent =
                scored[(rng.next_u64() as usize) % elite_n].0;
            next.push(mutate(&mut rng, &parent));
        }
        population = next;
    }

    FullResult {
        best: best_overall,
        best_score: best_score_overall,
        history,
    }
}

/// Sample a v2 frame track: 12 bodies x (x, y, angle) at 30 fps.
pub fn sample_v2(g: &KeyLoop, seconds: f64, hold: usize) -> (Vec<[[f64; 3]; 12]>, f64, bool) {
    let mut sim = QwopSim::new();
    let x0 = sim.torso_x();
    let mut frames = Vec::new();
    // 25 Hz physics -> 30 fps sampling: accumulate sim time.
    let mut next_sample = 0.0;
    let mut i = 0;
    while sim.time() < seconds && !sim.fallen() {
        sim.step_action(g[(i / hold) % PERIOD] as usize);
        i += 1;
        if sim.time() >= next_sample {
            frames.push(sim.bodies_xyw());
            next_sample += 1.0 / 30.0;
        }
    }
    let dist = (sim.torso_x() - x0) / 10.0;
    (frames, dist, sim.fallen())
}

/// Flail baseline: keys-up drift.
pub fn sample_flail(seconds: f64) -> (Vec<[[f64; 3]; 12]>, f64) {
    let mut sim = QwopSim::new();
    let x0 = sim.torso_x();
    let mut frames = Vec::new();
    let mut next_sample = 0.0;
    while sim.time() < seconds && !sim.fallen() {
        sim.step_idle();
        if sim.time() >= next_sample {
            frames.push(sim.bodies_xyw());
            next_sample += 1.0 / 30.0;
        }
    }
    ((frames), (sim.torso_x() - x0) / 10.0)
}

/// Feedback-policy evolution: same ES loop over FbPolicy genomes.
/// Seeded from a strong open-loop winner when provided.
pub struct FbResult {
    pub best: FbPolicy,
    pub best_score: f64,
    pub best_dist: f64,
    pub history: Vec<(usize, f64)>,
}

fn random_fb(rng: &mut XorShift64) -> FbPolicy {
    let mut keys = [0u8; LOOP_LEN];
    for v in &mut keys {
        *v = (rng.next_u64() % 9) as u8;
    }
    FbPolicy {
        keys,
        hold: 4.0 + rng.next_f64() * 10.0,
        k_fb: rng.next_f64(),
        reset: rng.next_f64() < 0.5,
    }
}

fn mutate_fb(rng: &mut XorShift64, p: &FbPolicy, scale: f64) -> FbPolicy {
    // Annealed exploration: big kicks early, fine polish late.
    let mut out = *p;
    let p_key = 0.1 + 0.25 * scale;
    for v in &mut out.keys {
        if rng.next_f64() < p_key {
            *v = (rng.next_u64() % 9) as u8;
        }
    }
    out.hold =
        (out.hold + rng.gauss() * (0.3 + 1.2 * scale)).clamp(2.0, 16.0);
    out.k_fb =
        (out.k_fb + rng.gauss() * (0.05 + 0.25 * scale)).clamp(0.0, 2.0);
    if rng.next_f64() < 0.1 {
        out.reset = !out.reset;
    }
    out
}

/// WR-style structural prior: a walking loop needs both cross families —
/// QP-family {Q,P} and WO-family {W,O} (same-leg QO/WP never walks).
/// Repair converts random keys until each family holds >= 2 slots.
pub fn enforce_beats(rng: &mut XorShift64, keys: &mut [u8; 12]) {
    let qp = |k: u8| k == 1 || k == 4 || k == 6;
    let wo = |k: u8| k == 2 || k == 3 || k == 7;
    for _ in 0..24 {
        let n_qp = keys.iter().filter(|k| qp(**k)).count();
        let n_wo = keys.iter().filter(|k| wo(**k)).count();
        if n_qp >= 2 && n_wo >= 2 {
            break;
        }
        let i = (rng.next_u64() as usize) % 12;
        if n_qp < 2 {
            keys[i] = [1, 4, 6][(rng.next_u64() as usize) % 3];
        } else {
            keys[i] = [2, 3, 7][(rng.next_u64() as usize) % 3];
        }
    }
}

pub const FB_EVAL_STEPS: usize = 1125; // 45 s at 25 Hz

pub fn train_fb(
    gens: usize,
    pop: usize,
    seed: u64,
    seed_loop: Option<KeyLoop>,
    seed_pol: Option<FbPolicy>,
    scrape_w: f64,
    alt_w: f64,
    alt_keep: f64,
    max_minutes: f64,
    force_beats: bool,
    patience: usize,
) -> FbResult {
    let t0 = std::time::Instant::now();
    let budget = std::time::Duration::from_secs_f64(max_minutes * 60.0);
    let mut rng = XorShift64::new(seed);
    let mut population: Vec<FbPolicy> = (0..pop)
        .map(|i| {
            let mut pol = if i < pop / 2 {
                if let Some(base_pol) = seed_pol {
                    mutate_fb(&mut rng, &base_pol, 1.0)
                } else if let Some(loop_) = seed_loop {
                    let base = FbPolicy {
                        keys: loop_,
                        hold: 6.0,
                        k_fb: 0.5,
                        reset: i % 2 == 0,
                    };
                    mutate_fb(&mut rng, &base, 1.0)
                } else {
                    random_fb(&mut rng)
                }
            } else {
                random_fb(&mut rng)
            };
            if force_beats {
                enforce_beats(&mut rng, &mut pol.keys);
            }
            pol
        })
        .collect();
    // Exact copy of the seed: delicate behaviors (alternation) must
    // survive gen 0 so selection can see them.
    if let Some(base_pol) = seed_pol {
        if !population.is_empty() {
            population[0] = base_pol;
        }
    }
    let mut history = Vec::new();
    let mut best_overall = population[0];
    let mut best_score_overall = f64::NEG_INFINITY;
    let mut best_dist_overall = 0.0;
    let mut stale = 0usize;

    for g in 0..gens {
        let mut scored: Vec<(FbPolicy, f64, f64, f64)> = population
            .iter()
            .map(|pol| {
                let (s, _, d, a) = rollout_fb(pol, FB_EVAL_STEPS, scrape_w, alt_w, alt_keep);
                (*pol, s, d, a)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let elite_n = (pop / 4).max(2).min(pop);
        if scored[0].1 > best_score_overall {
            best_score_overall = scored[0].1;
            best_overall = scored[0].0;
            best_dist_overall = scored[0].2;
            stale = 0;
        } else {
            stale += 1;
        }
        history.push((g, scored[0].1));
        if g % 10 == 0 || g + 1 == gens {
            eprintln!(
                "fb gen {g:4} best {:7.3} dist {:6.2} alt {:.2} overall {:7.3}",
                scored[0].1, scored[0].2, scored[0].3, best_score_overall
            );
        }
        let mut next = Vec::with_capacity(pop);
        for (pol, _, _, _) in scored.iter().take(elite_n) {
            next.push(*pol);
        }
        while next.len() < pop {
            let parent =
                scored[(rng.next_u64() as usize) % elite_n].0;
            let scale = 1.5 - 1.0 * (g as f64 / gens as f64);
            let mut child = mutate_fb(&mut rng, &parent, scale);
            if force_beats {
                enforce_beats(&mut rng, &mut child.keys);
            }
            next.push(child);
        }
        population = next;
        if t0.elapsed() >= budget {
            eprintln!(
                "fb: wall-clock budget reached at gen {g} ({} planned)",
                gens
            );
            break;
        }
        if patience > 0 && stale >= patience {
            eprintln!(
                "fb: plateaued for {stale} gens, stopping early at gen {g}"
            );
            break;
        }
    }

    FbResult {
        best: best_overall,
        best_score: best_score_overall,
        best_dist: best_dist_overall,
        history,
    }
}

/// Sample a v2 frame track under a feedback policy.
pub fn sample_fb(
    pol: &FbPolicy,
    seconds: f64,
) -> (Vec<[[f64; 3]; 12]>, f64, bool) {
    use evogait_sim_core::policy::WARM_STEPS;
    let mut sim = QwopSim::new();
    let mut runner = FbRunner::new(*pol);
    for _ in 0..WARM_STEPS {
        if sim.fallen() {
            break;
        }
        let a = runner.action(&sim);
        sim.step_action(a);
    }
    let x0 = sim.torso_x();
    let mut frames = Vec::new();
    let mut next_sample = 0.0;
    while sim.time() < seconds && !sim.fallen() {
        let a = runner.action(&sim);
        sim.step_action(a);
        if sim.time() >= next_sample {
            frames.push(sim.bodies_xyw());
            next_sample += 1.0 / 30.0;
        }
    }
    let dist = (sim.torso_x() - x0) / 10.0;
    (frames, dist, sim.fallen())
}

pub fn policy_json(pol: &FbPolicy) -> String {
    let keys = pol
        .keys
        .iter()
        .map(|k| k.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"keys\":[{keys}],\"hold\":{},\"k_fb\":{},\"reset\":{}}}",
        pol.hold, pol.k_fb, pol.reset
    )
}

pub fn gait_v2_json_fb(
    pol: &evogait_sim_core::policy::FbPolicy,
    frames: &[[[f64; 3]; 12]],
    distance: f64,
    fell: bool,
) -> String {
    let body_names = [
        "torso", "head", "leftArm", "leftForearm", "leftThigh", "leftCalf",
        "leftFoot", "rightArm", "rightForearm", "rightThigh", "rightCalf",
        "rightFoot",
    ];
    let names = body_names
        .iter()
        .map(|n| format!("\"{n}\""))
        .collect::<Vec<_>>()
        .join(",");
    let keys = pol
        .keys
        .iter()
        .map(|k| k.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let fr = frames
        .iter()
        .map(|f| {
            format!(
                "[{}]",
                f.iter().map(fmt_body).collect::<Vec<_>>().join(",")
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"format\":2,\"fps\":30,\"bodies\":[{names}],\"keys\":[{keys}],\"fb\":{{\"hold\":{:.3},\"k_fb\":{:.3},\"reset\":{}}},\"distance\":{distance:.3},\"fell\":{fell},\"frames\":[{fr}]}}",
        pol.hold, pol.k_fb, pol.reset
    )
}

fn fmt_body(b: &[f64; 3]) -> String {
    format!("[{:.3},{:.3},{:.3}]", b[0], b[1], b[2])
}

pub fn gait_v2_json(
    loop_: &KeyLoop,
    frames: &[[[f64; 3]; 12]],
    distance: f64,
    fell: bool,
) -> String {
    let body_names = [
        "torso", "head", "leftArm", "leftForearm", "leftThigh", "leftCalf",
        "leftFoot", "rightArm", "rightForearm", "rightThigh", "rightCalf",
        "rightFoot",
    ];
    let names = body_names
        .iter()
        .map(|n| format!("\"{n}\""))
        .collect::<Vec<_>>()
        .join(",");
    let keys = loop_
        .iter()
        .map(|k| k.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let fr = frames
        .iter()
        .map(|f| {
            format!(
                "[{}]",
                f.iter().map(fmt_body).collect::<Vec<_>>().join(",")
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"format\":2,\"fps\":30,\"bodies\":[{names}],\"keys\":[{keys}],\"distance\":{distance:.3},\"fell\":{fell},\"frames\":[{fr}]}}"
    )
}
