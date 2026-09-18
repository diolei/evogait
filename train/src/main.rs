use evogait_sim_core::{STEPS_10S, XorShift64, rollout_cpg};
use std::env;
use std::fs;

mod full;

fn parse_flag(args: &[String], name: &str, default: usize) -> usize {
    let mut i = 0;
    while i < args.len() {
        if args[i] == format!("--{name}") && i + 1 < args.len() {
            if let Ok(v) = args[i + 1].parse::<usize>() {
                return v;
            }
        }
        i += 1;
    }
    default
}

fn parse_float(args: &[String], name: &str, default: f64) -> f64 {
    let mut i = 0;
    while i < args.len() {
        if args[i] == name && i + 1 < args.len() {
            if let Ok(v) = args[i + 1].parse::<f64>() {
                return v;
            }
        }
        i += 1;
    }
    default
}

/// Throughput benchmark: fixed eval batch, prints steps/s for budget math.
fn main_bench() {
    use std::time::Instant;
    let t0 = Instant::now();
    let mut n = 0;
    for _ in 0..200 {
        let mut sim = evogait_sim_core::QwopSim::new();
        for i in 0..250 {
            sim.step_action((i * 7 + 3) % 9);
            n += 1;
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "bench: {n} eval-steps in {dt:.1}s = {:.0} steps/s",
        n as f64 / dt
    );
}

fn parse_out(args: &[String]) -> String {
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--out" && i + 1 < args.len() {
            return args[i + 1].clone();
        }
        i += 1;
    }
    "out".to_string()
}

fn gait_json(params: [f64; 4], frames: &[[f64; 4]], distance: f64) -> String {
    format!(
        "{{\"format\":1,\"fps\":30,\"joints\":[\"lh\",\"lk\",\"rh\",\"rk\"],\"params\":[{:.4},{:.4},{:.4},{:.4}],\"distance\":{:.3},\"frames\":[{}]}}",
        params[0],
        params[1],
        params[2],
        params[3],
        distance,
        frames
            .iter()
            .map(|q| format!("[{:.4},{:.4},{:.4},{:.4}]", q[0], q[1], q[2], q[3]))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn sample_gait(params: [f64; 4], seconds: f64) -> (Vec<[f64; 4]>, f64) {
    let cpg = evogait_sim_core::Cpg::from_vec(params);
    let mut sim = evogait_sim_core::Sim::new();
    let mut act = [0.0; 4];
    let mut frames: Vec<[f64; 4]> = Vec::new();
    let stride = 4; // 120Hz -> 30fps
    let mut i = 0;
    while sim.time() < seconds && !sim.fallen() {
        cpg.targets(sim.time(), &mut act);
        sim.step(act);
        if i % stride == 0 {
            frames.push(sim.joint_angles());
        }
        i += 1;
    }
    let dist = sim.torso_x();
    (frames, dist)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let model = parse_str(&args, "--model", "toy");
    if model == "full" {
        return main_full(&args);
    }
    main_toy(&args);
}

fn parse_str(args: &[String], name: &str, default: &str) -> String {
    let mut i = 0;
    while i < args.len() {
        if args[i] == name && i + 1 < args.len() {
            return args[i + 1].clone();
        }
        i += 1;
    }
    default.to_string()
}

/// Full-athlete key-loop evolution -> FORMAT v2 JSON.
fn main_full(args: &[String]) {
    let policy = parse_str(args, "--policy", "loop");
    if policy == "fb" {
        return main_fb(args);
    }
    let gens = parse_flag(args, "gens", 120);
    let pop = parse_flag(args, "pop", 24);
    let hold = parse_flag(args, "hold", 6);
    let out = parse_out(args);
    let seed = parse_flag(args, "seed", 0xE0A17);

    let res = full::train(gens, pop, seed as u64, hold);
    fs::create_dir_all(&out).expect("create out dir");
    let (frames, dist, fell) =
        full::sample_v2(&res.best, full::SAMPLE_SECONDS, hold);
    fs::write(
        format!("{out}/gait-best.json"),
        full::gait_v2_json(&res.best, &frames, dist, fell),
    )
    .expect("write gait");
    let (f_frames, f_dist) = full::sample_flail(full::SAMPLE_SECONDS);
    let flail_loop = [0u8; full::PERIOD];
    fs::write(
        format!("{out}/gait-early.json"),
        full::gait_v2_json(&flail_loop, &f_frames, f_dist, true),
    )
    .expect("write early gait");
    let hist = res
        .history
        .iter()
        .map(|(g, s)| format!("[{g},{s:.3}]"))
        .collect::<Vec<_>>()
        .join(",");
    fs::write(format!("{out}/reward.json"), format!("{{\"history\":[{hist}]}}"))
        .expect("write reward");
    println!(
        "full: best {:.3} (loop {:?}) -> {out}/gait-best.json + gait-early.json + reward.json",
        res.best_score, res.best
    );
}

/// Feedback-policy evolution -> policy-best.json + v2 gait + reward.
fn main_fb(args: &[String]) {
    if args.iter().any(|a| a == "--bench") {
        return main_bench();
    }
    // Provenance: which engine source produced this run. Best-effort;
    // a stale binary once produced numbers its own artifact couldn't
    // reproduce, and we never want to debug that blind again.
    let rev = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    eprintln!("evogait engine rev: {rev}");
    let gens = parse_flag(args, "gens", 80);
    let pop = parse_flag(args, "pop", 24);
    let out = parse_out(args);
    let seed = parse_flag(args, "seed", 0xFB01);
    let seed_loop = parse_loop(args);
    let seed_pol = parse_policy_file(args);
    let scrape_w = parse_float(args, "--scrape-w", 0.4);
    let alt_w = parse_float(args, "--alt-w", 2.0);
    let alt_keep = parse_float(args, "--alt-keep", 0.35);
    let max_minutes = parse_float(args, "--max-minutes", 25.0);
    let force_beats = args.iter().any(|a| a == "--force-beats");
    let patience = parse_flag(args, "patience", 0);

    let res = full::train_fb(
        gens,
        pop,
        seed as u64,
        seed_loop,
        seed_pol,
        scrape_w,
        alt_w,
        alt_keep,
        max_minutes,
        force_beats,
        patience,
    );
    fs::create_dir_all(&out).expect("create out dir");
    let (frames, dist, fell) =
        full::sample_fb(&res.best, full::SAMPLE_SECONDS);
    fs::write(
        format!("{out}/gait-best.json"),
        full::gait_v2_json_fb(&res.best, &frames, dist, fell),
    )
    .expect("write gait");
    let flail = evogait_sim_core::policy::FbPolicy::open_loop(
        [0u8; 12],
        6.0,
    );
    let (f_frames, f_dist, f_fell) =
        full::sample_fb(&flail, full::SAMPLE_SECONDS);
    fs::write(
        format!("{out}/gait-early.json"),
        full::gait_v2_json_fb(&flail, &f_frames, f_dist, f_fell),
    )
    .expect("write early gait");
    fs::write(
        format!("{out}/policy-best.json"),
        full::policy_json(&res.best),
    )
    .expect("write policy");
    let hist = res
        .history
        .iter()
        .map(|(g, s)| format!("[{g},{s:.3}]"))
        .collect::<Vec<_>>()
        .join(",");
    fs::write(format!("{out}/reward.json"), format!("{{\"history\":[{hist}]}}"))
        .expect("write reward");
    println!(
        "fb: best {:.3} dist {:.2} -> {out}/policy-best.json + gait JSONs + reward.json",
        res.best_score, res.best_dist
    );
}

/// Read a policy-best.json file for --seed-policy chaining.
fn parse_policy_file(
    args: &[String],
) -> Option<evogait_sim_core::policy::FbPolicy> {
    let path = parse_str(args, "--seed-policy", "");
    if path.is_empty() {
        return None;
    }
    let text = std::fs::read_to_string(&path).ok()?;
    // Minimal JSON scan: "keys":[..12 numbers..],"hold":x,"k_fb":y,"reset":b.
    let keys: Vec<u8> = text
        .split('[')
        .nth(1)?
        .split(']')
        .next()?
        .split(',')
        .filter_map(|t| t.trim().parse::<u8>().ok().map(|v| v.min(8)))
        .collect();
    if keys.len() != 12 {
        return None;
    }
    let mut arr = [0u8; 12];
    arr.copy_from_slice(&keys);
    let num_after = |key: &str| -> Option<f64> {
        text.split(key)
            .nth(1)?
            .split(|c| c == ',' || c == '}')
            .next()?
            .trim()
            .parse::<f64>()
            .ok()
    };
    let reset = text.contains("\"reset\":true");
    Some(evogait_sim_core::policy::FbPolicy {
        keys: arr,
        hold: num_after("\"hold\":").unwrap_or(6.0).clamp(2.0, 16.0),
        k_fb: num_after("\"k_fb\":").unwrap_or(0.5).clamp(0.0, 2.0),
        reset,
    })
}

fn parse_loop(args: &[String]) -> Option<[u8; 12]> {    let s = parse_str(args, "--seed-loop", "");
    if s.is_empty() {
        return None;
    }
    let vals: Vec<u8> = s
        .split(',')
        .filter_map(|t| t.trim().parse::<u8>().ok().map(|v| v.min(8)))
        .collect();
    if vals.len() != 12 {
        return None;
    }
    let mut out = [0u8; 12];
    out.copy_from_slice(&vals);
    Some(out)
}

fn main_toy(args: &[String]) {
    let gens = parse_flag(&args, "gens", 200);
    let pop = parse_flag(&args, "pop", 24);
    let out = parse_out(&args);

    let mut rng = XorShift64::new(0xE0A1_7);
    // Genome: [amp_hip, amp_knee, freq_hz, knee_offset].
    let mut mean = [0.45, 0.6, 1.0, -0.35];
    let mut sigma = 0.18;
    let steps = STEPS_10S / 2;
    let mut history: Vec<(usize, f64)> = Vec::new();

    for g in 0..gens {
        // Sample population around mean.
        let mut scored: Vec<([f64; 4], f64)> = Vec::with_capacity(pop);
        for _ in 0..pop {
            let mut p = [0.0; 4];
            for k in 0..4 {
                p[k] = mean[k] + rng.gauss() * sigma;
            }
            // Clamp to valid box.
            p[0] = p[0].clamp(0.0, 1.0);
            p[1] = p[1].clamp(0.0, 1.0);
            p[2] = p[2].clamp(0.4, 2.5);
            p[3] = p[3].clamp(-0.9, 0.1);
            let (dist, fell) = rollout_cpg(p, steps);
            let score = dist - if fell { 2.0 } else { 0.0 };
            scored.push((p, score));
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let elite = scored.len().max(1).min(6).max(pop / 4).min(scored.len());
        // Move mean toward elite average, decay sigma.
        let mut next = [0.0; 4];
        for (p, _) in scored.iter().take(elite) {
            for k in 0..4 {
                next[k] += p[k] / elite as f64;
            }
        }
        mean = next;
        sigma = (sigma * 0.985).max(0.02);
        let best = scored[0].1;
        history.push((g, best));
        if g % 20 == 0 || g + 1 == gens {
            eprintln!("gen {g:4} best {best:7.3} mean [{:.2},{:.2},{:.2},{:.2}]", mean[0], mean[1], mean[2], mean[3]);
        }
    }

    fs::create_dir_all(&out).expect("create out dir");
    // Gait sample at 30fps for 6s from best mean + untrained baseline.
    let (frames, dist) = sample_gait(mean, 6.0);
    fs::write(format!("{out}/gait-best.json"), gait_json(mean, &frames, dist))
        .expect("write gait");
    let flail = [0.0, 0.0, 1.0, -0.05];
    let (f_frames, f_dist) = sample_gait(flail, 6.0);
    fs::write(
        format!("{out}/gait-early.json"),
        gait_json(flail, &f_frames, f_dist),
    )
    .expect("write early gait");
    let hist = history
        .iter()
        .map(|(g, s)| format!("[{g},{s:.3}]"))
        .collect::<Vec<_>>()
        .join(",");
    fs::write(format!("{out}/reward.json"), format!("{{\"history\":[{hist}]}}"))
        .expect("write reward");
    println!("wrote {out}/gait-best.json + {out}/gait-early.json + {out}/reward.json");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_file_round_trips_losslessly() {
        // Regression test: checkpoint files must reload bit-identically.
        // Truncated floats once cost us a phantom 0.23 score gap, and a
        // stale hold clamp silently rewrote genomes on load.
        let pol = evogait_sim_core::policy::FbPolicy {
            keys: [7, 6, 4, 7, 8, 7, 1, 1, 0, 7, 4, 2],
            hold: 7.617423198,
            k_fb: 1.6931407,
            reset: true,
        };
        let json = full::policy_json(&pol);
        let dir = std::env::temp_dir().join("evogait-roundtrip-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("policy-best.json");
        std::fs::write(&path, &json).unwrap();
        let args = vec![
            "train".to_string(),
            "--seed-policy".to_string(),
            path.to_string_lossy().to_string(),
        ];
        let back = parse_policy_file(&args).expect("parse");
        assert_eq!(back.keys, pol.keys);
        assert_eq!(back.hold.to_bits(), pol.hold.to_bits());
        assert_eq!(back.k_fb.to_bits(), pol.k_fb.to_bits());
        assert_eq!(back.reset, pol.reset);
        // Wide holds survive the round trip (old clamp silently cut 16→12).
        let wide = evogait_sim_core::policy::FbPolicy { hold: 15.2, ..pol };
        let json2 = full::policy_json(&wide);
        std::fs::write(&path, &json2).unwrap();
        let back2 = parse_policy_file(&args).expect("parse wide");
        assert_eq!(back2.hold.to_bits(), 15.2f64.to_bits());
    }
}
