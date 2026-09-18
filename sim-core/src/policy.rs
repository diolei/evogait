//! Feedback-modulated key policy: Discrete-9 keys (QWOP-authentic
//! vocabulary) whose *timing* reacts to the body.
//!
//! - Foot-contact phase reset re-entrains the loop on touchdown.
//! - Torso-over-feet overhang stretches drive keys when extended and
//!   hurries recovery keys when behind — a stumble reflex in ~4 numbers.
//!
//! Genome: 12 discrete keys + hold + k_fb + reset flag (~15 numbers).

use super::qwop::{ACTIONS_9, QwopSim};

pub const LOOP_LEN: usize = 12;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FbPolicy {
    pub keys: [u8; LOOP_LEN],
    /// Base hold per key, physics steps (2..12).
    pub hold: f64,
    /// Overhang modulation strength (0..2).
    pub k_fb: f64,
    /// Touchdown phase reset on/off.
    pub reset: bool,
}

impl FbPolicy {
    /// Open-loop equivalent: fixed timing, no feedback.
    pub fn open_loop(keys: [u8; LOOP_LEN], hold: f64) -> Self {
        Self {
            keys,
            hold: hold.clamp(2.0, 16.0),
            k_fb: 0.0,
            reset: false,
        }
    }
}

pub struct FbRunner {
    policy: FbPolicy,
    phase: f64,
    prev_touch: [bool; 2],
    last_action: usize,
    switches: usize,
    upright_steps: usize,
    steps: usize,
    // Lead-leg alternation tracking.
    lead: i8, // -1 left, +1 right, 0 undecided/neutral
    lead_run: usize,
    counted_lead: i8,
    lead_changes: usize,
    left_lead_steps: usize,
    right_lead_steps: usize,
    counted_steps: usize,
    spread_sum: f64,
    spread_n: usize,
}

/// Deadband for lead detection (game units): smaller separations don't
/// count as either leg leading.
pub const LEAD_DEADBAND: f64 = 0.15;
/// A new lead counts only if held this many steps (0.2 s): rejects jitter.
pub const LEAD_HOLD_STEPS: usize = 5;
/// Lead time only counts while the torso advances faster than this
/// (game units/s): rocking in place earns nothing.
pub const LEAD_MOVE_MIN: f64 = 1.0;

impl FbRunner {
    pub fn new(policy: FbPolicy) -> Self {
        Self {
            policy,
            phase: 0.0,
            prev_touch: [false; 2],
            last_action: 0,
            switches: 0,
            upright_steps: 0,
            steps: 0,
            lead: 0,
            lead_run: 0,
            counted_lead: 0,
            lead_changes: 0,
            left_lead_steps: 0,
            right_lead_steps: 0,
            counted_steps: 0,
            spread_sum: 0.0,
            spread_n: 0,
        }
    }

    pub fn last_action(&self) -> usize {
        self.last_action
    }
    pub fn switches(&self) -> usize {
        self.switches
    }
    pub fn upright_frac(&self) -> f64 {
        if self.steps == 0 {
            1.0
        } else {
            self.upright_steps as f64 / self.steps as f64
        }
    }

    /// Shared lead time 0..0.5 (0.5 = perfectly shared). Valid only with
    /// at least 2 debounced lead changes — sitting centered earns nothing.
    pub fn alternation(&self) -> f64 {
        if self.lead_changes < 2 || self.counted_steps == 0 {
            return 0.0;
        }
        let l = self.left_lead_steps as f64 / self.counted_steps as f64;
        let r = self.right_lead_steps as f64 / self.counted_steps as f64;
        l.min(r)
    }

    pub fn lead_changes(&self) -> usize {
        self.lead_changes
    }

    /// Mean foot separation (game units). Lunging gaits spread 10+;
    /// compact striding stays low.
    pub fn mean_spread(&self) -> f64 {
        if self.spread_n == 0 {
            0.0
        } else {
            self.spread_sum / self.spread_n as f64
        }
    }

    fn track_lead(&mut self, left_x: f64, right_x: f64, moving: bool) {
        let sep = left_x - right_x;
        let cur: i8 = if sep > LEAD_DEADBAND {
            -1
        } else if sep < -LEAD_DEADBAND {
            1
        } else {
            0
        };
        if cur == 0 || !moving {
            self.lead_run = 0;
            return;
        }
        if cur == self.lead {
            self.lead_run += 1;
        } else {
            self.lead = cur;
            self.lead_run = 1;
        }
        if self.lead_run == LEAD_HOLD_STEPS {
            // Confirm a held lead. A neutral gap back to the same lead
            // is not a change.
            if self.counted_steps > 0 && cur != self.counted_lead {
                self.lead_changes += 1;
            }
            self.counted_lead = cur;
        }
        if self.lead_run >= LEAD_HOLD_STEPS {
            self.counted_steps += 1;
            if cur < 0 {
                self.left_lead_steps += 1;
            } else {
                self.right_lead_steps += 1;
            }
        }
    }

    /// Current Discrete-9 action. Reads body state, mutates only runner.
    pub fn action(&mut self, sim: &QwopSim) -> usize {
        let b = sim.bodies_xyw();
        let torso_x = b[0][0];
        let feet_mid = (b[6][0] + b[11][0]) / 2.0;
        let overhang = ((torso_x - feet_mid) / 2.0).clamp(-1.0, 1.0);
        self.track_lead(b[6][0], b[11][0], sim.torso_vx() > LEAD_MOVE_MIN);
        self.spread_sum += (b[6][0] - b[11][0]).abs();
        self.spread_n += 1;

        let touch = sim.foot_contacts();
        if self.policy.reset {
            let touchdown = (touch[0] && !self.prev_touch[0])
                || (touch[1] && !self.prev_touch[1]);
            if touchdown {
                self.phase = 0.0;
            }
        }
        self.prev_touch = touch;

        let idx = ((self.phase + 1e-9).floor() as usize) % LOOP_LEN;
        let key = self.policy.keys[idx].min(8) as usize;
        let entry = ACTIONS_9[key];
        let drive = entry[0] || entry[1]; // thigh keys push

        let hold = self.policy.hold.clamp(2.0, 16.0);
        let k = self.policy.k_fb.clamp(0.0, 2.0);
        let mut rate = 1.0 / hold;
        if drive {
            rate /= 1.0 + k * overhang.max(0.0);
        } else {
            rate *= 1.0 + k * (-overhang).max(0.0);
        }
        self.phase += rate;

        if self.steps == 0 || key != self.last_action {
            if self.steps > 0 {
                self.switches += 1;
            }
            self.last_action = key;
        }
        self.steps += 1;
        if sim.torso_y() < 0.5 {
            self.upright_steps += 1;
        }
        key
    }
}

/// Anti-scrape shaping (ported from qwop-python's flex-gait wrapper):
/// penalize the combination of low torso + flexed knees, lightly —
/// distance stays the primary term. Calibrated for our game units
/// (standing torso height ~9.7, slump ~5-7).
pub const SCRAPE_W: f64 = 0.4;
pub const SCRAPE_H: f64 = 7.5;
pub const SCRAPE_FLEX_MIN: f64 = 0.45;

/// Mean scrape signal over a rollout (0 = clean).
pub fn scrape_signal(sim: &QwopSim) -> f64 {
    let flex = sim.knee_flexion();
    let h = sim.torso_height();
    if h < SCRAPE_H && flex >= SCRAPE_FLEX_MIN {
        (SCRAPE_H - h) * flex
    } else {
        0.0
    }
}

/// Roll out a feedback policy. Returns (score, fell, distance, alternation).
/// `alt_keep`: fraction of the positive term a non-alternating gait keeps
/// (0.35 soft, ~0.05 hard constraint).
/// `scrape_w` prices knee-dragging, `alt_w` prices shared lead time
/// (0 disables either term).
/// Silent pre-roll: policy steps before the clock starts, so the scored
/// window covers steady-state gait — never the standing-start splits.
/// Mirrors qwop-python's settle_spawn, but settles INTO the gait.
pub const WARM_STEPS: usize = 50;

pub fn rollout_fb(
    policy: &FbPolicy,
    steps: usize,
    scrape_w: f64,
    alt_w: f64,
    alt_keep: f64,
) -> (f64, bool, f64, f64) {
    let mut sim = QwopSim::new();
    let mut runner = FbRunner::new(*policy);
    // Warm-start: settle into the gait before scoring.
    for _ in 0..WARM_STEPS {
        if sim.fallen() {
            break;
        }
        let a = runner.action(&sim);
        sim.step_action(a);
    }
    let x0 = sim.torso_x();
    let mut scrape_sum = 0.0;
    let mut scrape_n = 0usize;
    for _ in 0..steps {
        if sim.fallen() {
            break;
        }
        let a = runner.action(&sim);
        sim.step_action(a);
        scrape_sum += scrape_signal(&sim);
        scrape_n += 1;
    }
    let dist = (sim.torso_x() - x0) / 10.0;
    let scrape = if scrape_n > 0 {
        scrape_sum / scrape_n as f64
    } else {
        0.0
    };
    // Compact-stride shaping: foot spread above SPREAD_OK prices
    // splits-style lunging; short quick steps score best.
    const SPREAD_OK: f64 = 4.0;
    const SPREAD_W: f64 = 0.1;
    let spread_pen =
        SPREAD_W * (runner.mean_spread() - SPREAD_OK).max(0.0);
    // Graduated alternation gate: shared lead time scales the whole
    // positive term from 35% (one-sided) to 100% (alt >= 0.3). The user
    // prefers visible lead-swapping over raw metres, so the pressure is
    // continuous — not just above a threshold the search can game.
    const ALT_FULL: f64 = 0.3;
    let gate = alt_keep
        + (1.0 - alt_keep) * (runner.alternation() / ALT_FULL).clamp(0.0, 1.0);
    let score = (dist * (1.0 + 0.5 * runner.upright_frac() + alt_w * runner.alternation())
        + 0.002 * runner.switches() as f64)
        * gate
        - scrape_w * scrape
        - spread_pen
        - if sim.fallen() { 2.0 } else { 0.0 };
    (score, sim.fallen(), dist, runner.alternation())
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOOP: [u8; LOOP_LEN] = [7, 5, 2, 7, 1, 1, 7, 7, 3, 1, 6, 6];

    #[test]
    fn open_loop_replays_key_sequence() {
        let mut sim = QwopSim::new();
        let mut r = FbRunner::new(FbPolicy::open_loop(LOOP, 6.0));
        let mut seen = Vec::new();
        for _ in 0..13 {
            seen.push(r.action(&sim));
            sim.step_idle();
        }
        // hold=6: first key repeats 6 times, then advances.
        assert!(seen[0..6].iter().all(|&a| a == LOOP[0] as usize));
        assert_eq!(seen[6], LOOP[1] as usize);
    }

    #[test]
    fn touchdown_reset_is_deterministic() {
        // Two identical feedback runners on identical sims agree exactly.
        let mk = || {
            let sim = QwopSim::new();
            let r = FbRunner::new(FbPolicy {
                keys: LOOP,
                hold: 6.0,
                k_fb: 0.8,
                reset: true,
            });
            (sim, r)
        };
        let (mut s1, mut r1) = mk();
        let (mut s2, mut r2) = mk();
        for _ in 0..60 {
            let a1 = r1.action(&s1);
            let a2 = r2.action(&s2);
            assert_eq!(a1, a2);
            s1.step_idle();
            s2.step_idle();
        }
        // First action is always the loop head.
        let (_, mut r3) = mk();
        let s3 = QwopSim::new();
        assert_eq!(r3.action(&s3), LOOP[0] as usize);
    }

    #[test]
    fn scrape_metric_separates_poses() {
        // Standing start: no scrape signal.
        let sim = QwopSim::new();
        assert_eq!(scrape_signal(&sim), 0.0);
        assert!(sim.torso_height() > SCRAPE_H);
        // The 6.6 m winner drags its rear knee: must accumulate signal.
        let dragger = FbPolicy {
            keys: [4, 3, 7, 2, 7, 2, 3, 7, 3, 3, 6, 6],
            hold: 4.0,
            k_fb: 0.0,
            reset: false,
        };
        let mut sim2 = QwopSim::new();
        let mut r = FbRunner::new(dragger);
        let mut sum = 0.0;
        let mut n = 0;
        for _ in 0..500 {
            if sim2.fallen() {
                break;
            }
            let a = r.action(&sim2);
            sim2.step_action(a);
            sum += scrape_signal(&sim2);
            n += 1;
        }
        assert!(
            sum / n as f64 > 0.3,
            "knee-drag gait should scrape, got {}",
            sum / n as f64
        );
    }

    #[test]
    fn alternation_metric_logic() {
        // Unit-test the metric with crafted lead traces (no sim needed).
        let feed = |r: &mut FbRunner, seq: &[(f64, f64, bool)]| {
            for (l, rr, m) in seq {
                r.track_lead(*l, *rr, *m);
            }
        };
        // Perfect sharing while moving: L x6, R x6, L x6, R x6.
        let mut r = FbRunner::new(FbPolicy::open_loop([0u8; 12], 6.0));
        let mut seq = Vec::new();
        for k in 0..4 {
            let (l, rr) = if k % 2 == 0 { (1.0, 0.0) } else { (0.0, 1.0) };
            for _ in 0..6 {
                seq.push((l, rr, true));
            }
        }
        feed(&mut r, &seq);
        assert!(r.lead_changes() >= 3, "changes={}", r.lead_changes());
        assert!(
            (r.alternation() - 0.5).abs() < 0.05,
            "alt={}",
            r.alternation()
        );
        // One-sided while moving: no alternation.
        let mut r2 = FbRunner::new(FbPolicy::open_loop([0u8; 12], 6.0));
        feed(&mut r2, &vec![(1.0, 0.0, true); 24]);
        assert_eq!(r2.alternation(), 0.0);
        // Sharing while still: rocking in place earns nothing.
        let mut r3 = FbRunner::new(FbPolicy::open_loop([0u8; 12], 6.0));
        let mut seq3 = Vec::new();
        for k in 0..4 {
            let (l, rr) = if k % 2 == 0 { (1.0, 0.0) } else { (0.0, 1.0) };
            for _ in 0..6 {
                seq3.push((l, rr, false));
            }
        }
        feed(&mut r3, &seq3);
        assert_eq!(r3.alternation(), 0.0);
        // Fast jitter while moving: never held 5 steps, no alternation.
        let mut r4 = FbRunner::new(FbPolicy::open_loop([0u8; 12], 6.0));
        let mut seq4 = Vec::new();
        for k in 0..24 {
            let (l, rr) = if k % 2 == 0 { (1.0, 0.0) } else { (0.0, 1.0) };
            seq4.push((l, rr, true));
        }
        feed(&mut r4, &seq4);
        assert_eq!(r4.alternation(), 0.0);
    }

    #[test]
    fn spread_metric_separates_styles() {
        // The lunging winner spreads its feet far wider than the
        // keys-up flail. Guards the metric wiring, not any threshold.
        let run = |pol: FbPolicy| -> f64 {
            let mut sim = QwopSim::new();
            let mut r = FbRunner::new(pol);
            for _ in 0..WARM_STEPS {
                if sim.fallen() {
                    break;
                }
                let a = r.action(&sim);
                sim.step_action(a);
            }
            for _ in 0..200 {
                if sim.fallen() {
                    break;
                }
                let a = r.action(&sim);
                sim.step_action(a);
            }
            r.mean_spread()
        };
        let lunge = run(FbPolicy {
            keys: [5, 4, 7, 2, 4, 0, 6, 3, 7, 3, 1, 6],
            hold: 6.0,
            k_fb: 1.7,
            reset: false,
        });
        let flail = run(FbPolicy::open_loop([0u8; 12], 6.0));
        assert!(
            lunge > flail + 2.0,
            "lunger should spread wider ({lunge:.2} vs {flail:.2})"
        );
        assert!(lunge > 4.0, "lunger spread documents wide style");
    }

    #[test]
    fn feedback_recovers_from_shove() {
        // Same base loop, same shove: feedback must survive at least as
        // long as the open-loop replay.
        let open = FbPolicy::open_loop(LOOP, 6.0);
        let fb = FbPolicy {
            keys: LOOP,
            hold: 6.0,
            k_fb: 0.8,
            reset: true,
        };
        let survive = |pol: &FbPolicy| -> usize {
            let mut sim = QwopSim::new();
            let mut r = FbRunner::new(*pol);
            for i in 0..400 {
                if sim.fallen() {
                    return i;
                }
                if i == 100 {
                    sim.nudge_torso(6.0);
                }
                let a = r.action(&sim);
                sim.step_action(a);
            }
            400
        };
        let a = survive(&open);
        let b = survive(&fb);
        assert!(
            b >= a,
            "feedback ({b}) should out-survive open loop ({a}) after a shove"
        );
    }
}
