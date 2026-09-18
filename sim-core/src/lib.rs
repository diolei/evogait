//! evogait-sim-core: deterministic QWOP-inspired 2D biped, zero dependencies.
//!
//! Two models: the original 4-joint toy (`Sim` + `Cpg`, kept for debug)
//! and the full 12-body athlete (`qwop::QwopSim`) ported from the
//! `bicrick/qwop-python` Box2D plan.

pub mod policy;
pub mod qwop;

pub use policy::{FbPolicy, FbRunner, rollout_fb};
pub use qwop::{ACTIONS_9, BODY_ORDER, N_BODIES, QwopSim};

pub const DT: f64 = 1.0 / 120.0;
pub const N_JOINTS: usize = 4;
/// Joint order: [left-hip, left-knee, right-hip, right-knee].
pub const JOINT_NAMES: [&str; N_JOINTS] = ["lh", "lk", "rh", "rk"];

const GRAVITY: f64 = 9.81;
const TORSO_MASS: f64 = 10.0;
const HIP_OFFSET: f64 = 0.10;
const THIGH_LEN: f64 = 0.50;
const CALF_LEN: f64 = 0.50;
const NOMINAL_HIP_Y: f64 = 0.92;

const JOINT_KP: f64 = 90.0;
const JOINT_KD: f64 = 9.0;
const JOINT_MAX_TARGET: f64 = 1.0;

const GROUND_KP: f64 = 2200.0;
const GROUND_KD: f64 = 120.0;
const GROUND_MU: f64 = 0.9;
const PROPULSION_K: f64 = 14.0;

pub const FALL_Y: f64 = 0.45;
pub const STEPS_10S: usize = 1200;

/// Full simulator state. All `f64`, fixed `DT`, no randomness inside `step`.
#[derive(Clone, Debug)]
pub struct Sim {
    t: f64,
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    q: [f64; N_JOINTS],
    qd: [f64; N_JOINTS],
    fell: bool,
}

impl Sim {
    pub fn new() -> Self {
        let mut s = Self {
            t: 0.0,
            x: 0.0,
            y: NOMINAL_HIP_Y,
            vx: 0.0,
            vy: 0.0,
            q: [0.0; N_JOINTS],
            qd: [0.0; N_JOINTS],
            fell: false,
        };
        s.reset();
        s
    }

    pub fn reset(&mut self) {
        self.t = 0.0;
        self.x = 0.0;
        self.y = NOMINAL_HIP_Y;
        self.vx = 0.0;
        self.vy = 0.0;
        // Slight stagger so alternating CPGs break symmetry immediately.
        self.q = [0.08, -0.12, -0.08, -0.12];
        self.qd = [0.0; N_JOINTS];
        self.fell = false;
    }

    /// One fixed step. `action` joints targets in [-1, 1].
    pub fn step(&mut self, action: [f64; N_JOINTS]) {
        // Clamp targets.
        let mut tgt = [0.0; N_JOINTS];
        for i in 0..N_JOINTS {
            tgt[i] = action[i].clamp(-JOINT_MAX_TARGET, JOINT_MAX_TARGET);
        }
        // Joint PD (semi-implicit Euler).
        for i in 0..N_JOINTS {
            let qdd = JOINT_KP * (tgt[i] - self.q[i]) - JOINT_KD * self.qd[i];
            self.qd[i] += qdd * DT;
            // Clamp joint velocity for stability on weak hardware floats.
            self.qd[i] = self.qd[i].clamp(-20.0, 20.0);
            self.q[i] += self.qd[i] * DT;
            self.q[i] = self.q[i].clamp(-1.4, 1.4);
        }

        // Feet + ground reaction (spring-damper + Coulomb-ish friction +
        // swing-back propulsion while in stance).
        let mut fx = 0.0;
        let mut fy = 0.0;
        for leg in 0..2 {
            let hi = leg * 2;
            let ki = leg * 2 + 1;
            let side = if leg == 0 { -1.0 } else { 1.0 };
            let hip_x = self.x + side * HIP_OFFSET;
            let hip_y = self.y;
            let qh = self.q[hi];
            let qk = self.q[ki];
            let knee_x = hip_x + THIGH_LEN * qh.sin();
            let knee_y = hip_y - THIGH_LEN * qh.cos();
            let qa = qh + qk;
            let foot_x = knee_x + CALF_LEN * qa.sin();
            let foot_y = knee_y - CALF_LEN * qa.cos();
            // Foot velocity via finite difference of joint rates (analytic).
            let dqh = self.qd[hi];
            let dqa = self.qd[hi] + self.qd[ki];
            let foot_vx =
                THIGH_LEN * qh.cos() * dqh + CALF_LEN * qa.cos() * dqa + self.vx;
            let foot_vy =
                THIGH_LEN * qh.sin() * dqh + CALF_LEN * qa.sin() * dqa + self.vy;
            let _ = foot_x;
            if foot_y < 0.0 {
                let pen = -foot_y;
                let mut fn_ = GROUND_KP * pen - GROUND_KD * foot_vy;
                if fn_ < 0.0 {
                    fn_ = 0.0;
                }
                if fn_ > 3000.0 {
                    fn_ = 3000.0;
                }
                // Friction opposes slip.
                let mut fx_leg = -GROUND_MU * fn_ * (foot_vx * 5.0).tanh();
                // Propulsion: swinging the stance leg backward shoves forward.
                fx_leg += -PROPULSION_K * dqh * (fn_ / 800.0).min(1.5);
                fx += fx_leg;
                fy += fn_;
            }
        }

        // Torso ballistic + ground forces.
        let ax = fx / TORSO_MASS;
        let ay = -GRAVITY + fy / TORSO_MASS;
        self.vx += ax * DT;
        self.vy += ay * DT;
        // Mild air drag keeps velocities bounded.
        self.vx *= 1.0 - 0.02 * DT;
        self.vy *= 1.0 - 0.02 * DT;
        self.x += self.vx * DT;
        self.y += self.vy * DT;
        if self.y < 0.05 {
            self.y = 0.05;
            self.vy = 0.0;
        }
        self.t += DT;
        if self.y < FALL_Y {
            self.fell = true;
        }
    }

    pub fn torso_x(&self) -> f64 {
        self.x
    }
    pub fn torso_y(&self) -> f64 {
        self.y
    }
    pub fn time(&self) -> f64 {
        self.t
    }
    pub fn fallen(&self) -> bool {
        self.fell
    }
    pub fn joint_angles(&self) -> [f64; N_JOINTS] {
        self.q
    }

    /// Foot heights (left, right). Negative = penetration before resolve.
    pub fn foot_heights(&self) -> [f64; 2] {
        let mut out = [0.0; 2];
        for leg in 0..2 {
            let hi = leg * 2;
            let ki = leg * 2 + 1;
            let side = if leg == 0 { -1.0 } else { 1.0 };
            let hip_x = self.x + side * HIP_OFFSET;
            let hip_y = self.y;
            let _ = hip_x;
            let qh = self.q[hi];
            let qa = qh + self.q[ki];
            let knee_y = hip_y - THIGH_LEN * qh.cos();
            out[leg] = knee_y - CALF_LEN * qa.cos();
        }
        out
    }
}

impl Default for Sim {
    fn default() -> Self {
        Self::new()
    }
}

/// Tiny central-pattern-generator policy: alternating sine drives.
///
/// `params` = [amp_hip, amp_knee, freq_hz, knee_offset].
/// Output order matches `JOINT_NAMES`, right leg antiphase to left.
#[derive(Clone, Copy, Debug)]
pub struct Cpg {
    pub amp_hip: f64,
    pub amp_knee: f64,
    pub freq_hz: f64,
    pub knee_offset: f64,
}

impl Cpg {
    pub fn targets(&self, t: f64, out: &mut [f64; N_JOINTS]) {
        let w = 2.0 * std::f64::consts::PI * self.freq_hz;
        let l = (w * t).sin();
        let r = (w * t + std::f64::consts::PI).sin();
        out[0] = self.amp_hip * l;
        out[2] = self.amp_hip * r;
        out[1] = self.knee_offset + self.amp_knee * l.max(0.0);
        out[3] = self.knee_offset + self.amp_knee * r.max(0.0);
    }

    pub fn params_vec(&self) -> [f64; 4] {
        [self.amp_hip, self.amp_knee, self.freq_hz, self.knee_offset]
    }

    pub fn from_vec(p: [f64; 4]) -> Self {
        Self {
            amp_hip: p[0].clamp(0.0, 1.0),
            amp_knee: p[1].clamp(0.0, 1.0),
            freq_hz: p[2].clamp(0.4, 2.5),
            knee_offset: p[3].clamp(-0.9, 0.1),
        }
    }
}

/// Deterministic xorshift64 RNG (no dependencies, WASM-safe).
#[derive(Clone, Debug)]
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        let s = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        Self { state: s }
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    pub fn next_f64(&mut self) -> f64 {
        // 53-bit uniform in [0, 1).
        const DIV: f64 = (1u64 << 53) as f64;
        ((self.next_u64() >> 11) as f64) / DIV
    }
    /// Standard normal via Box-Muller.
    pub fn gauss(&mut self) -> f64 {
        let mut u1 = self.next_f64();
        let u2 = self.next_f64();
        if u1 < 1e-12 {
            u1 = 1e-12;
        }
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Roll out a CPG genome. Returns (distance_m, fell).
pub fn rollout_cpg(params: [f64; 4], steps: usize) -> (f64, bool) {
    let cpg = Cpg::from_vec(params);
    let mut sim = Sim::new();
    let mut act = [0.0; N_JOINTS];
    for _ in 0..steps {
        if sim.fallen() {
            break;
        }
        cpg.targets(sim.time(), &mut act);
        sim.step(act);
    }
    (sim.torso_x(), sim.fallen())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sim_is_deterministic_for_same_actions() {
        let acts: Vec<[f64; N_JOINTS]> = (0..300)
            .map(|i| {
                let p = (i as f64 * 0.05).sin() * 0.6;
                [p, -0.2, -p, -0.2]
            })
            .collect();
        let mut a = Sim::new();
        let mut b = Sim::new();
        for act in &acts {
            a.step(*act);
            b.step(*act);
        }
        assert!((a.torso_x() - b.torso_x()).abs() < 1e-12);
        assert!((a.torso_y() - b.torso_y()).abs() < 1e-12);
        assert_eq!(a.joint_angles(), b.joint_angles());
    }

    #[test]
    fn alternating_cpg_moves_forward_without_falling() {
        let params = [0.55, 0.7, 1.1, -0.35];
        let (dist, fell) = rollout_cpg(params, STEPS_10S);
        assert!(!fell, "sensible CPG should stay upright, y fell");
        assert!(
            dist > 1.0,
            "expected forward progress > 1m in 10s, got {dist:.3}"
        );
    }

    #[test]
    fn zero_drive_collapses_or_stalls() {
        let (dist, _) = rollout_cpg([0.0, 0.0, 1.0, -0.05], STEPS_10S);
        assert!(
            dist.abs() < 1.0,
            "passive doll should not walk, got {dist:.3}"
        );
    }

    #[test]
    fn rng_is_deterministic_per_seed() {
        let mut a = XorShift64::new(42);
        let mut b = XorShift64::new(42);
        for _ in 0..16 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        let mut c = XorShift64::new(7);
        let g: Vec<f64> = (0..200).map(|_| c.gauss()).collect();
        let mean = g.iter().sum::<f64>() / g.len() as f64;
        assert!(mean.abs() < 0.25, "gauss mean near 0, got {mean:.3}");
    }

    #[test]
    fn cpg_right_leg_is_antiphase_to_left() {
        let cpg = Cpg::from_vec([0.6, 0.5, 1.0, -0.3]);
        let mut a = [0.0; N_JOINTS];
        let mut b = [0.0; N_JOINTS];
        cpg.targets(0.0, &mut a);
        cpg.targets(0.5, &mut b);
        // Half period at 1Hz swaps legs.
        assert!((a[0] + b[0]).abs() < 1e-9);
        assert!((a[2] + b[2]).abs() < 1e-9);
    }
}
