//! Full QWOP athlete: 12 rigid bodies + 11 revolute joints + track.
//!
//! Port of the `bicrick/qwop-python` Box2D plan (which itself mirrors
//! `smanolloff/qwop-gym` and the Flash original). Coordinates are kept
//! verbatim: +x right, +y DOWN, gravity +10, meters, 25 Hz physics.
//! Zero dependencies, deterministic, WASM-safe.

pub const GRAVITY: f64 = 10.0;
pub const STEP_DT: f64 = 0.04;
pub const SUBSTEPS: usize = 2;
/// Velocity-solver sweeps per substep (motors + friction + joints,
/// solved together like Box2D — never 30 rigid projection passes).
pub const VEL_ITERS: usize = 8;
/// Penetration slop: positional correction tolerates this much overlap.
pub const SLOP: f64 = 0.005;
/// Positional correction factor (gentle — momentum lives in velocities).
pub const CORR_BETA: f64 = 0.2;

pub const N_BODIES: usize = 12;
pub const BODY_ORDER: [&str; N_BODIES] = [
    "torso",
    "head",
    "leftArm",
    "leftForearm",
    "leftThigh",
    "leftCalf",
    "leftFoot",
    "rightArm",
    "rightForearm",
    "rightThigh",
    "rightCalf",
    "rightFoot",
];

/// Track top surface height (segment center 10.74275, half-height 1.6).
pub const TRACK_TOP: f64 = 10.74275 - 1.6;
pub const TRACK_FRICTION: f64 = 0.2;

#[derive(Clone, Copy, Debug)]
pub struct BodyDef {
    pub pos: [f64; 2],
    pub angle: f64,
    pub hw: f64,
    pub hh: f64,
    pub friction: f64,
    pub density: f64,
}

/// Spawn table, verbatim from qwop-python `data.py`.
pub const BODIES: [BodyDef; N_BODIES] = [
    BodyDef { pos: [2.5111726226000157, -1.8709517533957938], angle: -1.2514497119301329, hw: 3.275, hh: 1.425, friction: 0.2, density: 1.0 }, // torso
    BodyDef { pos: [3.888130278719558, -5.621802929095265], angle: 0.06448415835225099, hw: 1.075, hh: 1.325, friction: 0.2, density: 1.0 }, // head
    BodyDef { pos: [4.417861014480877, -2.806563606410589], angle: 0.9040095895272826, hw: 1.85, hh: 0.625, friction: 0.2, density: 1.0 }, // leftArm
    BodyDef { pos: [5.830008603424893, -2.8733539631159584], angle: -1.2049772618421237, hw: 1.75, hh: 0.55, friction: 0.2, density: 1.0 }, // leftForearm
    BodyDef { pos: [2.5648987628203876, 1.648090668682522], angle: -2.0177234426823394, hw: 2.525, hh: 1.0, friction: 0.2, density: 1.0 }, // leftThigh
    BodyDef { pos: [3.12585731974087, 5.525511655361298], angle: -1.5903971528225265, hw: 2.5, hh: 0.75, friction: 0.2, density: 1.0 }, // leftCalf
    BodyDef { pos: [3.926921842806667, 8.08884032049622], angle: 0.12027524643408766, hw: 1.35, hh: 0.675, friction: 1.5, density: 3.0 }, // leftFoot
    BodyDef { pos: [1.1812303663272852, -3.5000256518601014], angle: -0.5222217404634386, hw: 1.95, hh: 0.75, friction: 0.2, density: 1.0 }, // rightArm
    BodyDef { pos: [0.4078206420797428, -1.0599953233084172], angle: -1.7553358283857299, hw: 2.225, hh: 0.675, friction: 0.2, density: 1.0 }, // rightForearm
    BodyDef { pos: [1.6120186135678773, 2.0615320561881516], angle: 1.4849422964528027, hw: 2.65, hh: 1.0, friction: 0.2, density: 1.0 }, // rightThigh
    BodyDef { pos: [-0.07253905736790486, 5.347881871063159], angle: -0.7588859967104447, hw: 2.5, hh: 0.75, friction: 0.2, density: 1.0 }, // rightCalf
    BodyDef { pos: [-1.1254742643908706, 7.567193169625567], angle: 0.5897605418219602, hw: 1.35, hh: 0.725, friction: 1.5, density: 3.0 }, // rightFoot
];

#[derive(Clone, Copy, Debug)]
pub struct JointDef {
    pub a: usize,
    pub b: usize,
    pub anchor: [f64; 2],
    pub low: f64,
    pub high: f64,
    /// Max motor torque; 0 = passive.
    pub max_torque: f64,
}

/// Joint table, verbatim anchors/limits. Order: neck, shoulders, hips,
/// elbows, knees, ankles.
pub const JOINTS: [JointDef; 11] = [
    JointDef { a: 1, b: 0, anchor: [3.5885141908253755, -4.526224223627244], low: -0.5, high: 0.0, max_torque: 0.0 }, // neck
    JointDef { a: 7, b: 0, anchor: [2.228476821818547, -4.086468732185028], low: -0.5, high: 1.5, max_torque: 1000.0 }, // rightShoulder
    JointDef { a: 2, b: 0, anchor: [3.6241979856895377, -3.5334881618011442], low: -2.0, high: 0.0, max_torque: 1000.0 }, // leftShoulder
    JointDef { a: 4, b: 0, anchor: [2.0030339754142847, 0.23737160622781284], low: -1.5, high: 0.5, max_torque: 6000.0 }, // leftHip
    JointDef { a: 9, b: 0, anchor: [1.2475900729227194, -0.011046642863645761], low: -1.3, high: 0.7, max_torque: 6000.0 }, // rightHip
    JointDef { a: 3, b: 2, anchor: [5.52537533, -1.63856204], low: -0.1, high: 0.5, max_torque: 0.0 }, // leftElbow
    JointDef { a: 8, b: 7, anchor: [-0.00609085, -2.80047588], low: -0.1, high: 0.5, max_torque: 0.0 }, // rightElbow
    JointDef { a: 5, b: 4, anchor: [3.384323411985692, 3.5168931240916876], low: -1.6, high: 0.0, max_torque: 3000.0 }, // leftKnee
    JointDef { a: 10, b: 9, anchor: [1.4982369235492752, 4.175600306005656], low: -1.3, high: 0.3, max_torque: 3000.0 }, // rightKnee
    JointDef { a: 6, b: 5, anchor: [3.31232250, 7.94770485], low: -0.5, high: 0.5, max_torque: 0.0 }, // leftAnkle
    JointDef { a: 11, b: 10, anchor: [-1.6562855402197227, 6.961551452557676], low: -0.5, high: 0.5, max_torque: 0.0 }, // rightAnkle
];

/// Discrete-9 reduced action set: index -> (q, w, o, p) held keys.
pub const ACTIONS_9: [[bool; 4]; 9] = [
    [false, false, false, false], // 0 none
    [true, false, false, false],  // 1 Q
    [false, true, false, false],  // 2 W
    [false, false, true, false],  // 3 O
    [false, false, false, true],  // 4 P
    [true, true, false, false],   // 5 QW
    [true, false, false, true],   // 6 QP
    [false, true, true, false],   // 7 WO
    [false, false, true, true],   // 8 OP
];

/// Motor speeds + hip limit overrides for a key state.
/// Returns (hip_speed, knee_speed, shoulder_speed, hip_limits).
fn motor_targets(
    keys: [bool; 4],
) -> (f64, f64, f64, ([f64; 2], [f64; 2])) {
    let [q, w, o, p] = keys;
    let hip = if q && !w {
        2.5
    } else if w && !q {
        -2.5
    } else {
        0.0
    };
    let knee = if o && !p {
        2.5
    } else if p && !o {
        -2.5
    } else {
        0.0
    };
    let shoulder = if q && !w {
        -2.0
    } else if w && !q {
        2.0
    } else {
        0.0
    };
    // O/P reshape hip ranges (qwop-python controls).
    let limits = if o && !p {
        ([-1.0, 1.0], [-1.3, 0.7])
    } else if p && !o {
        ([-1.5, 0.5], [-0.8, 1.2])
    } else {
        ([-1.5, 0.5], [-1.3, 0.7])
    };
    (hip, knee, shoulder, limits)
}

#[derive(Clone, Debug)]
struct Body {
    pos: [f64; 2],
    angle: f64,
    vel: [f64; 2],
    angvel: f64,
    inv_mass: f64,
    inv_inertia: f64,
    hw: f64,
    hh: f64,
    friction: f64,
}

impl Body {
    fn from_def(d: &BodyDef) -> Self {
        let m = d.density * (2.0 * d.hw) * (2.0 * d.hh);
        let i = m * ((2.0 * d.hw).powi(2) + (2.0 * d.hh).powi(2)) / 12.0;
        Self {
            pos: d.pos,
            angle: d.angle,
            vel: [0.0, 0.0],
            angvel: 0.0,
            inv_mass: 1.0 / m,
            inv_inertia: 1.0 / i,
            hw: d.hw,
            hh: d.hh,
            friction: d.friction,
        }
    }

    fn to_world(&self, local: [f64; 2]) -> [f64; 2] {
        let (s, c) = (self.angle.sin(), self.angle.cos());
        [
            self.pos[0] + local[0] * c - local[1] * s,
            self.pos[1] + local[0] * s + local[1] * c,
        ]
    }

    fn to_local(&self, world: [f64; 2]) -> [f64; 2] {
        let dx = world[0] - self.pos[0];
        let dy = world[1] - self.pos[1];
        let (s, c) = (self.angle.sin(), self.angle.cos());
        [dx * c + dy * s, -dx * s + dy * c]
    }

    fn corners(&self) -> [[f64; 2]; 4] {
        [
            self.to_world([-self.hw, -self.hh]),
            self.to_world([self.hw, -self.hh]),
            self.to_world([self.hw, self.hh]),
            self.to_world([-self.hw, self.hh]),
        ]
    }
}

/// Full athlete simulation.
#[derive(Clone, Debug)]
pub struct QwopSim {
    bodies: Vec<Body>,
    /// Reference relative angles captured at spawn (joint angle = rel - ref).
    joint_ref: Vec<f64>,
    /// Spawn-time local anchors per joint (must be fixed, not recomputed).
    joint_local: Vec<[[f64; 2]; 2]>,
    time: f64,
    score_time: f64,
    fallen: bool,
    keys: [bool; 4],
}

impl QwopSim {
    pub fn new() -> Self {
        let mut s = Self {
            bodies: BODIES.iter().map(Body::from_def).collect(),
            joint_ref: vec![0.0; JOINTS.len()],
            joint_local: vec![[[0.0; 2]; 2]; JOINTS.len()],
            time: 0.0,
            score_time: 0.0,
            fallen: false,
            keys: [false; 4],
        };
        for (i, j) in JOINTS.iter().enumerate() {
            s.joint_ref[i] =
                s.bodies[j.b].angle - s.bodies[j.a].angle;
            s.joint_local[i] = [
                s.bodies[j.a].to_local(j.anchor),
                s.bodies[j.b].to_local(j.anchor),
            ];
        }
        s.settle();
        s
    }

    /// Keys-up settle like qwop-python: <=20 steps, then zero velocities.
    fn settle(&mut self) {
        self.keys = [false; 4];
        let mut slow_frames = 0;
        for _ in 0..20 {
            self.physics_step();
            let slow = self.bodies.iter().all(|b| {
                (b.vel[0].abs() < 0.45)
                    && (b.vel[1].abs() < 0.45)
                    && (b.angvel.abs() < 1.5)
            });
            if slow {
                slow_frames += 1;
                if slow_frames >= 2 {
                    break;
                }
            } else {
                slow_frames = 0;
            }
        }
        for b in &mut self.bodies {
            b.vel = [0.0, 0.0];
            b.angvel = 0.0;
        }
        self.time = 0.0;
        self.score_time = 0.0;
    }

    pub fn set_keys(&mut self, keys: [bool; 4]) {
        self.keys = keys;
    }

    /// One 40 ms physics step under a Discrete-9 action.
    pub fn step_action(&mut self, action: usize) {
        let keys = ACTIONS_9[action.min(8)];
        self.keys = keys;
        self.physics_step();
        self.score_time += 1.0 / 30.0;
        self.time += STEP_DT;
        self.check_fall();
    }

    /// Keys-up physics step (settle / flail baseline).
    pub fn step_idle(&mut self) {
        self.keys = [false; 4];
        self.physics_step();
        self.time += STEP_DT;
        self.check_fall();
    }

    fn physics_step(&mut self) {
        let dt = STEP_DT / SUBSTEPS as f64;
        let (hip_spd, knee_spd, shoulder_spd, (lh_lim, rh_lim)) =
            motor_targets(self.keys);
        for _ in 0..SUBSTEPS {
            // Forces.
            for b in &mut self.bodies {
                b.vel[1] += GRAVITY * dt;
            }
            // Contacts vs the track plane (deepest 2 corners per body).
            let contacts = self.gather_contacts();
            let mut acc_n = vec![0.0; contacts.len()];
            let mut acc_t = vec![0.0; contacts.len()];
            let mut acc_p = vec![[0.0; 2]; JOINTS.len()];
            let mut acc_l = vec![0.0; JOINTS.len()];
            let mut acc_m = vec![0.0; JOINTS.len()];
            for _ in 0..VEL_ITERS {
                self.solve_joints_vel(
                    dt,
                    hip_spd,
                    knee_spd,
                    shoulder_spd,
                    lh_lim,
                    rh_lim,
                    &mut acc_p,
                    &mut acc_l,
                    &mut acc_m,
                );
                Self::solve_contacts_vel(
                    &mut self.bodies,
                    &contacts,
                    &mut acc_n,
                    &mut acc_t,
                );
            }
            // Integrate positions (momentum preserved — no refresh).
            for b in &mut self.bodies {
                b.pos[0] += b.vel[0] * dt;
                b.pos[1] += b.vel[1] * dt;
                b.angle += b.angvel * dt;
            }
            // Gentle positional correction (slop-tolerant).
            self.correct_positions(lh_lim, rh_lim);
        }
        // Head stabilizing torque (qwop-python game loop).
        {
            let head_angle = self.bodies[1].angle;
            self.bodies[1].angvel += -4.0 * (head_angle + 0.2) * STEP_DT;
        }
    }

    /// Contact points for this substep: body, world point, pen, friction.
    fn gather_contacts(&self) -> Vec<(usize, [f64; 2], f64, f64)> {
        let mut out = Vec::new();
        for bi in 0..self.bodies.len() {
            let b = &self.bodies[bi];
            let mu = b.friction.max(TRACK_FRICTION);
            let mut deep: Vec<([f64; 2], f64)> = Vec::new();
            for c in b.corners() {
                let pen = c[1] - TRACK_TOP;
                if pen > 0.0 {
                    deep.push((c, pen));
                }
            }
            deep.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            for (c, pen) in deep.into_iter().take(2) {
                out.push((bi, c, pen, mu));
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_joints_vel(
        &mut self,
        dt: f64,
        hip_spd: f64,
        knee_spd: f64,
        shoulder_spd: f64,
        lh_lim: [f64; 2],
        rh_lim: [f64; 2],
        acc_p: &mut [[f64; 2]],
        acc_l: &mut [f64],
        acc_m: &mut [f64],
    ) {
        // (joint_idx, speed): Q/W push hips opposite, like QWOP.
        let motors: [(usize, f64); 6] = [
            (1, shoulder_spd),
            (2, -shoulder_spd),
            (3, -hip_spd),
            (4, hip_spd),
            (7, -knee_spd),
            (8, knee_spd),
        ];
        let mut motor_of = [0.0f64; 11];
        let mut has_motor = [false; 11];
        for (ji, target) in motors {
            motor_of[ji] = target;
            has_motor[ji] = true;
        }
        for ji in 0..JOINTS.len() {
            let j = &JOINTS[ji];
            let (mut low, mut high) = (j.low, j.high);
            if ji == 3 {
                (low, high) = (lh_lim[0], lh_lim[1]);
            }
            if ji == 4 {
                (low, high) = (rh_lim[0], rh_lim[1]);
            }
            let ia = self.bodies[j.a].inv_inertia;
            let ib = self.bodies[j.b].inv_inertia;
            let wa = self.bodies[j.a].inv_mass;
            let wb = self.bodies[j.b].inv_mass;

            // --- point-to-point (revolute), 2x2 effective mass ---
            let (la, lb) = (self.joint_local[ji][0], self.joint_local[ji][1]);
            let pa = self.bodies[j.a].to_world(la);
            let pb = self.bodies[j.b].to_world(lb);
            // Use midpoint anchor for balanced arms.
            let p = [(pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0];
            let ra = [p[0] - self.bodies[j.a].pos[0], p[1] - self.bodies[j.a].pos[1]];
            let rb = [p[0] - self.bodies[j.b].pos[0], p[1] - self.bodies[j.b].pos[1]];
            // K = M^-1 + [r×] I^-1 [r×]^T
            let k11 = wa + wb + ia * ra[1] * ra[1] + ib * rb[1] * rb[1];
            let k12 = -(ia * ra[0] * ra[1] + ib * rb[0] * rb[1]);
            let k22 = wa + wb + ia * ra[0] * ra[0] + ib * rb[0] * rb[0];
            let det = k11 * k22 - k12 * k12;
            if det > 1e-12 {
                let va = self.bodies[j.a].vel;
                let vb = self.bodies[j.b].vel;
                let waa = self.bodies[j.a].angvel;
                let wbb = self.bodies[j.b].angvel;
                // v + w × r, cross in 2D: w × r = w*[-ry, rx].
                let pva = [va[0] - waa * ra[1], va[1] + waa * ra[0]];
                let pvb = [vb[0] - wbb * rb[1], vb[1] + wbb * rb[0]];
                let cd = [pvb[0] - pva[0], pvb[1] - pva[1]];
                let lam = [
                    -(k22 * cd[0] - k12 * cd[1]) / det,
                    -(-k12 * cd[0] + k11 * cd[1]) / det,
                ];
                acc_p[ji][0] += lam[0];
                acc_p[ji][1] += lam[1];
                Self::apply_point_impulse(
                    &mut self.bodies, j.a, j.b, ra, rb, lam,
                );
            }

            // --- angular limit ---
            let ang = self.joint_angle(ji);
            let eff = ia + ib;
            if eff > 1e-12 {
                let rel_w =
                    self.bodies[j.b].angvel - self.bodies[j.a].angvel;
                if ang <= low {
                    let jimp = -rel_w / eff;
                    let old = acc_l[ji];
                    acc_l[ji] = (old + jimp).max(0.0);
                    let dj = acc_l[ji] - old;
                    self.bodies[j.a].angvel -= dj * ia;
                    self.bodies[j.b].angvel += dj * ib;
                } else if ang >= high {
                    let jimp = -rel_w / eff;
                    let old = acc_l[ji];
                    acc_l[ji] = (old + jimp).min(0.0);
                    let dj = acc_l[ji] - old;
                    self.bodies[j.a].angvel -= dj * ia;
                    self.bodies[j.b].angvel += dj * ib;
                } else {
                    acc_l[ji] = 0.0;
                }
            }

            // --- velocity motor (torque-clamped, brake at 0) ---
            if has_motor[ji] && j.max_torque > 0.0 && eff > 1e-12 {
                let target = motor_of[ji];
                let rel_w =
                    self.bodies[j.b].angvel - self.bodies[j.a].angvel;
                let jimp = (target - rel_w) / eff;
                let max_imp = j.max_torque * dt;
                let old = acc_m[ji];
                acc_m[ji] = (old + jimp).clamp(-max_imp, max_imp);
                let dj = acc_m[ji] - old;
                self.bodies[j.a].angvel -= dj * ia;
                self.bodies[j.b].angvel += dj * ib;
            }
        }
    }

    fn apply_point_impulse(
        bodies: &mut [Body],
        a: usize,
        b: usize,
        ra: [f64; 2],
        rb: [f64; 2],
        lam: [f64; 2],
    ) {
        let (wa, wb, ia, ib) = (
            bodies[a].inv_mass,
            bodies[b].inv_mass,
            bodies[a].inv_inertia,
            bodies[b].inv_inertia,
        );
        bodies[a].vel[0] -= lam[0] * wa;
        bodies[a].vel[1] -= lam[1] * wa;
        bodies[a].angvel -= ia * (ra[0] * lam[1] - ra[1] * lam[0]);
        bodies[b].vel[0] += lam[0] * wb;
        bodies[b].vel[1] += lam[1] * wb;
        bodies[b].angvel += ib * (rb[0] * lam[1] - rb[1] * lam[0]);
    }

    /// Velocity-level contacts: normal push + Coulomb friction that sticks.
    fn solve_contacts_vel(
        bodies: &mut [Body],
        contacts: &[(usize, [f64; 2], f64, f64)],
        acc_n: &mut [f64],
        acc_t: &mut [f64],
    ) {
        // Ground normal points up: y is down, so n = [0, -1].
        for (ci, (bi, p, _pen, mu)) in contacts.iter().enumerate() {
            let b = &bodies[*bi];
            let r = [p[0] - b.pos[0], p[1] - b.pos[1]];
            // Normal: vn = (v + w × r) · n ; r × n = -rx.
            let vn = (b.vel[0] - b.angvel * r[1]) * 0.0
                + (b.vel[1] + b.angvel * r[0]) * -1.0;
            let eff_n = b.inv_mass + b.inv_inertia * r[0] * r[0];
            if eff_n > 1e-12 {
                let jn = -vn / eff_n;
                let old = acc_n[ci];
                acc_n[ci] = (old + jn).max(0.0);
                let dj = acc_n[ci] - old;
                let bodies_b = &mut bodies[*bi];
                bodies_b.vel[1] += dj * -1.0 * bodies_b.inv_mass;
                bodies_b.angvel += bodies_b.inv_inertia * (r[0] * (dj * -1.0) - r[1] * 0.0);
            }
            // Friction along t = [1, 0]: vt = (v + w × r) · t; r × t = -ry.
            let b2 = &bodies[*bi];
            let vt = (b2.vel[0] - b2.angvel * r[1]) * 1.0
                + (b2.vel[1] + b2.angvel * r[0]) * 0.0;
            let eff_t = b2.inv_mass + b2.inv_inertia * r[1] * r[1];
            if eff_t > 1e-12 {
                let jt = -vt / eff_t;
                let max_f = mu * acc_n[ci];
                let old = acc_t[ci];
                acc_t[ci] = (old + jt).clamp(-max_f, max_f);
                let dj = acc_t[ci] - old;
                let bodies_c = &mut bodies[*bi];
                bodies_c.vel[0] += dj * bodies_c.inv_mass;
                bodies_c.angvel += bodies_c.inv_inertia * (r[0] * 0.0 - r[1] * dj);
            }
        }
    }

    /// Gentle positional correction with slop (momentum untouched).
    fn correct_positions(&mut self, lh_lim: [f64; 2], rh_lim: [f64; 2]) {
        let _ = (lh_lim, rh_lim);
        for ji in 0..JOINTS.len() {
            let j = &JOINTS[ji];
            let (la, lb) = (self.joint_local[ji][0], self.joint_local[ji][1]);
            let pa = self.bodies[j.a].to_world(la);
            let pb = self.bodies[j.b].to_world(lb);
            let err = [pb[0] - pa[0], pb[1] - pa[1]];
            let wa = self.bodies[j.a].inv_mass;
            let wb = self.bodies[j.b].inv_mass;
            let wsum = wa + wb;
            if wsum > 1e-12 {
                self.bodies[j.a].pos[0] += err[0] * CORR_BETA * wa / wsum;
                self.bodies[j.a].pos[1] += err[1] * CORR_BETA * wa / wsum;
                self.bodies[j.b].pos[0] -= err[0] * CORR_BETA * wb / wsum;
                self.bodies[j.b].pos[1] -= err[1] * CORR_BETA * wb / wsum;
            }
        }
        for bi in 0..self.bodies.len() {
            let w = self.bodies[bi].inv_mass;
            if w <= 0.0 {
                continue;
            }
            let corners = self.bodies[bi].corners();
            for c in corners {
                let pen = c[1] - TRACK_TOP;
                if pen > SLOP {
                    self.bodies[bi].pos[1] -= (pen - SLOP) * CORR_BETA;
                }
            }
        }
    }

    fn joint_angle(&self, ji: usize) -> f64 {
        let j = &JOINTS[ji];
        (self.bodies[j.b].angle - self.bodies[j.a].angle) - self.joint_ref[ji]
    }


    fn check_fall(&mut self) {
        // Head (1), arms (2,7), forearms (3,8) touching track = fallen.
        for &bi in &[1usize, 2, 3, 7, 8] {
            for c in self.bodies[bi].corners() {
                if c[1] >= TRACK_TOP - 0.05 {
                    self.fallen = true;
                    return;
                }
            }
        }
    }

    pub fn fallen(&self) -> bool {
        self.fallen
    }

    pub fn torso_x(&self) -> f64 {
        self.bodies[0].pos[0]
    }

    pub fn torso_y(&self) -> f64 {
        self.bodies[0].pos[1]
    }

    pub fn torso_vx(&self) -> f64 {
        self.bodies[0].vel[0]
    }

    /// Impart horizontal velocity to the torso (trip-recovery probes).
    pub fn nudge_torso(&mut self, vx: f64) {
        self.bodies[0].vel[0] += vx;
    }

    /// Foot contact bits (left, right): any sole corner at the track.
    /// Used by feedback policies for touchdown phase reset.
    pub fn foot_contacts(&self) -> [bool; 2] {
        let mut out = [false; 2];
        for (slot, bi) in [6usize, 11].iter().enumerate() {
            for c in self.bodies[*bi].corners() {
                if c[1] >= TRACK_TOP - 0.05 {
                    out[slot] = true;
                    break;
                }
            }
        }
        out
    }

    /// Mean knee flexion in 0..1 (0 = straight at upper limit).
    /// Used by the anti-scrape reward term.
    pub fn knee_flexion(&self) -> f64 {
        let mut sum = 0.0;
        for ji in [7usize, 8] {
            let j = &JOINTS[ji];
            let span = (j.high - j.low).max(1e-6);
            let ang = self.joint_angle(ji);
            sum += ((j.high - ang) / span).clamp(0.0, 1.0);
        }
        sum / 2.0
    }

    /// Torso height above the track (game units, y grows downward).
    /// Standing is ~9.7; deep knee positions sink toward ~5.
    pub fn torso_height(&self) -> f64 {
        TRACK_TOP - self.bodies[0].pos[1]
    }

    /// Score in metres, like qwop-python: round(torso.x)/10.
    pub fn score(&self) -> f64 {
        (self.bodies[0].pos[0]).round() / 10.0
    }

    pub fn time(&self) -> f64 {
        self.time
    }

    /// 60-float observation: 12 bodies x (x, y, angle, vx, vy), normalized.
    pub fn obs(&self) -> [f32; 60] {
        // Order matches qwop-python observations.py.
        const ORDER: [usize; 12] = [0, 1, 2, 5, 6, 3, 4, 7, 10, 11, 8, 9];
        let ranges: [(f64, f64); 5] = [
            (-10.0, 1050.0),
            (-10.0, 10.0),
            (-6.0, 6.0),
            (-20.0, 60.0),
            (-25.0, 60.0),
        ];
        let mut out = [0.0f32; 60];
        for (slot, &bi) in ORDER.iter().enumerate() {
            let b = &self.bodies[bi];
            let vals = [b.pos[0], b.pos[1], b.angle, b.vel[0], b.vel[1]];
            for k in 0..5 {
                let (lo, hi) = ranges[k];
                let center = (lo + hi) / 2.0;
                let maxdev = (hi - lo) / 2.0;
                out[slot * 5 + k] =
                    ((vals[k] - center) / maxdev).clamp(-1.0, 1.0) as f32;
            }
        }
        out
    }

    /// Body state for rendering/export: 12 x (x, y, angle).
    pub fn bodies_xyw(&self) -> [[f64; 3]; N_BODIES] {
        let mut out = [[0.0; 3]; N_BODIES];
        for i in 0..N_BODIES {
            out[i] = [
                self.bodies[i].pos[0],
                self.bodies[i].pos[1],
                self.bodies[i].angle,
            ];
        }
        out
    }

    pub fn body_hw_hh(&self, i: usize) -> [f64; 2] {
        [self.bodies[i].hw, self.bodies[i].hh]
    }
}

impl Default for QwopSim {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_matches_table() {
        // Spawn defs are verbatim; sim starts from them (settle may sag).
        assert_eq!(BODIES.len(), 12);
        assert!((BODIES[0].pos[0] - 2.5111726226000157).abs() < 1e-9);
        assert!((BODIES[0].hw - 3.275).abs() < 1e-12);
        assert!((BODIES[6].friction - 1.5).abs() < 1e-12);
        assert!((BODIES[6].hw - 1.35).abs() < 1e-12);
        let s = QwopSim::new();
        assert_eq!(s.bodies.len(), 12);
    }

    #[test]
    fn settle_zeroes_velocities() {
        let s = QwopSim::new();
        for b in &s.bodies {
            assert!(b.vel[0].abs() < 1e-9);
            assert!(b.vel[1].abs() < 1e-9);
            assert!(b.angvel.abs() < 1e-9);
        }
        assert!(!s.fallen());
    }

    #[test]
    fn rollout_is_deterministic() {
        let seq: Vec<usize> = (0..120).map(|i| (i * 7 + 3) % 9).collect();
        let mut a = QwopSim::new();
        let mut b = QwopSim::new();
        for act in &seq {
            a.step_action(*act);
            b.step_action(*act);
        }
        assert!((a.torso_x() - b.torso_x()).abs() < 1e-9);
        assert_eq!(a.obs(), b.obs());
        assert_eq!(a.fallen(), b.fallen());
    }

    #[test]
    fn idle_stays_upright_briefly() {
        let mut s = QwopSim::new();
        for _ in 0..25 {
            s.step_idle();
        }
        assert!(s.torso_x().is_finite());
        // Spawn is a standing pose; 1 s keys-up should not face-plant.
        assert!(!s.fallen(), "standing spawn should survive 1 s idle");
    }

    #[test]
    fn q_motor_moves_hips() {
        let mut keys_up = QwopSim::new();
        let mut q = QwopSim::new();
        for _ in 0..10 {
            keys_up.step_idle();
            q.step_action(1); // Q
        }
        let ang_up = keys_up.joint_angle(4);
        let ang_q = q.joint_angle(4);
        assert!(
            (ang_q - ang_up).abs() > 1e-4,
            "Q should drive rightHip, Δ={}",
            ang_q - ang_up
        );
    }

    #[test]
    fn obs_stays_in_range() {
        let mut s = QwopSim::new();
        for i in 0..60 {
            s.step_action(i % 9);
            for v in s.obs() {
                assert!(v >= -1.0 && v <= 1.0 && v.is_finite());
            }
        }
    }

    #[test]
    fn scripted_lunge_translates() {
        // Viability probe 1 — propulsion: a WOO lunge cycle must carry
        // the torso forward (~1 m) before falling. Falling forward is
        // authentic QWOP; twitching in place is not.
        let mut s = QwopSim::new();
        let x0 = s.torso_x();
        let script = [2usize, 3, 3];
        for i in 0..500 {
            if s.fallen() {
                break;
            }
            s.step_action(script[(i / 12) % script.len()]);
        }
        let travel = (s.torso_x() - x0) / 10.0;
        assert!(
            travel > 0.8,
            "lunge cycle should translate >0.8 m, got {travel:.3} m"
        );
    }

    #[test]
    fn gentle_cycle_stays_upright() {
        // Viability probe 2 — balance: a gentle QWO cycle stays up.
        let mut s = QwopSim::new();
        let x0 = s.torso_x();
        let script = [1usize, 2, 3];
        for i in 0..500 {
            if s.fallen() {
                break;
            }
            s.step_action(script[(i / 12) % script.len()]);
        }
        let travel = (s.torso_x() - x0) / 10.0;
        assert!(!s.fallen(), "gentle cycle should stay upright");
        assert!(travel > 0.2, "should drift forward, got {travel:.3} m");
    }

    #[test]
    fn joints_hold_anchors_after_rollout() {        let mut s = QwopSim::new();
        for i in 0..50 {
            s.step_action((i * 3 + 1) % 9);
        }
        for ji in 0..JOINTS.len() {
            let j = &JOINTS[ji];
            let (la, lb) = (s.joint_local[ji][0], s.joint_local[ji][1]);
            let pa = s.bodies[j.a].to_world(la);
            let pb = s.bodies[j.b].to_world(lb);
            let d = ((pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2)).sqrt();
            assert!(d < 0.6, "joint {ji} drifted {d:.3} m");
        }
    }
}
