use evogait_sim_core::policy::{FbPolicy, FbRunner};
use evogait_sim_core::{N_BODIES, QwopSim};
use wasm_bindgen::prelude::*;

/// Browser stepping over the same 12-body core as native training.
/// Human QWOP mode drives this live; AI mode replays exported JSON.
#[wasm_bindgen]
pub struct Walker {
    sim: QwopSim,
}

#[wasm_bindgen]
impl Walker {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Walker {
        Walker {
            sim: QwopSim::new(),
        }
    }

    pub fn reset(&mut self) {
        self.sim = QwopSim::new();
    }

    /// Step with a Discrete-9 action (0-8). Returns torso x.
    pub fn step_action(&mut self, action: usize) -> f64 {
        self.sim.step_action(action.min(8));
        self.sim.torso_x()
    }

    /// Keys-up step. Returns torso x.
    pub fn step_idle(&mut self) -> f64 {
        self.sim.step_idle();
        self.sim.torso_x()
    }

    pub fn torso_x(&self) -> f64 {
        self.sim.torso_x()
    }

    pub fn torso_y(&self) -> f64 {
        self.sim.torso_y()
    }

    pub fn fallen(&self) -> bool {
        self.sim.fallen()
    }

    /// Body count (always 12).
    pub fn body_count(&self) -> usize {
        N_BODIES
    }

    /// Write 12 x (x, y, angle) into `out` (length >= 36).
    pub fn bodies(&self, out: &mut [f64]) {
        let b = self.sim.bodies_xyw();
        let n = N_BODIES.min(out.len() / 3);
        for i in 0..n {
            out[i * 3] = b[i][0];
            out[i * 3 + 1] = b[i][1];
            out[i * 3 + 2] = b[i][2];
        }
    }

    /// Write 12 x (half_width, half_height) into `out` (length >= 24).
    pub fn extents(&self, out: &mut [f64]) {
        let n = N_BODIES.min(out.len() / 2);
        for i in 0..n {
            let e = self.sim.body_hw_hh(i);
            out[i * 2] = e[0];
            out[i * 2 + 1] = e[1];
        }
    }
}

/// Live feedback policy runner: the evolved AI steps the sim itself,
/// so the browser can never diverge from training.
#[wasm_bindgen]
pub struct WalkerPol {
    sim: QwopSim,
    runner: FbRunner,
    pol: FbPolicy,
}

#[wasm_bindgen]
impl WalkerPol {
    /// `keys`: 12 Discrete-9 actions; `hold`: steps per key;
    /// `k_fb`: feedback strength; `reset`: touchdown reset on/off.
    #[wasm_bindgen(constructor)]
    pub fn new(
        keys: &[u8],
        hold: f64,
        k_fb: f64,
        reset: bool,
    ) -> WalkerPol {
        let mut arr = [0u8; 12];
        for (i, k) in keys.iter().take(12).enumerate() {
            arr[i] = (*k).min(8);
        }
        let pol = FbPolicy {
            keys: arr,
            hold,
            k_fb,
            reset,
        };
        WalkerPol {
            sim: QwopSim::new(),
            runner: FbRunner::new(pol),
            pol,
        }
        .warmed()
    }

    /// Silent pre-roll: settle into the gait before the first frame,
    /// matching training's warm-start (no opening splits on screen).
    fn warmed(mut self) -> Self {
        for _ in 0..50 {
            if self.sim.fallen() {
                break;
            }
            let a = self.runner.action(&self.sim);
            self.sim.step_action(a);
        }
        self
    }

    pub fn reset(&mut self) {
        self.sim = QwopSim::new();
        self.runner = FbRunner::new(self.pol);
        for _ in 0..50 {
            if self.sim.fallen() {
                break;
            }
            let a = self.runner.action(&self.sim);
            self.sim.step_action(a);
        }
    }

    /// Step the policy once. Returns the Discrete-9 action taken.
    pub fn step(&mut self) -> usize {
        let a = self.runner.action(&self.sim);
        self.sim.step_action(a);
        a
    }

    pub fn last_action(&self) -> usize {
        self.runner.last_action()
    }

    pub fn torso_x(&self) -> f64 {
        self.sim.torso_x()
    }

    pub fn fallen(&self) -> bool {
        self.sim.fallen()
    }

    pub fn body_count(&self) -> usize {
        N_BODIES
    }

    pub fn bodies(&self, out: &mut [f64]) {
        let b = self.sim.bodies_xyw();
        let n = N_BODIES.min(out.len() / 3);
        for i in 0..n {
            out[i * 3] = b[i][0];
            out[i * 3 + 1] = b[i][1];
            out[i * 3 + 2] = b[i][2];
        }
    }

    pub fn extents(&self, out: &mut [f64]) {
        let n = N_BODIES.min(out.len() / 2);
        for i in 0..n {
            let e = self.sim.body_hw_hh(i);
            out[i * 2] = e[0];
            out[i * 2 + 1] = e[1];
        }
    }
}
