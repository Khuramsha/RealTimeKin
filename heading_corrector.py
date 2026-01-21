import numpy as np

"""
This module implements a lightweight *heading/yaw* correction layer for OpenSenseRT-style
pipelines:

    IMU orientation estimate (q_meas) -> HeadingCorrector.apply() -> corrected quats (q_corr)
        -> OpenSim IK -> model-predicted sensor orientations (q_model)
        -> HeadingCorrector.update() uses IK residuals to slowly adjust yaw offsets

The core idea is to correct the *unobservable* drift axis when using accel+gyro only
(no magnetometer). In standard OpenSim coordinates, the global vertical axis is Y,
so "yaw" is treated as rotation about the Y axis.

"""

def parse_sensor_names(header_text):
    fields = [f for f in header_text.strip().split('\t') if f]
    if not fields:
        return []
    if fields[0].lower() == 'time':
        fields = fields[1:]
    return fields


# =============================================================================
# Quaternion utilities (w, x, y, z), scalar-first
# =============================================================================
def quat_normalize(q):
    """Return q normalized to unit length (fallback to identity if near-zero)."""
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quat_mul(q1, q2):
    """Quaternion multiplication q = q1 ⊗ q2 (apply q2 then q1).

    Both inputs must be [w, x, y, z]. Output is [w, x, y, z].
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_inv(q):
    """Quaternion inverse (for unit quats this is the conjugate)."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z]) / np.dot(q, q)


def quat_log(q):
    """Log map of a unit quaternion to a 3D rotation vector (axis*angle, radians).
    Ensures the shortest-arc representation by flipping sign if w < 0.
    """
    q = quat_normalize(q)
    w, x, y, z = q
    # Flip to ensure shortest rotation (q and -q represent same rotation).
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    v = np.array([x, y, z])
    v_norm = np.linalg.norm(v)
    w = np.clip(w, -1.0, 1.0)
    if v_norm < 1e-12:
        # Small-angle approximation: angle≈2||v||, axis≈v/||v||
        return 2.0 * v
    angle = 2.0 * np.arctan2(v_norm, w)
    axis = v / v_norm
    return axis * angle


# =============================================================================
# Rotation matrix -> quaternion
# =============================================================================
def quat_from_rot_matrix(R):
    """Convert a 3x3 rotation matrix to a unit quaternion [w, x, y, z]."""
    trace = np.trace(R)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return quat_normalize(np.array([w, x, y, z]))


# =============================================================================
# Yaw quaternion (about OpenSim vertical axis)
# =============================================================================
def quat_axis_angle(axis, angle):

    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / n
    half = 0.5 * float(angle)
    s = np.sin(half)
    return np.array([np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s])


def quat_yaw(angle, z_hat=np.array([0.0, 1.0, 0.0])):
    # Yaw rotation about the global vertical axis (OpenSim is typically Y-up).
    return quat_axis_angle(z_hat, angle)


# =============================================================================
# HeadingCorrector: closed-loop yaw correction using IK residuals
# =============================================================================
class HeadingCorrector:
    def __init__(
        self,
        sensor_names,
        model,
        rate,
        update_hz=10.0,
        pelvis_gain=0.02,
        leg_gain=0.08,
        yaw_limit=np.deg2rad(20.0),
        yaw_outlier=np.deg2rad(35.0),
        stance_gyro_thresh=np.deg2rad(40.0),
        residual_skip=np.deg2rad(90.0),
        use_bias=False,
    ):
        self.sensor_names = sensor_names
        self.sensor_indices = {name: idx for idx, name in enumerate(sensor_names)}

        # --- Timing / update rate ----------------------------------------------
        self.rate = rate
        self.update_period = 1.0 / update_hz
        self.last_update_time = -np.inf

        # --- Controller state --------------------------------------------------
        # yaw_offsets: slow heading correction, per group
        self.use_bias = use_bias
        self.yaw_offsets = {'P': 0.0, 'L': 0.0, 'R': 0.0}

        # yaw_rate_bias: optional yaw-rate bias estimate, per group (experimental)
        self.yaw_rate_bias = {'P': 0.0, 'L': 0.0, 'R': 0.0}

        # --- Controller parameters --------------------------------------------
        self.pelvis_gain = pelvis_gain
        self.leg_gain = leg_gain
        self.yaw_limit = yaw_limit
        self.yaw_outlier = yaw_outlier
        self.residual_skip = residual_skip
        self.stance_gyro_thresh = stance_gyro_thresh

        # OpenSim ground is typically Y-up; use this as the "yaw axis" for projection.
        self.z_hat = np.array([0.0, 1.0, 0.0])

        # --- Stance estimation -------------------------------------------------
        # We estimate stance using quaternion delta of the foot IMU (proxy for angular velocity).
        self.prev_quat = {}
        self.prev_time = {}
        self.stance_state = {'L': False, 'R': False}

        # --- Group definitions -------------------------------------------------
        # Groups are used to reduce the number of correction states:
        #   P: pelvis
        #   L: left leg chain (thigh/shank/foot)
        #   R: right leg chain (thigh/shank/foot)
        self.sensor_groups = {
            'P': ['pelvis_imu'],
            'L': ['femur_l_imu', 'tibia_l_imu', 'calcn_l_imu'],
            'R': ['femur_r_imu', 'tibia_r_imu', 'calcn_r_imu'],
        }

        # Map from sensor name -> (OpenSim body name, OpenSim frame/component name)
        # The frame name is optional; if None, q_bs defaults to identity.
        self.sensor_body_map = {
            'pelvis_imu': ('pelvis', 'pelvis_imu'),
            'femur_l_imu': ('femur_l', 'femur_l_imu'),
            'tibia_l_imu': ('tibia_l', 'tibia_l_imu'),
            'calcn_l_imu': ('calcn_l', 'calcn_l_imu'),
            'femur_r_imu': ('femur_r', 'femur_r_imu'),
            'tibia_r_imu': ('tibia_r', 'tibia_r_imu'),
            'calcn_r_imu': ('calcn_r', 'calcn_r_imu'),
        }
        self.sensor_frame_quat = self._cache_sensor_frames(model)


    # -------------------------------------------------------------------------
    # Frame caching: retrieve sensor frame rotation relative to its body segment
    # -------------------------------------------------------------------------
    def _cache_sensor_frames(self, model):
        """Cache sensor-frame orientation in parent (body) for each IMU frame in the calibrated model.
        We expect IMUPlacer to have created frames named like '<body>_imu' under each body.
        frame.getTransformInParent().R() is the rotation from the IMU frame to its parent body frame.
        
        Cache q_bs (body->sensor) rotations from the OpenSim model.

        This attempts to find each sensor frame as:
            /bodyset/{body_name}/{frame_name}

        If a frame isn't found (or frame_name is None), q_bs falls back to identity.
        
        """
        frame_quat = {}
        self.missing_frames = []
        for name, (body_name, frame_name) in self.sensor_body_map.items():
            frame_name = frame_name or name
            q_bs = np.array([1.0, 0.0, 0.0, 0.0])
            frame_path = f"/bodyset/{body_name}/{frame_name}"
            try:
                frame = model.getComponent(frame_path)
                rot = frame.getTransformInParent().R()
                R = np.array([
                    [rot.get(0, 0), rot.get(0, 1), rot.get(0, 2)],
                    [rot.get(1, 0), rot.get(1, 1), rot.get(1, 2)],
                    [rot.get(2, 0), rot.get(2, 1), rot.get(2, 2)],
                ])
                q_bs = quat_from_rot_matrix(R)
            except Exception:
                # If the component path doesn't exist or API fails, keep identity.
                self.missing_frames.append(frame_path)
                q_bs = np.array([1.0, 0.0, 0.0, 0.0])
            frame_quat[name] = q_bs
        return frame_quat


    # -------------------------------------------------------------------------
    # Apply yaw correction: q_corr = q_yaw(delta) ⊗ q_meas
    # -------------------------------------------------------------------------
    def apply(self, q_meas):
        """Apply current group yaw offsets to measured quaternions (one quaternion per sensor).
        q_meas is expected to be an array of shape (num_sensors, 4) in the same order as sensor_names.

        Apply the current yaw correction offsets to measured quaternions.

        Parameters
        ----------
        q_meas : np.ndarray, shape (N, 4)
            Measured quaternions for all sensors in the same order as sensor_names.

        Returns
        -------
        np.ndarray, shape (N, 4)
            Corrected quaternions q_corr, with group-specific yaw offsets applied.
        """
        q_meas = np.asarray(q_meas, dtype=float)
        q_corr = np.zeros_like(q_meas)
        for name, idx in self.sensor_indices.items():
            group = self._group_for_sensor(name)
            delta = float(self.yaw_offsets.get(group, 0.0))
            q_corr[idx] = quat_normalize(quat_mul(quat_yaw(delta, self.z_hat), quat_normalize(q_meas[idx])))
        return q_corr


    # -------------------------------------------------------------------------
    # Stance detection (walking-only): estimated via foot IMU angular velocity
    # -------------------------------------------------------------------------
    def update_stance(self, q_meas, time_s):
        """Estimate stance from orientation changes of foot IMUs.
        This is a fallback when raw gyro is not available; it approximates angular speed from quaternion deltas.
        """
        for side, foot_name in [('L', 'calcn_l_imu'), ('R', 'calcn_r_imu')]:
            if foot_name not in self.sensor_indices:
                continue
            idx = self.sensor_indices[foot_name]
            q_curr = quat_normalize(q_meas[idx])
            q_prev = self.prev_quat.get(foot_name, q_curr)
            t_prev = self.prev_time.get(foot_name, time_s)
            dt_true = max(time_s - t_prev, 1e-6)

            # Relative rotation from previous to current.
            q_delta = quat_mul(quat_inv(q_prev), q_curr)

            # Rotation vector magnitude approximates rotation angle (radians).
            rot_vec = quat_log(q_delta)
            ang_vel = np.linalg.norm(rot_vec) / dt_true  # rad/s
            self.stance_state[side] = ang_vel < self.stance_gyro_thresh

            self.prev_quat[foot_name] = q_curr
            self.prev_time[foot_name] = time_s


    # -------------------------------------------------------------------------
    # Model sensor orientations: q_model = q_gb ⊗ q_bs
    # -------------------------------------------------------------------------
    def compute_model_quats(self, model, state):
        """Compute model-predicted sensor orientations in ground (q_model).

        For each sensor:
        - query the body orientation in ground q_gb from OpenSim
        - multiply by cached q_bs (body->sensor) rotation

        Returns
        -------
        dict[str, np.ndarray]
            Map: sensor_name -> quaternion [w, x, y, z] in ground frame.
        """
        model_quats = {}
        for name, (body_name, _) in self.sensor_body_map.items():
            if name not in self.sensor_indices:
                continue
            try:
                body = model.getBodySet().get(body_name)
            except Exception:
                continue
            rot = body.getTransformInGround(state).R()
            R = np.array([
                [rot.get(0, 0), rot.get(0, 1), rot.get(0, 2)],
                [rot.get(1, 0), rot.get(1, 1), rot.get(1, 2)],
                [rot.get(2, 0), rot.get(2, 1), rot.get(2, 2)],
            ])
            q_gb = quat_from_rot_matrix(R)

            # Fixed sensor frame orientation relative to body (identity if unknown).
            q_bs = self.sensor_frame_quat.get(name, np.array([1.0, 0.0, 0.0, 0.0]))
            model_quats[name] = quat_mul(q_gb, q_bs)
        return model_quats


    # -------------------------------------------------------------------------
    # Closed-loop update: use IK residual to update yaw offsets slowly
    # -------------------------------------------------------------------------
    def update(self, q_meas, q_corr, q_model, time_s, dt, ik_success=True, residual_metric=None):
        """Update yaw correction state using IK residuals (called after IK).

        Parameters
        ----------
        q_meas : np.ndarray (N,4)
            Measured quaternions (pre-correction)
        q_corr : np.ndarray (N,4)
            Corrected quaternions fed to IK
        q_model : dict[str, np.ndarray]
            Model-predicted sensor orientations from the IK pose
        time_s : float
            Current time in seconds (used to enforce update_hz)
        dt : float
            Time step (seconds). Used for stance estimation and bias integration
        ik_success : bool
            If IK failed, skip update to avoid chasing bad solutions
        residual_metric : float or None
            Optional global residual quality metric from IK. Used for gating

        Behavior
        --------
        1) Rate-limit updates to update_hz (via update_period)
        2) Update stance estimate from foot IMUs
        3) For each group (P/L/R):
           - if L/R, only update during stance for that side
           - compute yaw residuals for each sensor in group
           - use median residual as robust statistic
           - update yaw offset (or yaw-rate bias) with proportional correction
           - clamp yaw offsets to yaw_limit
        """
        # --- Global gating -----------------------------------------------------
        if not ik_success:
            return
        if residual_metric is not None and not np.isfinite(residual_metric):
            return
        if residual_metric is not None and residual_metric > self.residual_skip:
            return
        if (time_s - self.last_update_time) < self.update_period:
            return

        # --- Update stance classification -------------------------------------
        self.update_stance(q_meas, time_s)

        # --- Group-wise correction update -------------------------------------
        for group, sensors in self.sensor_groups.items():
            if group == 'L' and not self.stance_state.get('L', False):
                continue
            if group == 'R' and not self.stance_state.get('R', False):
                continue

            # Collect yaw residuals for sensors in this group.
            e_yaws = []
            for sensor in sensors:
                if sensor not in self.sensor_indices:
                    continue
                idx = self.sensor_indices[sensor]
                if sensor not in q_model:
                    continue

                # Residual: model vs corrected measurement
                q_err = quat_mul(quat_normalize(q_model[sensor]), quat_inv(quat_normalize(q_corr[idx])))

                # Convert residual to a rotation vector in ground frame
                e_vec = quat_log(q_err)

                # Extract yaw component by projection onto global vertical axis
                e_yaw = float(np.dot(e_vec, self.z_hat))
                e_yaws.append(e_yaw)

            if not e_yaws:
                continue

            # Robust residual statistic (median reduces effect of a slipping sensor)
            e_med = float(np.median(e_yaws))

            # Per-group outlier rejection
            if abs(e_med) > self.yaw_outlier:
                continue

            gain = self.pelvis_gain if group == 'P' else self.leg_gain

            if self.use_bias:
                # Experimental: interpret residual as yaw drift signal and update a yaw-rate bias.
                # Note: dividing by dt makes this behave like a rate correction; tune carefully.
                self.yaw_rate_bias[group] += -gain * e_med
                self.yaw_offsets[group] += self.yaw_rate_bias[group] * self.update_period
            else:
                # Yaw-offset controller: delta_yaw <- delta_yaw - k * e_yaw
                # Simple proportional yaw-offset correction (usually the most stable starting point)
                self.yaw_offsets[group] += -gain * e_med

            # Clamp yaw offsets to keep corrections bounded
            self.yaw_offsets[group] = float(
                np.clip(self.yaw_offsets[group], -self.yaw_limit, self.yaw_limit)
            )

        self.last_update_time = time_s


    # -------------------------------------------------------------------------
    # Helper: map a sensor name to its correction group (P/L/R)
    # -------------------------------------------------------------------------
    def _group_for_sensor(self, name):
        """Return correction group for a given sensor name (defaults to 'P')."""
        for group, sensors in self.sensor_groups.items():
            if name in sensors:
                return group
        return 'P'

